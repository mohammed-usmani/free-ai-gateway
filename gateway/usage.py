"""Usage per turn: one turn per caller request (or realtime response), made of priced
components. Raw counts and the unit prices used are stored, so cost can be recomputed."""

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, fields
from datetime import datetime, timezone

from gateway.pricing import Counts, Pricer, cost

BUCKETS = [f.name for f in fields(Counts)]
GROUPS = {
    "model": ["c.provider", "c.model"], "provider": ["c.provider"], "mode": ["c.mode"], "alias": ["t.alias"],
    "endpoint": ["t.endpoint"], "kind": ["c.kind"], "day": ["date(t.started_at, 'unixepoch')"],
}

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS turns (turn_id TEXT PRIMARY KEY, conversation_id TEXT, session_id TEXT, mode TEXT,
  endpoint TEXT, alias TEXT, started_at REAL, ended_at REAL, cost REAL, cost_complete INTEGER);
CREATE INDEX IF NOT EXISTS turns_started ON turns(started_at);
CREATE INDEX IF NOT EXISTS turns_conversation ON turns(conversation_id);
CREATE TABLE IF NOT EXISTS components (id INTEGER PRIMARY KEY, turn_id TEXT, kind TEXT, mode TEXT, provider TEXT,
  model TEXT, attempt TEXT, {", ".join(f"{b} REAL" for b in BUCKETS)}, provider_cost REAL, unit_prices TEXT,
  price_source TEXT, cost REAL, latency_ms INTEGER, at REAL, fallbacks INTEGER);
CREATE INDEX IF NOT EXISTS components_turn ON components(turn_id);
CREATE TABLE IF NOT EXISTS attempts (at REAL, mode TEXT, provider TEXT, model TEXT, key TEXT, ok INTEGER,
  status INTEGER, error TEXT, latency_ms INTEGER, cache_hit INTEGER);
CREATE INDEX IF NOT EXISTS attempts_at ON attempts(at);
"""


class UsageStore:
    def __init__(self, path: str, pricer: Pricer, clock=time.time):
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        have = {r[1] for r in self._db.execute("PRAGMA table_info(components)")}
        for column in [*BUCKETS, "provider_cost", "unit_prices", "price_source", "cost", "latency_ms", "at", "fallbacks", "attempt",
                       "mode", "kind", "provider", "model"]:
            if column not in have:  # databases from an older version
                kind = "TEXT" if column in ("unit_prices", "price_source", "attempt", "mode", "kind", "provider", "model") else "REAL"
                self._db.execute(f"ALTER TABLE components ADD COLUMN {column} {kind}")
        self._lock = threading.Lock()
        self.pricer = pricer
        self._clock = clock
        self._sessions: dict[str, str] = {}  # session token → current turn
        self._session_info: dict[str, tuple[str, str]] = {}  # session token → (session id, mode)

    def _now(self) -> float:
        return self._clock()

    def open_turn(self, mode, endpoint, alias, conversation_id=None, session_id=None) -> str:
        turn_id = f"turn_{uuid.uuid4().hex[:24]}"
        with self._lock:
            self._db.execute("INSERT INTO turns VALUES (?,?,?,?,?,?,?,NULL,NULL,NULL)",
                             (turn_id, conversation_id, session_id, mode, endpoint, alias, self._now()))
        return turn_id

    def add(self, turn_id, kind, mode, provider, model, counts: Counts, attempt="ok",
            latency_ms=None, provider_cost=None, fallbacks=0) -> dict:
        at = self._now()
        price = self.pricer.price(provider, model, mode, datetime.fromtimestamp(at, timezone.utc))
        if provider_cost is not None:
            value, source = float(provider_cost), "provider"
        elif not any(asdict(counts).values()):
            value, source = None, price.source if price else None  # nothing measured: unknown, not free
        else:
            value, source = cost(counts, price), price.source if price else None
        row = {"turn_id": turn_id, "kind": kind, "mode": mode, "provider": provider, "model": model,
               "attempt": attempt, **asdict(counts), "provider_cost": provider_cost,
               "unit_prices": json.dumps(price.units) if price else None, "price_source": source,
               "cost": value, "latency_ms": latency_ms, "at": at, "fallbacks": fallbacks}
        with self._lock:
            self._db.execute(f"INSERT INTO components ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                             list(row.values()))
            self._refresh_turn(turn_id)
        return row

    def _refresh_turn(self, turn_id):
        total, missing, n = self._db.execute(
            "SELECT SUM(cost), SUM(cost IS NULL), COUNT(*) FROM components WHERE turn_id = ?", (turn_id,)).fetchone()
        # Nothing recorded is unknown, not free.
        complete = n > 0 and not missing
        self._db.execute("UPDATE turns SET cost = ?, cost_complete = ? WHERE turn_id = ?",
                         (total if n else None, int(complete), turn_id))

    def close_turn(self, turn_id):
        with self._lock:
            self._db.execute("UPDATE turns SET ended_at = ? WHERE turn_id = ?", (self._now(), turn_id))
            self._refresh_turn(turn_id)

    def log_attempt(self, row: dict):
        """One provider attempt (success, failure or skipped step), for /v1/metrics."""
        with self._lock:
            self._db.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (row["at"], row["mode"], row["provider"], row["model"], row["key"], int(row["ok"]),
                              row["status"], row["error"], row["latency_ms"], int(row["cache_hit"])))

    def metrics(self, window_seconds: float) -> dict:
        """Reliability per provider/model (from every attempt) and fallback rate per alias."""
        since = self._now() - window_seconds
        rows = self._db.execute("SELECT * FROM attempts WHERE at >= ?", (since,)).fetchall()
        groups: dict[tuple, list] = {}
        for r in rows:
            groups.setdefault((r["mode"], r["provider"], r["model"]), []).append(r)
        providers = []
        for (mode, provider, model), items in sorted(groups.items()):
            tried = [r for r in items if not (r["error"] or "").startswith("skipped")]
            ok = [r for r in tried if r["ok"]]
            lat = sorted(r["latency_ms"] for r in ok)
            errors: dict[str, int] = {}
            for r in tried:
                if not r["ok"]:
                    errors[r["error"] or "unknown"] = errors.get(r["error"] or "unknown", 0) + 1
            providers.append({
                "mode": mode, "provider": provider, "model": model, "requests": len(tried),
                "skipped": len(items) - len(tried), "success_rate": len(ok) / len(tried) if tried else None,
                "p50_ms": _pct(lat, 0.5), "p95_ms": _pct(lat, 0.95), "errors": errors,
                "cache_hit_rate": sum(r["cache_hit"] for r in ok) / len(ok) if ok else None,
            })
        aliases = [dict(r) for r in self._db.execute(
            "SELECT t.mode, t.alias, COUNT(DISTINCT t.turn_id) AS turns, "
            "AVG(CASE WHEN COALESCE(c.fallbacks, 0) > 0 THEN 1.0 ELSE 0.0 END) AS fallback_rate "
            "FROM turns t JOIN components c USING (turn_id) WHERE t.started_at >= ? GROUP BY t.mode, t.alias "
            "ORDER BY t.mode, t.alias", (since,)).fetchall()]
        return {"window_seconds": window_seconds, "providers": providers, "aliases": aliases}

    # --- realtime sessions: the cascade's LLM call lands on the session's current turn ---
    # Keyed by a private per-session token (sent only on the loopback call), not the session
    # id the client sees, so no other caller can attach usage to a session.

    def register_session(self, token: str, session_id: str, mode: str):
        self._session_info[token] = (session_id, mode)

    def unregister_session(self, token: str):
        self._session_info.pop(token, None)
        self._sessions.pop(token, None)

    def session_of(self, token: str) -> tuple[str, str] | None:
        return self._session_info.get(token)

    def set_session_turn(self, token: str, turn_id: str | None):
        if turn_id is None:
            self._sessions.pop(token, None)
        else:
            self._sessions[token] = turn_id

    def session_turn(self, token: str) -> str | None:
        return self._sessions.get(token)

    # --- reads ---

    def turn(self, turn_id: str) -> dict | None:
        t = self._db.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
        if not t:
            return None
        comps = self._db.execute("SELECT * FROM components WHERE turn_id = ? ORDER BY id", (turn_id,)).fetchall()
        return {**_turn_dict(t), "components": [_component_dict(c) for c in comps]}

    def turns(self, conversation_id: str | None = None, limit: int = 50) -> list[dict]:
        sql, args = "SELECT turn_id FROM turns", []
        if conversation_id:
            sql, args = sql + " WHERE conversation_id = ?", [conversation_id]
        rows = self._db.execute(sql + " ORDER BY started_at DESC LIMIT ?", [*args, limit]).fetchall()
        return [self.turn(r["turn_id"]) for r in rows]

    def totals(self, start: float | None, end: float | None, group_by: str) -> list[dict]:
        if group_by not in GROUPS:
            raise ValueError(f"group_by must be one of {', '.join(GROUPS)}")
        cols = GROUPS[group_by]
        names = [c.split(".")[-1] if "(" not in c else "day" for c in cols]
        sums = ", ".join(f"SUM(c.{b}) AS {b}" for b in BUCKETS)
        sql = (f"SELECT {', '.join(f'{c} AS {n}' for c, n in zip(cols, names))}, COUNT(DISTINCT t.turn_id) AS turns, "
               f"COUNT(*) AS calls, {sums}, SUM(c.cost) AS cost, "
               f"SUM(CASE WHEN c.mode = 'paid' THEN c.cost END) AS billed_cost, "
               f"SUM(CASE WHEN c.mode != 'paid' THEN c.cost END) AS list_price_estimate, "
               f"SUM(c.cost IS NULL) AS unpriced_calls "
               f"FROM components c JOIN turns t USING (turn_id) WHERE t.started_at >= ? AND t.started_at < ? "
               f"GROUP BY {', '.join(cols)} ORDER BY {', '.join(cols)}")
        rows = self._db.execute(sql, (start or 0, end or 1e12)).fetchall()
        return [_numbers(dict(r)) for r in rows]

    def reprice(self, start: float | None, end: float | None) -> int:
        """Recompute component cost from stored counts with today's prices (provider-reported
        costs are kept). Returns how many components were repriced."""
        rows = self._db.execute(
            "SELECT c.* FROM components c JOIN turns t USING (turn_id) WHERE t.started_at >= ? AND t.started_at < ? "
            "AND c.provider_cost IS NULL", (start or 0, end or 1e12)).fetchall()
        with self._lock:
            for r in rows:
                counts = Counts(**{b: (r[b] or 0) for b in BUCKETS})
                price = self.pricer.price(r["provider"], r["model"], r["mode"], datetime.fromtimestamp(r["at"], timezone.utc))
                value = cost(counts, price) if any(asdict(counts).values()) else None
                self._db.execute("UPDATE components SET cost = ?, unit_prices = ?, price_source = ? WHERE id = ?",
                                 (value, json.dumps(price.units) if price else None, price.source if price else None, r["id"]))
            for turn_id in {r["turn_id"] for r in rows}:
                self._refresh_turn(turn_id)
        return len(rows)


def _pct(values: list, q: float):
    if not values:
        return None
    return values[min(len(values) - 1, max(0, round(q * len(values) + 0.5) - 1))]


def _numbers(d: dict) -> dict:
    for b in BUCKETS:
        if b in d and d[b] is not None:
            d[b] = float(d[b]) if b == "audio_seconds" else int(d[b])
    return d


def _basis(mode: str | None) -> str:
    return "billed" if mode == "paid" else "list_price_estimate"  # free mode: what it would have cost


def _turn_dict(t) -> dict:
    d = dict(t)
    d["cost_basis"] = _basis(d.get("mode"))
    d["cost_complete"] = None if d["cost_complete"] is None else bool(d["cost_complete"])
    return d


def _component_dict(c) -> dict:
    d = _numbers(dict(c))
    d["cost_basis"] = _basis(d.get("mode"))
    d["unit_prices"] = json.loads(d["unit_prices"]) if d["unit_prices"] else None
    return d
