"""Key health for the realtime voice backends, kept in SQLite so it survives restarts.

(Chat, embeddings and batch audio use LiteLLM's own cooldowns.)
"""

import hashlib
import sqlite3
import threading
import time
from collections.abc import Callable

AUTH_COOLDOWN = 24 * 3600
RATE_LIMIT_COOLDOWN = 60
ERROR_COOLDOWN = 15


def _key_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def mask(key: str) -> str:
    return f"{key[:3]}…{key[-4:]}" if len(key) > 8 else "…"


def cooldown_for(status: int | None, retry_after: float | None) -> float:
    if status in (401, 403):
        return AUTH_COOLDOWN
    if status == 429:
        return retry_after or RATE_LIMIT_COOLDOWN
    return ERROR_COOLDOWN


class KeyPool:
    def __init__(self, path: str, clock: Callable[[], float] = time.time):
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cooldowns (provider TEXT, key_id TEXT, until REAL, "
            "status INTEGER, PRIMARY KEY (provider, key_id))"
        )
        self._clock = clock
        self._cursor: dict[str, int] = {}
        self._lock = threading.Lock()

    def _until(self, provider: str, keys: list[str]) -> dict[str, float]:
        rows = self._db.execute(
            "SELECT key_id, until FROM cooldowns WHERE provider = ?", (provider,)
        ).fetchall()
        by_id = dict(rows)
        return {k: by_id.get(_key_id(k), 0.0) for k in keys}

    def pick(self, provider: str, keys: list[str]) -> str | None:
        """Next usable key, round-robin; None when every key is cooling down."""
        if not keys:
            return None
        now = self._clock()
        until = self._until(provider, keys)
        with self._lock:
            start = self._cursor.get(provider, 0)
            for i in range(len(keys)):
                key = keys[(start + i) % len(keys)]
                if until[key] <= now:
                    self._cursor[provider] = (start + i + 1) % len(keys)
                    return key
        return None

    def report(self, provider: str, key: str, status: int | None, retry_after: float | None = None):
        """Record a call's outcome. 2xx clears a cooldown; anything else starts one."""
        if status is not None and 200 <= status < 300:
            self._db.execute(
                "DELETE FROM cooldowns WHERE provider = ? AND key_id = ?", (provider, _key_id(key))
            )
            return
        until = self._clock() + cooldown_for(status, retry_after)
        self._db.execute(
            "INSERT OR REPLACE INTO cooldowns VALUES (?, ?, ?, ?)",
            (provider, _key_id(key), until, status),
        )

    def soonest(self, provider: str, keys: list[str]) -> float | None:
        """Seconds until the first cooling key recovers (0 if one is usable now)."""
        if not keys:
            return None
        now = self._clock()
        return max(0.0, min(self._until(provider, keys).values()) - now)

    def states(self, provider: str, keys: list[str]) -> list[dict]:
        now = self._clock()
        return [
            {"key": mask(k), "state": "cooling" if u > now else "ok", "seconds_left": round(max(0, u - now))}
            for k, u in self._until(provider, keys).items()
        ]
