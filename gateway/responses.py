"""Stored conversations for the Responses API.

Only when a request has `store` true (OpenAI's default): the gateway keeps the input and
output items so `previous_response_id` works with any provider, and calls providers with
store=false. Rows older than the retention window are purged; DELETE removes one."""

import json
import sqlite3
import threading
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (id TEXT PRIMARY KEY, created_at REAL, input_items TEXT,
  output_items TEXT, response TEXT);
CREATE INDEX IF NOT EXISTS responses_created ON responses(created_at);
"""


class ResponseStore:
    def __init__(self, path: str, clock=time.time, retention_days: float = 30):
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._clock = clock
        self._retention = retention_days * 86400
        self._lock = threading.Lock()

    def save(self, response_id: str, input_items: list, output_items: list, response: dict):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?)",
                             (response_id, self._clock(), json.dumps(input_items), json.dumps(output_items),
                              json.dumps(response, default=str)))
        self.purge()

    def get(self, response_id: str) -> dict | None:
        row = self._db.execute("SELECT * FROM responses WHERE id = ? AND created_at >= ?",
                               (response_id, self._clock() - self._retention)).fetchone()
        if not row:
            return None
        return {"input_items": json.loads(row[2]), "output_items": json.loads(row[3]), "response": json.loads(row[4])}

    def history(self, response_id: str) -> list | None:
        """Everything said up to and including that response, as input items."""
        stored = self.get(response_id)
        return None if stored is None else stored["input_items"] + stored["output_items"]

    def delete(self, response_id: str) -> bool:
        with self._lock:
            return self._db.execute("DELETE FROM responses WHERE id = ?", (response_id,)).rowcount > 0

    def purge(self):
        with self._lock:
            self._db.execute("DELETE FROM responses WHERE created_at < ?", (self._clock() - self._retention,))


def as_items(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    return list(value)
