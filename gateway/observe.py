"""One LiteLLM callback for every provider attempt: logs it for metrics, and benches keys
that hit a daily quota until the reset (persisted, so a restart doesn't forget)."""

import asyncio
import hashlib
import logging
import time
from collections.abc import Callable

import litellm
from litellm.integrations.custom_logger import CustomLogger

from gateway.keypool import KeyPool
from gateway.quota import daily_reset_seconds

log = logging.getLogger(__name__)


def key_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:12]


class _Hook(CustomLogger):
    """LiteLLM picks up callbacks once, so one process-wide hook fans out to every Observer."""

    def __init__(self):
        super().__init__()
        self.observers: list["Observer"] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        for o in list(self.observers):
            await o.on_success(kwargs, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        for o in list(self.observers):
            await o.on_failure(kwargs, start_time, end_time)


_HOOK = _Hook()
litellm.callbacks.append(_HOOK)


class Observer:
    def __init__(self, pool: KeyPool, sink: Callable[[dict], None] | None = None):
        self.pool = pool
        self.sink = sink
        self._deployments: dict[str, tuple] = {}  # model_id → (router, mode, group, provider, key)
        self._inflight = 0
        _HOOK.observers.append(self)

    def close(self):
        if self in _HOOK.observers:
            _HOOK.observers.remove(self)

    @property
    def pending(self) -> int:
        return self._inflight

    def track(self, router, mode: str):
        """Remember a router's deployments and re-apply daily cooldowns saved before a restart."""
        for d in router.model_list:
            model_id = d["model_info"]["id"]
            params = d["litellm_params"]
            provider = params["model"].split("/", 1)[0]
            key = params.get("api_key") or ""
            self._deployments[model_id] = (router, mode, d["model_name"], provider, key)
            left = self.pool.remaining(f"quota:{mode}:{d['model_name']}", key)
            if left > 0:
                router.cooldown_cache.add_deployment_to_cooldown(model_id, Exception("daily quota"), 429, left)

    def skipped(self, mode: str, group: str, reason: str):
        self._write({"at": time.time(), "mode": mode, "provider": group.split("/", 1)[0], "model": group.split("/", 1)[-1],
                     "key": None, "ok": False, "status": None, "error": f"skipped: {reason}", "latency_ms": 0,
                     "cache_hit": False})

    def _entry(self, kwargs):
        params = kwargs.get("litellm_params") or {}
        return self._deployments.get((params.get("model_info") or {}).get("id"))

    def _write(self, row: dict):
        if self.sink:
            try:
                self.sink(row)
            except Exception:  # noqa: BLE001 — metrics must never break a request
                log.exception("could not log an attempt")

    def _row(self, entry, ok, start, end, exc=None, cache_hit=False) -> dict:
        _, mode, group, provider, key = entry
        return {"at": time.time(), "mode": mode, "provider": provider, "model": group.split("/", 1)[-1],
                "key": key_id(key) if key else None, "ok": ok, "status": getattr(exc, "status_code", None),
                "error": type(exc).__name__ if exc else None,
                "latency_ms": int((end - start).total_seconds() * 1000) if start and end else 0, "cache_hit": cache_hit}

    async def on_success(self, kwargs, start_time, end_time):
        if entry := self._entry(kwargs):
            self._write(self._row(entry, True, start_time, end_time, cache_hit=bool(kwargs.get("cache_hit"))))

    async def on_failure(self, kwargs, start_time, end_time):
        entry = self._entry(kwargs)
        if not entry:
            return
        self._inflight += 1
        try:
            exc = kwargs.get("exception")
            self._write(self._row(entry, False, start_time, end_time, exc))
            router, mode, group, provider, key = entry
            seconds = daily_reset_seconds(exc, provider)
            if seconds:
                model_id = (kwargs.get("litellm_params") or {}).get("model_info", {}).get("id")
                router.cooldown_cache.add_deployment_to_cooldown(model_id, exc, 429, seconds)
                self.pool.report(f"quota:{mode}:{group}", key, 429, retry_after=seconds)
                log.warning("%s key …%s hit its daily quota; benched for %.0f min", group, key[-4:], seconds / 60)
        except Exception:  # noqa: BLE001
            log.exception("quota classification failed")
        finally:
            self._inflight -= 1
            await asyncio.sleep(0)
