"""Server side of the OpenAI Realtime protocol, with backend failover mid-session.

A backend is one way of answering (Gemini Live, or the STT→LLM→TTS cascade). An option
builds a backend with a fresh key, or returns None when it has no usable key. When the
active backend fails, its key is cooled down and the same option is asked again (next
key), then the next option. The client's session config is re-applied; conversation
history is not carried over.
"""

import asyncio
import base64
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Protocol

from fastapi import WebSocket, WebSocketDisconnect

from gateway.keypool import KeyPool

log = logging.getLogger(__name__)

Emit = Callable[[dict], Awaitable[None]]


class Backend(Protocol):
    name: str
    failed: asyncio.Event
    failure: tuple[str, str, int | None] | None  # provider, key, status

    async def start(self, config: dict, emit: Emit) -> None: ...
    async def update(self, config: dict) -> None: ...
    async def send_audio(self, pcm16_24k: bytes) -> None: ...
    async def send_text(self, text: str) -> None: ...
    async def function_result(self, call_id: str, output: str) -> None: ...
    async def respond(self) -> None: ...
    async def cancel(self) -> None: ...
    async def stop(self) -> None: ...


BackendOption = Callable[[], Backend | None]


def event(kind: str, **fields) -> dict:
    return {"type": kind, "event_id": f"event_{uuid.uuid4().hex[:20]}", **fields}


def error_event(message: str, kind: str = "invalid_request_error", code: str | None = None) -> dict:
    return event("error", error={"type": kind, "code": code, "message": message})


class NoBackend(Exception):
    pass


class RealtimeSession:
    def __init__(self, ws: WebSocket, options, pool: KeyPool, session_id: str | None = None,
                 initial_config: dict | None = None):
        """`options` is a list, or a function of the session config returning one (resolved when
        the first backend starts, so `session.type: "transcription"` can pick STT-only)."""
        self.ws = ws
        self.session_id = session_id or f"sess_{uuid.uuid4().hex[:20]}"
        self.options = options
        self.pool = pool
        self.config: dict = dict(initial_config or {})
        self.backend: Backend | None = None
        self._option = 0
        self._lock = asyncio.Lock()
        self._watch: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()

    async def send(self, payload: dict):
        async with self._send_lock:
            await self.ws.send_json(payload)

    async def run(self):
        await self.ws.accept()
        await self.send(event("session.created", session={"id": self.session_id, **self.config}))
        try:
            while True:
                raw = await self.ws.receive_text()
                try:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict):
                        raise ValueError("expected a JSON object")
                    await self._handle(msg)
                except ValueError as exc:  # bad JSON, bad base64 (binascii.Error is a ValueError)
                    await self.send(error_event(f"malformed event: {exc}"))
        except (WebSocketDisconnect, NoBackend):
            pass
        finally:
            if self._watch:  # it may be mid-swap: let it finish unwinding before stopping
                self._watch.cancel()
                try:
                    await self._watch
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            if self.backend:
                await self.backend.stop()

    async def _handle(self, msg: dict):
        kind = msg.get("type")
        if kind == "session.update":
            self.config.update(msg.get("session") or {})
            async with self._lock:  # never restart a backend that failover is retiring
                backend = self.backend
                if backend and not backend.failed.is_set():
                    try:
                        await backend.update(self.config)
                    except Exception:  # noqa: BLE001 — a backend that can't take the config has failed
                        log.exception("realtime backend %s rejected session.update", backend.name)
                        backend.failed.set()
            await self.send(event("session.updated", session=self.config))
        elif kind == "input_audio_buffer.append":
            pcm = base64.b64decode(msg.get("audio", ""), validate=True)
            await (await self._active()).send_audio(pcm)
        elif kind == "input_audio_buffer.commit":
            await self.send(event("input_audio_buffer.committed"))
        elif kind == "input_audio_buffer.clear":
            await self.send(event("input_audio_buffer.cleared"))
        elif kind == "conversation.item.create":
            await self._item(msg.get("item") or {})
        elif kind == "response.create":
            await (await self._active()).respond()
        elif kind == "response.cancel":
            async with self._lock:
                if self.backend and not self.backend.failed.is_set():
                    await self.backend.cancel()
        else:
            await self.send(error_event(f"unsupported event type {kind!r}"))

    async def _item(self, item: dict):
        backend = await self._active()
        if item.get("type") == "function_call_output":
            await backend.function_result(item.get("call_id", ""), item.get("output", ""))
        elif item.get("type") == "message":
            text = " ".join(c.get("text", "") for c in item.get("content", []) if c.get("type") in ("input_text", "text"))
            await backend.send_text(text)
        else:
            await self.send(error_event(f"unsupported item type {item.get('type')!r}"))

    async def _active(self) -> Backend:
        async with self._lock:
            if self.backend and self.backend.failed.is_set():
                await self._retire(self.backend)
            if not self.backend:
                await self._start_next()
            return self.backend

    async def _retire(self, backend: Backend):
        if backend.failure:
            provider, key, status = backend.failure
            self.pool.report(provider, key, status)
        log.warning("realtime backend %s failed: %s", backend.name, backend.failure and backend.failure[2])
        await backend.stop()
        if self.backend is backend:
            self.backend = None

    async def _start_next(self):
        if callable(self.options):
            self.options = self.options(self.config)
        while self._option < len(self.options):
            backend = self.options[self._option]()
            if backend is None:
                self._option += 1
                continue
            try:
                await backend.start(self.config, self.send)
            except Exception:  # noqa: BLE001 — any start failure means try the next option
                log.exception("realtime backend %s failed to start", backend.name)
                await self._retire(backend)
                # A start failure is usually config or code, not the key: re-asking the same
                # option would spin. (Key failures surface after start, via `failed`.)
                self._option += 1
                continue
            self.backend = backend
            self._watch = asyncio.create_task(self._on_failure(backend))
            return
        await self.send(error_event("no realtime backend has a usable key", "server_error", "no_backend_available"))
        await self.ws.close(code=1011)
        raise NoBackend

    async def _on_failure(self, backend: Backend):
        """Swap as soon as a backend fails, not only on the next client event."""
        await backend.failed.wait()
        async with self._lock:
            if self.backend is backend:
                await self._retire(backend)
                try:
                    await self._start_next()
                except NoBackend:
                    pass
                else:
                    await self.send(event("session.updated", session=self.config))
