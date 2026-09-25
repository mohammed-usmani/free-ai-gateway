import asyncio
import base64

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from gateway.keypool import KeyPool
from gateway.realtime.session import RealtimeSession


class FakeBackend:
    def __init__(self, name, log, fail_on_start=False):
        self.name, self.log, self.fail_on_start = name, log, fail_on_start
        self.failed = asyncio.Event()
        self.failure = None
        self.emit = None

    async def start(self, config, emit):
        if self.fail_on_start:
            self.failure = (self.name, f"key-{self.name}", 401)
            raise RuntimeError("bad key")
        self.emit = emit
        self.log.append(("start", self.name, dict(config)))

    async def update(self, config):
        self.log.append(("update", self.name, dict(config)))

    async def send_audio(self, pcm):
        self.log.append(("audio", self.name, pcm))
        if pcm == b"die":
            self.failure = (self.name, f"key-{self.name}", 429)
            self.failed.set()
            return
        await self.emit({"type": "response.output_audio.delta", "delta": base64.b64encode(self.name.encode()).decode()})

    async def send_text(self, text):
        self.log.append(("text", self.name, text))

    async def function_result(self, call_id, output):
        self.log.append(("tool", self.name, call_id, output))

    async def respond(self):
        self.log.append(("respond", self.name))

    async def cancel(self):
        self.log.append(("cancel", self.name))

    async def stop(self):
        self.log.append(("stop", self.name))


def app_with(options, pool):
    app = FastAPI()

    @app.websocket("/rt")
    async def rt(ws: WebSocket):
        await RealtimeSession(ws, options, pool).run()

    return app


@pytest.fixture
def pool(tmp_path):
    return KeyPool(str(tmp_path / "k.db"))


def option(backends):
    """An option that hands out the given backends in turn, then None (no usable key)."""
    it = iter(backends)
    return lambda: next(it, None)


def audio(pcm):
    return {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}


def next_of(ws, kind):
    while True:
        event = ws.receive_json()
        if event["type"] == kind:
            return event


def test_session_config_is_kept_and_audio_reaches_the_backend(pool):
    log = []
    client = TestClient(app_with([option([FakeBackend("a", log)])], pool))
    with client.websocket_connect("/rt") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({"type": "session.update", "session": {"instructions": "be brief"}})
        assert next_of(ws, "session.updated")["session"]["instructions"] == "be brief"
        ws.send_json(audio(b"hello"))
        assert base64.b64decode(next_of(ws, "response.output_audio.delta")["delta"]) == b"a"
    assert ("start", "a", {"instructions": "be brief"}) in log
    assert ("audio", "a", b"hello") in log
    assert log[-1] == ("stop", "a")


def test_mid_session_failure_moves_to_next_backend_with_same_config(pool):
    log = []
    options = [option([FakeBackend("a", log)]), option([FakeBackend("b", log)])]
    with TestClient(app_with(options, pool)).websocket_connect("/rt") as ws:
        ws.receive_json()
        ws.send_json({"type": "session.update", "session": {"instructions": "x", "tools": [{"name": "t"}]}})
        ws.send_json(audio(b"hi"))
        next_of(ws, "response.output_audio.delta")
        ws.send_json(audio(b"die"))
        ws.send_json(audio(b"again"))
        assert base64.b64decode(next_of(ws, "response.output_audio.delta")["delta"]) == b"b"
    assert ("start", "b", {"instructions": "x", "tools": [{"name": "t"}]}) in log
    assert ("stop", "a") in log
    assert pool.pick("a", ["key-a"]) is None  # the failing key is cooling down


def test_backend_failing_to_start_is_skipped(pool):
    log = []
    options = [option([FakeBackend("a", log, fail_on_start=True)]), option([FakeBackend("b", log)])]
    with TestClient(app_with(options, pool)).websocket_connect("/rt") as ws:
        ws.receive_json()
        ws.send_json(audio(b"hi"))
        assert base64.b64decode(next_of(ws, "response.output_audio.delta")["delta"]) == b"b"


def test_same_option_is_retried_with_its_next_key_first(pool):
    log = []
    options = [option([FakeBackend("a1", log), FakeBackend("a2", log)]), option([FakeBackend("b", log)])]
    with TestClient(app_with(options, pool)).websocket_connect("/rt") as ws:
        ws.receive_json()
        ws.send_json(audio(b"die"))
        ws.send_json(audio(b"x"))
        assert base64.b64decode(next_of(ws, "response.output_audio.delta")["delta"]) == b"a2"


def test_no_backend_left_sends_error_and_closes(pool):
    with TestClient(app_with([option([])], pool)).websocket_connect("/rt") as ws:
        ws.receive_json()
        ws.send_json(audio(b"hi"))
        assert next_of(ws, "error")["error"]["code"] == "no_backend_available"


def test_text_tools_and_response_control_are_forwarded(pool):
    log = []
    with TestClient(app_with([option([FakeBackend("a", log)])], pool)).websocket_connect("/rt") as ws:
        ws.receive_json()
        ws.send_json({"type": "conversation.item.create",
                      "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hey"}]}})
        ws.send_json({"type": "conversation.item.create",
                      "item": {"type": "function_call_output", "call_id": "c1", "output": "42"}})
        ws.send_json({"type": "response.create"})
        ws.send_json({"type": "response.cancel"})
        ws.send_json({"type": "session.update", "session": {"voice": "v"}})
        next_of(ws, "session.updated")
    assert [e for e in log if e[0] != "start"][:5] == [
        ("text", "a", "hey"), ("tool", "a", "c1", "42"), ("respond", "a"), ("cancel", "a"),
        ("update", "a", {"voice": "v"})]


def test_unknown_events_get_a_non_fatal_error(pool):
    with TestClient(app_with([option([])], pool)).websocket_connect("/rt") as ws:
        ws.receive_json()
        ws.send_json({"type": "bogus.event"})
        assert next_of(ws, "error")["error"]["type"] == "invalid_request_error"


class BrokenOnStart(FakeBackend):
    """Like a real PipecatBackend whose build raises: no failure recorded, no await point."""

    async def start(self, config, emit):
        raise ValueError("no adapter")


def test_backend_that_cannot_start_is_not_retried_forever(pool):
    options = [lambda: BrokenOnStart("x", []), option([FakeBackend("b", [])])]
    with TestClient(app_with(options, pool)).websocket_connect("/rt") as ws:
        ws.receive_json()
        ws.send_json(audio(b"hi"))
        assert base64.b64decode(next_of(ws, "response.output_audio.delta")["delta"]) == b"b"


class UpdateFails(FakeBackend):
    async def update(self, config):
        raise KeyError("name")


def test_update_failure_fails_over_instead_of_closing(pool):
    log = []
    options = [option([UpdateFails("a", log)]), option([FakeBackend("b", log)])]
    with TestClient(app_with(options, pool)).websocket_connect("/rt") as ws:
        ws.receive_json()
        ws.send_json(audio(b"hi"))
        next_of(ws, "response.output_audio.delta")
        ws.send_json({"type": "session.update", "session": {"tools": [{}]}})
        ws.send_json(audio(b"again"))
        assert base64.b64decode(next_of(ws, "response.output_audio.delta")["delta"]) == b"b"


def test_malformed_client_messages_get_an_error_event(pool):
    with TestClient(app_with([option([FakeBackend("a", [])])], pool)).websocket_connect("/rt") as ws:
        ws.receive_json()
        ws.send_text("not json")
        assert next_of(ws, "error")["error"]["type"] == "invalid_request_error"
        ws.send_json({"type": "input_audio_buffer.append", "audio": "@@not-base64@@"})
        assert next_of(ws, "error")["error"]["type"] == "invalid_request_error"
        ws.send_json(audio(b"ok"))
        next_of(ws, "response.output_audio.delta")
