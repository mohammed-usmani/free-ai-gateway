import json

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import Catalog, RealtimeChain, Step
from gateway.keypool import KeyPool
from gateway.modes import Modes
from gateway.router import Gateway

AUTH = {"Authorization": "Bearer gw-secret"}


@pytest.fixture
def make_client(upstream_url, tmp_path):
    def make(keys):
        o = {"api_base": upstream_url}
        aliases = {
            "chat": [Step("groq", "m1", o), Step("openrouter", "m2", o)],
            "embed": [Step("openai", "e1", o)],
            "stt": [Step("groq", "whisper", o), Step("openai", "whisper-1", o)],
            "tts": [Step("openai", "tts-1", {**o, "voice": "alloy"})],
        }
        catalog = Catalog(aliases, RealtimeChain([], [], "chat", []))
        return TestClient(create_app(Modes({"free": Gateway(catalog, keys)}), KeyPool(str(tmp_path / "k.db")), "gw-secret"))
    return make


def chat(client, **extra):
    return client.post("/v1/chat/completions", headers=AUTH,
                       json={"model": "chat", "messages": [{"role": "user", "content": "hi"}], **extra})


def test_requests_without_the_gateway_key_are_rejected(make_client):
    client = make_client({"groq": ["g1"]})
    assert client.post("/v1/chat/completions", json={}).status_code == 401
    assert client.post("/v1/chat/completions", json={}, headers={"Authorization": "Bearer nope"}).status_code == 401


def test_chat_returns_openai_shape_and_gateway_headers(make_client):
    r = chat(make_client({"groq": ["rl"], "openrouter": ["o1"]}))
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "m2:o1"
    assert r.headers["x-gateway-model"] == "openrouter/m2"
    assert r.headers["x-gateway-fallbacks"] == "1"


def test_streamed_chat_is_sse_ending_with_done(make_client):
    r = chat(make_client({"groq": ["g1"]}), stream=True)
    lines = [l for l in r.text.splitlines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    text = "".join(json.loads(l[6:])["choices"][0]["delta"].get("content") or "" for l in lines[:-1])
    assert text == "m1:g1"


def test_embeddings(make_client):
    r = make_client({"openai": ["e1"]}).post("/v1/embeddings", headers=AUTH, json={"model": "embed", "input": ["a", "b"]})
    assert r.status_code == 200
    assert [d["index"] for d in r.json()["data"]] == [0, 1]


def test_bad_request_is_passed_through(make_client):
    r = chat(make_client({"groq": ["bad"], "openrouter": ["o1"]}))
    assert r.status_code == 400
    assert "error" in r.json()


def test_exhausted_chain_is_503_with_retry_after_and_no_key_leak(make_client):
    r = chat(make_client({"groq": ["rl"], "openrouter": ["down"]}))
    assert r.status_code == 503
    assert int(r.headers["retry-after"]) > 0
    assert r.json()["error"]["type"] == "all_providers_failed"


def test_unknown_model_is_404(make_client):
    r = make_client({"groq": ["g1"]}).post("/v1/chat/completions", headers=AUTH,
                                            json={"model": "nope", "messages": []})
    assert r.status_code == 404


def test_transcription_falls_back(make_client):
    r = make_client({"groq": ["rl"], "openai": ["o1"]}).post(
        "/v1/audio/transcriptions", headers=AUTH, data={"model": "stt"},
        files={"file": ("a.wav", b"RIFF0000", "audio/wav")})
    assert r.status_code == 200
    assert r.json() == {"text": "whisper-1:o1"}


def test_speech_returns_audio_bytes(make_client):
    r = make_client({"openai": ["o1"]}).post("/v1/audio/speech", headers=AUTH,
                                             json={"model": "tts", "input": "hello"})
    assert r.status_code == 200
    assert r.content.startswith(b"RIFF")


def test_models_lists_usable_aliases(make_client):
    r = make_client({"openrouter": ["o1"]}).get("/v1/models", headers=AUTH)
    assert [m["id"] for m in r.json()["data"]] == ["chat"]


def test_health_masks_keys_and_needs_no_auth(make_client):
    r = make_client({"groq": ["gsk_verysecret_1234"]}).get("/health")
    assert r.status_code == 200
    assert "gsk_verysecret_1234" not in r.text
    assert r.json()["modes"]["free"]["groq"]["keys"] == 1


def test_realtime_requires_the_gateway_key_and_runs_a_session(upstream_url, tmp_path):
    from starlette.websockets import WebSocketDisconnect

    from gateway.realtime.session import RealtimeSession
    from tests.test_realtime_session import FakeBackend, option

    pool = KeyPool(str(tmp_path / "k.db"))
    log = []
    gw = Gateway(Catalog({}, RealtimeChain([], [], "chat", [])), {})
    app = create_app(Modes({"free": gw}), pool, "gw-secret",
                     realtime=lambda ws: RealtimeSession(ws, [option([FakeBackend("a", log)])], pool).run())
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
    with client.websocket_connect("/v1/realtime?model=anything", headers=AUTH) as ws:
        assert ws.receive_json()["type"] == "session.created"


def test_build_reads_env(monkeypatch, tmp_path):
    from gateway.app import build

    monkeypatch.setenv("GATEWAY_API_KEY", "gw-secret")
    monkeypatch.setenv("GROQ_API_KEYS", "g1,g2")
    monkeypatch.setenv("STATE_DB", str(tmp_path / "state.db"))
    monkeypatch.setenv("PRICE_REFRESH_HOURS", "0")
    client = TestClient(build())
    assert client.get("/health").json()["modes"]["free"]["groq"]["keys"] == 2
    assert any(r.path == "/v1/realtime" for r in client.app.routes)


def test_clients_cannot_redirect_or_replace_provider_keys(make_client, upstream_url):
    from tests import fake_upstream

    r = chat(make_client({"groq": ["g1"]}), api_base="http://127.0.0.1:1/v1", api_key="attacker",
             fallbacks=["x"], num_retries=9, mock_response="pwned")
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "m1:g1"
    assert fake_upstream.calls[-1][1] == "g1"


def test_mode_header_and_paid_prefix(upstream_url, tmp_path):
    o = {"api_base": upstream_url}
    free = Catalog({"chat": [Step("groq", "f", o)]}, RealtimeChain([], [], "chat", []))
    paid = Catalog({"chat": [Step("openai", "p", o)]}, RealtimeChain([], [], "chat", []),
                   fallback_to={"chat": "free/chat"})
    modes = Modes({"free": Gateway(free, {"groq": ["g"]}), "paid": Gateway(paid, {"openai": ["down"]})})
    client = TestClient(create_app(modes, KeyPool(str(tmp_path / "k.db")), "gw-secret"))
    r = client.post("/v1/chat/completions", headers=AUTH,
                    json={"model": "paid/chat", "messages": [{"role": "user", "content": "hi"}]})
    assert (r.status_code, r.headers["x-gateway-mode"], r.headers["x-gateway-model"]) == (200, "free", "groq/f")
    assert [m["id"] for m in client.get("/v1/models", headers=AUTH).json()["data"]] == ["chat", "paid/chat"]


@pytest.fixture
def usage_client(upstream_url, tmp_path):
    from gateway.pricing import Pricer
    from gateway.usage import UsageStore

    o = {"api_base": upstream_url}
    catalog = Catalog({"chat": [Step("openai", "m", o)], "embed": [Step("openai", "e1", o)],
                       "stt": [Step("openai", "whisper-1", o)], "tts": [Step("openai", "tts-1", {**o, "voice": "a"})]},
                      RealtimeChain([], [], "chat", []))
    pricer = Pricer({"free": {"openai/m": [{"effective_from": "2026-01-01", "input": 1e-6, "output": 2e-6}]}},
                    bundled={}, fetch=None)
    usage = UsageStore(str(tmp_path / "state.db"), pricer)
    app = create_app(Modes({"free": Gateway(catalog, {"openai": ["k"]})}), KeyPool(str(tmp_path / "k.db")),
                     "gw-secret", usage=usage)
    return TestClient(app), usage


def test_chat_turn_is_recorded_with_tokens_and_cost(usage_client):
    client, usage = usage_client
    r = client.post("/v1/chat/completions", headers={**AUTH, "x-gateway-conversation-id": "c1"},
                    json={"model": "chat", "messages": [{"role": "user", "content": "hi"}]})
    turn = client.get(f"/v1/usage/turns/{r.headers['x-gateway-turn-id']}", headers=AUTH).json()
    comp = turn["components"][0]
    assert (turn["conversation_id"], comp["kind"], comp["provider"], comp["model"]) == ("c1", "chat", "openai", "m")
    assert (comp["input"], comp["output"]) == (1, 1)
    assert comp["cost"] == pytest.approx(1e-6 + 2e-6)


def test_streamed_chat_records_usage_from_the_final_chunk(usage_client):
    client, usage = usage_client
    r = client.post("/v1/chat/completions", headers=AUTH,
                    json={"model": "chat", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    comp = usage.turn(r.headers["x-gateway-turn-id"])["components"][0]
    assert (comp["input"], comp["output"]) == (3, 5)


def test_session_header_attaches_to_the_sessions_turn(usage_client):
    client, usage = usage_client
    t = usage.open_turn("free", "realtime", "realtime", session_id="s9")
    usage.set_session_turn("s9", t)
    r = client.post("/v1/chat/completions", headers={**AUTH, "x-gateway-session": "s9"},
                    json={"model": "chat", "messages": [{"role": "user", "content": "hi"}]})
    assert r.headers["x-gateway-turn-id"] == t
    assert [c["kind"] for c in usage.turn(t)["components"]] == ["llm"]


def test_embeddings_speech_and_transcription_are_recorded(usage_client):
    client, usage = usage_client
    e = client.post("/v1/embeddings", headers=AUTH, json={"model": "embed", "input": "a"})
    s = client.post("/v1/audio/speech", headers=AUTH, json={"model": "tts", "input": "hello"})
    import io, wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000); w.writeframes(b"\x00\x00" * 16000)
    t = client.post("/v1/audio/transcriptions", headers=AUTH, data={"model": "stt"},
                    files={"file": ("a.wav", buf.getvalue(), "audio/wav")})
    comps = {r.headers["x-gateway-turn-id"]: usage.turn(r.headers["x-gateway-turn-id"])["components"][0]
             for r in (e, s, t)}
    assert comps[e.headers["x-gateway-turn-id"]]["input"] == 1
    assert comps[s.headers["x-gateway-turn-id"]]["characters"] == 5
    assert comps[t.headers["x-gateway-turn-id"]]["audio_seconds"] == pytest.approx(2.0)


def test_usage_totals_endpoint(usage_client):
    client, _ = usage_client
    client.post("/v1/chat/completions", headers=AUTH, json={"model": "chat", "messages": [{"role": "user", "content": "x"}]})
    rows = client.get("/v1/usage?group_by=mode", headers=AUTH).json()["data"]
    assert rows[0]["mode"] == "free" and rows[0]["calls"] == 1
    assert client.get("/v1/usage?group_by=bogus", headers=AUTH).status_code == 400
    assert client.get("/v1/usage").status_code == 401


@pytest.fixture
def priced_client(upstream_url, tmp_path):
    from gateway.pricing import Pricer
    from gateway.usage import UsageStore

    o = {"api_base": upstream_url}
    catalog = Catalog({"chat": [Step("openai", "gpt-4o-mini", o)]}, RealtimeChain([], [], "chat", []))
    pricer = Pricer({"free": {"openai/gpt-4o-mini": [{"effective_from": "2026-01-01", "input": 1e-6, "output": 2e-6}]}},
                    bundled={}, fetch=None)
    usage = UsageStore(str(tmp_path / "state.db"), pricer)
    app = create_app(Modes({"free": Gateway(catalog, {"openai": ["k"]})}), KeyPool(str(tmp_path / "k.db")),
                     "gw-secret", usage=usage)
    return TestClient(app), usage


def test_litellm_computed_stream_cost_is_not_taken_as_provider_cost(priced_client):
    client, usage = priced_client
    r = client.post("/v1/chat/completions", headers=AUTH,
                    json={"model": "chat", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    comp = usage.turn(r.headers["x-gateway-turn-id"])["components"][0]
    assert comp["price_source"] == "override"
    assert comp["cost"] == pytest.approx(3 * 1e-6 + 5 * 2e-6)


def test_usage_recording_failure_does_not_fail_the_request(priced_client, monkeypatch):
    client, usage = priced_client

    def boom(*a, **k):
        raise KeyError("effective_from")

    monkeypatch.setattr(usage, "add", boom)
    r = client.post("/v1/chat/completions", headers=AUTH,
                    json={"model": "chat", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200


async def test_interrupted_stream_records_estimated_usage():
    from gateway.app import stream_and_record

    class Chunk:
        def __init__(self, text):
            self.usage = None
            self.choices = [type("C", (), {"delta": type("D", (), {"content": text})()})()]

        def model_dump_json(self):
            return "{}"

    closed = []

    class Upstream:
        def __init__(self):
            self._it = iter([Chunk("hello "), Chunk("world"), Chunk("!")])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration

        async def aclose(self):
            closed.append(True)

    recorded = []
    gen = stream_and_record(Upstream(), lambda usage, text: recorded.append((usage, text)), scrub=str)
    await gen.__anext__()
    await gen.aclose()  # client went away after the first chunk
    assert recorded == [(None, "hello ")]
    assert closed == [True]
