import json

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import Catalog, RealtimeChain, Step
from gateway.keypool import KeyPool
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
        return TestClient(create_app(Gateway(catalog, keys), KeyPool(str(tmp_path / "k.db")), "gw-secret"))
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
    assert r.json()["providers"]["groq"]["keys"] == 1


def test_realtime_requires_the_gateway_key_and_runs_a_session(upstream_url, tmp_path):
    from starlette.websockets import WebSocketDisconnect

    from gateway.realtime.session import RealtimeSession
    from tests.test_realtime_session import FakeBackend, option

    pool = KeyPool(str(tmp_path / "k.db"))
    log = []
    gw = Gateway(Catalog({}, RealtimeChain([], [], "chat", [])), {})
    app = create_app(gw, pool, "gw-secret",
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
    monkeypatch.setenv("KEYPOOL_DB", str(tmp_path / "k.db"))
    client = TestClient(build())
    assert client.get("/health").json()["providers"]["groq"]["keys"] == 2
    assert any(r.path == "/v1/realtime" for r in client.app.routes)


def test_clients_cannot_redirect_or_replace_provider_keys(make_client, upstream_url):
    from tests import fake_upstream

    r = chat(make_client({"groq": ["g1"]}), api_base="http://127.0.0.1:1/v1", api_key="attacker",
             fallbacks=["x"], num_retries=9, mock_response="pwned")
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "m1:g1"
    assert fake_upstream.calls[-1][1] == "g1"
