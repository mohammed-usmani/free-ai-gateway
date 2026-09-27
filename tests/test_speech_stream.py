from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import Catalog, RealtimeChain, Step
from gateway.keypool import KeyPool
from gateway.modes import Modes
from gateway.router import Gateway
from tests import fake_upstream

AUTH = {"Authorization": "Bearer s"}


def client(url, steps, keys, tmp_path):
    cat = Catalog({"tts": steps}, RealtimeChain([], [], "chat", []))
    return TestClient(create_app(Modes({"free": Gateway(cat, keys)}), KeyPool(str(tmp_path / "k.db")), "s"))


def test_speech_streams_chunks_from_an_openai_compatible_provider(upstream_url, tmp_path):
    c = client(upstream_url, [Step("openai", "tts-1", {"api_base": upstream_url, "voice": "alloy"})], {"openai": ["k"]}, tmp_path)
    with c.stream("POST", "/v1/audio/speech", headers=AUTH, json={"model": "tts", "input": "hello", "response_format": "wav"}) as r:
        chunks = list(r.iter_bytes())
        assert r.headers["content-type"] == "audio/wav" and r.headers["x-gateway-model"] == "openai/tts-1"
    assert b"".join(chunks) == b"RIFFfake-audio:tts-1:k"


def test_speech_falls_back_before_the_first_chunk(upstream_url, tmp_path):
    c = client(upstream_url, [Step("openai", "tts-1", {"api_base": upstream_url, "voice": "alloy"}),
                              Step("groq", "orpheus", {"api_base": upstream_url, "voice": "autumn"})],
               {"openai": ["rl"], "groq": ["g"]}, tmp_path)
    r = c.post("/v1/audio/speech", headers=AUTH, json={"model": "tts", "input": "hi"})
    assert r.status_code == 200 and r.content.endswith(b"orpheus:g") and r.headers["x-gateway-fallbacks"] == "1"


def test_elevenlabs_uses_its_streaming_endpoint(upstream_url, tmp_path):
    root = upstream_url.removesuffix("/v1")
    c = client(upstream_url, [Step("elevenlabs", "eleven_flash_v2_5", {"api_base": root, "voice": "v123"})],
               {"elevenlabs": ["e"]}, tmp_path)
    r = c.post("/v1/audio/speech", headers=AUTH, json={"model": "tts", "input": "hi"})
    assert r.content == b"ID3eleven:v123:eleven_flash_v2_5"
    assert ("/v1/text-to-speech/v123/stream", "e") in fake_upstream.calls


def test_bad_speech_request_is_returned(upstream_url, tmp_path):
    c = client(upstream_url, [Step("openai", "tts-1", {"api_base": upstream_url, "voice": "a"}),
                              Step("groq", "o", {"api_base": upstream_url, "voice": "a"})], {"openai": ["bad"], "groq": ["g"]}, tmp_path)
    assert c.post("/v1/audio/speech", headers=AUTH, json={"model": "tts", "input": "hi"}).status_code == 400
