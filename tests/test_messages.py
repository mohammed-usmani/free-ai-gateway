import json

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import Catalog, RealtimeChain, Step, load_catalog
from gateway.keypool import KeyPool
from gateway.modes import Modes
from gateway.pricing import Pricer
from gateway.router import Gateway
from gateway.usage import UsageStore

KEY = {"x-api-key": "s", "anthropic-version": "2023-06-01"}


@pytest.fixture
def setup(upstream_url, tmp_path):
    cat = Catalog({"smart": [Step("groq", "m", {"api_base": upstream_url})]}, RealtimeChain([], [], "smart", []),
                  model_map={"claude-*": "smart"})
    usage = UsageStore(str(tmp_path / "s.db"), Pricer({}, bundled={}, fetch=None))
    return TestClient(create_app(Modes({"free": Gateway(cat, {"groq": ["k"]})}), KeyPool(str(tmp_path / "k.db")), "s", usage=usage)), usage


def test_anthropic_shape_with_x_api_key_and_claude_model_names(setup):
    client, usage = setup
    r = client.post("/v1/messages", headers=KEY, json={"model": "claude-sonnet-4-5", "max_tokens": 50,
                                                       "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message" and body["content"][0]["text"] == "m:k" and "_hidden_params" not in body
    comp = usage.turn(r.headers["x-gateway-turn-id"])["components"][0]
    assert (comp["kind"], comp["input"], comp["output"]) == ("messages", 1, 1)


def test_streaming_messages_are_anthropic_sse(setup):
    client, _ = setup
    r = client.post("/v1/messages", headers=KEY, json={"model": "smart", "max_tokens": 50, "stream": True,
                                                       "messages": [{"role": "user", "content": "hi"}]})
    events = [l.removeprefix("event: ") for l in r.text.splitlines() if l.startswith("event: ")]
    assert events[0] == "message_start" and events[-1] == "message_stop"


def test_errors_use_the_anthropic_shape(setup):
    client, _ = setup
    r = client.post("/v1/messages", headers=KEY, json={"model": "unknown", "max_tokens": 5, "messages": []})
    assert r.status_code == 404 and r.json()["type"] == "error" and r.json()["error"]["type"] == "not_found_error"
    assert client.post("/v1/messages", headers={"x-api-key": "wrong"}, json={}).json()["type"] == "error"


def test_shipped_model_map_sends_claude_names_somewhere():
    assert load_catalog("providers.paid.yaml").model_map.get("claude-*")
