import litellm
import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app, enable_cache
from gateway.config import Catalog, RealtimeChain, Step
from gateway.keypool import KeyPool
from gateway.modes import Modes
from gateway.pricing import Pricer
from gateway.router import Gateway
from gateway.usage import UsageStore
from tests import fake_upstream

AUTH = {"Authorization": "Bearer s"}
MSG = {"model": "chat", "messages": [{"role": "user", "content": "same prompt"}]}


@pytest.fixture
def setup(upstream_url, tmp_path):
    enable_cache(str(tmp_path / "cache"), ttl=60)
    o = {"api_base": upstream_url}
    cat = Catalog({"chat": [Step("openai", "m", o)], "cached-chat": [Step("openai", "m2", o)]},
                  RealtimeChain([], [], "chat", []), cache_aliases={"cached-chat"})
    usage = UsageStore(str(tmp_path / "s.db"), Pricer({}, bundled={}, fetch=None))
    yield TestClient(create_app(Modes({"free": Gateway(cat, {"openai": ["k"]})}), KeyPool(str(tmp_path / "k.db")), "s", usage=usage)), usage
    litellm.cache = None


def test_cache_is_off_unless_asked(setup):
    client, _ = setup
    client.post("/v1/chat/completions", headers=AUTH, json=MSG)
    client.post("/v1/chat/completions", headers=AUTH, json=MSG)
    assert len(fake_upstream.calls) == 2


def test_opted_in_repeat_is_served_from_cache_and_costs_nothing(setup):
    client, usage = setup
    h = {**AUTH, "x-gateway-cache": "true"}
    a = client.post("/v1/chat/completions", headers=h, json=MSG)
    b = client.post("/v1/chat/completions", headers=h, json=MSG)
    assert len(fake_upstream.calls) == 1
    assert a.json()["choices"] == b.json()["choices"] and b.headers["x-gateway-cache"] == "hit"
    comp = usage.turn(b.headers["x-gateway-turn-id"])["components"][0]
    assert (comp["attempt"], comp["cost"], comp["price_source"]) == ("cached", 0.0, "cache")


def test_alias_can_turn_caching_on(setup):
    client, _ = setup
    for _ in range(2):
        client.post("/v1/chat/completions", headers=AUTH, json={**MSG, "model": "cached-chat"})
    assert len(fake_upstream.calls) == 1


def test_cache_false_header_wins_over_alias(setup):
    client, _ = setup
    for _ in range(2):
        client.post("/v1/chat/completions", headers={**AUTH, "x-gateway-cache": "false"}, json={**MSG, "model": "cached-chat"})
    assert len(fake_upstream.calls) == 2
