import base64

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import Catalog, RealtimeChain, Step, load_catalog
from gateway.keypool import KeyPool
from gateway.modes import Modes
from gateway.pricing import Counts, Pricer
from gateway.router import Gateway
from gateway.usage import UsageStore

AUTH = {"Authorization": "Bearer s"}


@pytest.fixture
def setup(upstream_url, tmp_path):
    root = upstream_url.removesuffix("/v1")
    o = {"api_base": upstream_url}
    cat = Catalog({
        "rerank": [Step("jina_ai", "jina-reranker-v3", o)],
        "image": [Step("cloudflare", "@cf/black-forest-labs/flux-1-schnell",
                       {"custom_llm_provider": "cloudflare_images", "api_base": root, "account_id": "acct"}),
                  Step("pollinations", "flux", {"custom_llm_provider": "openai", "api_base": upstream_url})],
        "moderation": [Step("openai", "omni-moderation-latest", o)],
    }, RealtimeChain([], [], "x", []))
    usage = UsageStore(str(tmp_path / "s.db"), Pricer({"free": {"cloudflare/@cf/black-forest-labs/flux-1-schnell": [
        {"effective_from": "2026-01-01", "per_image": 0.0005}]}}, bundled={}, fetch=None))
    keys = {"jina_ai": ["j"], "cloudflare": ["cf"], "pollinations": ["p"], "openai": ["o"]}
    return TestClient(create_app(Modes({"free": Gateway(cat, keys)}), KeyPool(str(tmp_path / "k.db")), "s", usage=usage)), usage, keys


def test_rerank(setup):
    client, usage, _ = setup
    r = client.post("/v1/rerank", headers=AUTH, json={"model": "rerank", "query": "q", "documents": ["a", "bbb", "cc"], "top_n": 2})
    assert r.status_code == 200
    assert [x["index"] for x in r.json()["results"]] == [1, 2]
    comp = usage.turn(r.headers["x-gateway-turn-id"])["components"][0]
    assert comp["kind"] == "rerank" and (comp["searches"] == 1 or comp["input"] > 0)


def test_cloudflare_images_through_the_custom_provider(setup):
    client, usage, _ = setup
    r = client.post("/v1/images/generations", headers=AUTH, json={"model": "image", "prompt": "a cat"})
    assert r.status_code == 200
    img = base64.b64decode(r.json()["data"][0]["b64_json"]).decode()
    assert img == "acct:@cf/black-forest-labs/flux-1-schnell:a cat"
    comp = usage.turn(r.headers["x-gateway-turn-id"])["components"][0]
    assert (comp["images"], comp["cost"]) == (1, pytest.approx(0.0005))


def test_images_fall_back_to_pollinations(upstream_url, tmp_path):
    root = upstream_url.removesuffix("/v1")
    cat = Catalog({"image": [Step("cloudflare", "@cf/x", {"custom_llm_provider": "cloudflare_images", "api_base": root, "account_id": "a"}),
                             Step("pollinations", "flux", {"custom_llm_provider": "openai", "api_base": upstream_url})]},
                  RealtimeChain([], [], "x", []))
    c = TestClient(create_app(Modes({"free": Gateway(cat, {"cloudflare": ["rl"], "pollinations": ["p"]})}), KeyPool(str(tmp_path / "k2.db")), "s"))
    r = c.post("/v1/images/generations", headers=AUTH, json={"model": "image", "prompt": "a dog"})
    assert r.status_code == 200 and r.headers["x-gateway-model"] == "pollinations/flux"


def test_moderation(setup):
    client, usage, _ = setup
    r = client.post("/v1/moderations", headers=AUTH, json={"model": "moderation", "input": "I will kill"})
    assert r.status_code == 200 and r.json()["results"][0]["flagged"] is True
    comp = usage.turn(r.headers["x-gateway-turn-id"])["components"][0]
    assert (comp["kind"], comp["cost"]) == ("moderation", 0.0)


def test_shipped_catalogs_have_the_new_aliases():
    free, paid = load_catalog("providers.free.yaml"), load_catalog("providers.paid.yaml")
    for c in (free, paid):
        assert {"rerank", "image", "moderation"} <= set(c.aliases)
    assert free.aliases["image"][0].options["custom_llm_provider"] == "cloudflare_images"
