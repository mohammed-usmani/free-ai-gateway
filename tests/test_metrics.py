from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import Catalog, RealtimeChain, Step
from gateway.keypool import KeyPool
from gateway.modes import Modes
from gateway.pricing import Counts, Pricer
from gateway.router import Gateway
from gateway.usage import UsageStore

AUTH = {"Authorization": "Bearer s"}


def row(provider, ok, latency, error=None, at=1000.0, cache=False):
    return {"at": at, "mode": "free", "provider": provider, "model": "m", "key": "abc", "ok": ok, "status": None if ok else 429,
            "error": error, "latency_ms": latency, "cache_hit": cache}


def test_provider_metrics(tmp_path):
    store = UsageStore(str(tmp_path / "s.db"), Pricer({}, bundled={}, fetch=None), clock=lambda: 2000.0)
    for lat in (100, 200, 300, 400):
        store.log_attempt(row("groq", True, lat))
    store.log_attempt(row("groq", False, 50, "RateLimitError"))
    store.log_attempt(row("groq", False, 0, "skipped: no vision"))
    store.log_attempt(row("groq", True, 10, at=10.0))  # outside the window
    m = store.metrics(window_seconds=1500)["providers"][0]
    assert (m["provider"], m["requests"], m["skipped"]) == ("groq", 5, 1)
    assert m["success_rate"] == pytest.approx(0.8)
    assert (m["p50_ms"], m["p95_ms"]) == (200, 400)
    assert m["errors"] == {"RateLimitError": 1}


def test_alias_fallback_rate(tmp_path):
    store = UsageStore(str(tmp_path / "s.db"), Pricer({}, bundled={}, fetch=None))
    for fallbacks in (0, 0, 1, 2):
        t = store.open_turn("free", "chat", "chat-fast")
        store.add(t, "chat", "free", "groq", "m", Counts(input=1), fallbacks=fallbacks)
        store.close_turn(t)
    a = store.metrics(window_seconds=10**9)["aliases"][0]
    assert (a["alias"], a["turns"], a["fallback_rate"]) == ("chat-fast", 4, 0.5)


@pytest.fixture
def client(upstream_url, tmp_path):
    o = {"api_base": upstream_url}
    cat = Catalog({"chat": [Step("openai", "gpt-5.4"), Step("openai", "gpt-5.4-mini")], "x": [Step("openai", "nope-model")]},
                  RealtimeChain([], [], "chat", []))
    store = UsageStore(str(tmp_path / "s.db"), Pricer({}, fetch=None))
    store.log_attempt(row("openai", True, 120, at=__import__("time").time()))
    app = create_app(Modes({"free": Gateway(cat, {"openai": ["k"]})}), KeyPool(str(tmp_path / "k.db")), "s", usage=store)
    return TestClient(app)


def test_metrics_endpoints(client):
    body = client.get("/v1/metrics?window=100000000", headers=AUTH).json()
    assert body["providers"][0]["provider"] == "openai"
    text = client.get("/metrics", headers=AUTH).text
    assert 'gateway_attempts_total{mode="free",provider="openai",model="m",outcome="ok"} 1' in text
    assert client.get("/metrics").status_code == 401


def test_pricing_health_flags(client):
    health = client.get("/v1/pricing/health", headers=AUTH).json()["models"]
    by = {h["model"]: h for h in health}
    assert "unpriced" in by["nope-model"]["flags"]
    assert by["gpt-5.4"]["priced"] is True
    assert client.get("/health").json()["pricing"]["health"]["unpriced"] >= 1


def test_identical_prices_to_another_model_are_flagged():
    bundled = {"new-model": {"litellm_provider": "openai", "input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6},
               "old-model": {"litellm_provider": "openai", "input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6},
               "new-model-2026-01-01": {"litellm_provider": "openai", "input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6},
               "unique": {"litellm_provider": "openai", "input_cost_per_token": 9e-6, "output_cost_per_token": 9e-6}}
    pr = Pricer({"free": {"openai/overridden": [{"effective_from": "2025-01-01", "input": 1e-6}]}}, bundled=bundled, fetch=None)
    at = datetime(2026, 9, 27, tzinfo=timezone.utc)
    report = {r["model"]: r["flags"] for r in pr.health([("free", "openai", "new-model"), ("free", "openai", "unique"),
                                                         ("free", "openai", "overridden")], at)}
    assert report["new-model"] == ["same_price_as:old-model"]
    assert report["unique"] == []
    assert "override_older_than_90_days" in report["overridden"]
