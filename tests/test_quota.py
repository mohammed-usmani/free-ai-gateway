import asyncio
from datetime import datetime, timezone

import httpx
import litellm
import pytest

from gateway.config import Catalog, RealtimeChain, Step
from gateway.keypool import KeyPool
from gateway.observe import Observer
from gateway.quota import daily_reset_seconds
from gateway.router import Gateway
from tests import fake_upstream

NOW = datetime(2026, 9, 27, 20, 0, tzinfo=timezone.utc)  # 13:00 in Los Angeles


def rate_limit(message, headers=None, provider="gemini"):
    resp = httpx.Response(429, headers=headers or {}, request=httpx.Request("POST", "http://x"))
    return litellm.RateLimitError(message=message, llm_provider=provider, model="m", response=resp)


def test_gemini_per_day_quota_resets_at_pacific_midnight():
    exc = rate_limit('{"error": {"status": "RESOURCE_EXHAUSTED", "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}')
    assert daily_reset_seconds(exc, "gemini", NOW) == pytest.approx(11 * 3600)  # 13:00 → midnight PDT


def test_per_minute_limits_are_not_daily():
    assert daily_reset_seconds(rate_limit("quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"), "gemini", NOW) is None
    assert daily_reset_seconds(rate_limit("Rate limit reached ... on tokens per minute (TPM)", {"retry-after": "7"}, "groq"), "groq", NOW) is None
    assert daily_reset_seconds(litellm.BadRequestError("per day", "groq", "m"), "groq", NOW) is None


def test_groq_daily_uses_its_reset_header():
    exc = rate_limit("Rate limit reached for model x on requests per day (RPD): Limit 1000", {"x-ratelimit-reset-requests": "2m59.5s"}, "groq")
    assert daily_reset_seconds(exc, "groq", NOW) == pytest.approx(179.5)


def test_openrouter_free_daily_uses_reset_timestamp():
    reset_ms = int((NOW.timestamp() + 3600) * 1000)
    exc = rate_limit("Rate limit exceeded: free-models-per-day", {"X-RateLimit-Reset": str(reset_ms)}, "openrouter")
    assert daily_reset_seconds(exc, "openrouter", NOW) == pytest.approx(3600)


def test_cloudflare_daily_neurons_reset_at_utc_midnight():
    exc = rate_limit('{"errors":[{"code":4006,"message":"you have used up your daily free allocation of 10,000 neurons"}]}', provider="cloudflare")
    assert daily_reset_seconds(exc, "cloudflare", NOW) == pytest.approx(4 * 3600)


def test_sambanova_day_headers():
    exc = rate_limit("Rate limit exceeded", {"x-ratelimit-remaining-requests-day": "0", "x-ratelimit-reset-requests-day": "5000"}, "sambanova")
    assert daily_reset_seconds(exc, "sambanova", NOW) == pytest.approx(5000)


async def test_daily_limited_key_is_benched_and_stays_benched_after_restart(upstream_url, tmp_path):
    pool = KeyPool(str(tmp_path / "k.db"))
    obs = Observer(pool)
    cat = Catalog({"chat": [Step("gemini", "g", {"api_base": upstream_url, "custom_llm_provider": "openai"}),
                            Step("openrouter", "backup", {"api_base": upstream_url})]},
                  RealtimeChain([], [], "chat", []))

    def make():
        return Gateway(cat, {"gemini": ["day"], "openrouter": ["k2"]}, mode="free", observer=obs)

    g = make()
    for _ in range(3):
        await g.call("acompletion", "chat", messages=[{"role": "user", "content": "hi"}])
    for _ in range(100):
        if pool.pick("quota:free:gemini/g", ["day"]) is None:
            break
        await asyncio.sleep(0.05)
    assert pool.pick("quota:free:gemini/g", ["day"]) is None  # persisted
    fake_upstream.calls.clear()
    g2 = make()  # a restart
    for _ in range(4):
        await g2.call("acompletion", "chat", messages=[{"role": "user", "content": "hi"}])
    assert {k for _, k in fake_upstream.calls} == {"k2"}


async def test_attempts_are_logged(upstream_url, tmp_path):
    pool = KeyPool(str(tmp_path / "k.db"))
    rows = []
    obs = Observer(pool, sink=rows.append)
    cat = Catalog({"chat": [Step("groq", "a", {"api_base": upstream_url}), Step("openrouter", "b", {"api_base": upstream_url})]},
                  RealtimeChain([], [], "chat", []))
    g = Gateway(cat, {"groq": ["rl"], "openrouter": ["o"]}, mode="paid", observer=obs)
    await g.call("acompletion", "chat", messages=[{"role": "user", "content": "hi"}])
    for _ in range(100):
        if {r["provider"] for r in rows} >= {"groq", "openrouter"}:
            break
        await asyncio.sleep(0.05)
    got = {(r["provider"], r["ok"]) for r in rows}
    assert ("groq", False) in got and ("openrouter", True) in got
    assert all(r["mode"] == "paid" and r["latency_ms"] >= 0 for r in rows)
    assert "rl" not in str(rows)


async def test_daily_bench_outlasts_the_normal_cooldown(upstream_url, tmp_path):
    from gateway.config import ModeSettings

    pool = KeyPool(str(tmp_path / "k.db"))
    obs = Observer(pool)
    cat = Catalog({"chat": [Step("gemini", "g", {"api_base": upstream_url, "custom_llm_provider": "openai"})]},
                  RealtimeChain([], [], "chat", []), settings=ModeSettings(cooldown_seconds=1, num_retries=1))
    g = Gateway(cat, {"gemini": ["day", "good"]}, mode="free", observer=obs)
    for _ in range(20):
        await g.call("acompletion", "chat", messages=[{"role": "user", "content": "hi"}])
        if pool.pick("quota:free:gemini/g", ["day"]) is None:
            break
    for _ in range(100):
        if pool.pick("quota:free:gemini/g", ["day"]) is None:
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(1.6)  # past LiteLLM's own 1 s cooldown
    fake_upstream.calls.clear()
    for _ in range(6):
        await g.call("acompletion", "chat", messages=[{"role": "user", "content": "hi"}])
    assert {k for _, k in fake_upstream.calls} == {"good"}
