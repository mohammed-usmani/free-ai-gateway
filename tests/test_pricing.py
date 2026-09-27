from datetime import datetime, timezone

import pytest
from litellm.types.utils import Usage
from pipecat.metrics.metrics import LLMTokenUsage

from gateway.pricing import Counts, Pricer, cost, counts_from_litellm, counts_from_pipecat

AT = datetime(2026, 9, 27, tzinfo=timezone.utc)
BUNDLED = {"gpt-x": {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6, "litellm_provider": "openai"},
           "gemini/g": {"input_cost_per_token": 3e-6, "output_cost_per_token": 4e-6}}
LIVE = {"gpt-x": {"input_cost_per_token": 5e-6, "output_cost_per_token": 6e-6, "cache_read_input_token_cost": 5e-7}}
OVERRIDES = {"paid": {"openai/gpt-x": [
    {"effective_from": "2026-01-01", "input": 7e-6, "output": 8e-6},
    {"effective_from": "2026-10-01", "input": 9e-6, "output": 9e-6},
]}}


def pricer(fetch=lambda: LIVE):
    return Pricer(OVERRIDES, bundled=BUNDLED, fetch=fetch)


def test_override_wins_and_respects_effective_dates():
    p = pricer().price("openai", "gpt-x", "paid", AT)
    assert (p.units["input"], p.source) == (7e-6, "override")
    later = pricer().price("openai", "gpt-x", "paid", datetime(2026, 10, 2, tzinfo=timezone.utc))
    assert later.units["input"] == 9e-6


async def test_live_map_beats_bundled_after_refresh():
    pr = pricer()
    assert pr.price("openai", "gpt-x", "free", AT).source == "litellm-bundled"
    await pr.refresh()
    p = pr.price("openai", "gpt-x", "free", AT)
    assert p.units["input"] == 5e-6 and p.source.startswith("litellm-live@")


async def test_failed_refresh_keeps_the_last_good_map():
    calls = iter([LIVE, RuntimeError("github down")])

    def fetch():
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    pr = pricer(fetch)
    await pr.refresh()
    await pr.refresh()
    assert pr.price("openai", "gpt-x", "free", AT).units["input"] == 5e-6
    assert "github down" in pr.status()["last_error"]


def test_provider_prefixed_keys_are_found():
    assert pricer().price("gemini", "g", "free", AT).units["output"] == 4e-6


def test_unknown_model_has_no_price_and_is_listed():
    pr = pricer()
    assert pr.price("openai", "nope", "free", AT) is None
    assert "openai/nope" in pr.status()["unpriced"]


def test_litellm_gross_usage_becomes_disjoint_buckets():
    usage = Usage(prompt_tokens=100, completion_tokens=40,
                  prompt_tokens_details={"cached_tokens": 30},
                  completion_tokens_details={"reasoning_tokens": 10},
                  cache_creation_input_tokens=20)
    c = counts_from_litellm(usage)
    assert (c.input, c.cache_read, c.cache_write, c.output, c.reasoning) == (50, 30, 20, 40, 10)


def test_pipecat_net_and_gross_usage_agree():
    net = LLMTokenUsage(prompt_tokens=50, completion_tokens=5, total_tokens=105,
                        cache_read_input_tokens=30, cache_creation_input_tokens=20)
    gross = LLMTokenUsage(prompt_tokens=100, completion_tokens=5, total_tokens=105,
                          cache_read_input_tokens=30, cache_creation_input_tokens=20)
    assert counts_from_pipecat(net, net=True) == counts_from_pipecat(gross, net=False)
    assert counts_from_pipecat(gross, net=False).input == 50


def test_pipecat_audio_tokens_are_split_out():
    u = LLMTokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cache_read_input_tokens=20,
                      input_audio_tokens=60, output_audio_tokens=40, cache_read_input_audio_tokens=10)
    c = counts_from_pipecat(u, net=False)
    assert (c.input, c.cache_read, c.audio_input, c.audio_cache_read, c.output, c.audio_output) == (30, 10, 50, 10, 10, 40)


def test_cost_sums_buckets_and_refuses_to_guess_missing_units():
    p = pricer().price("openai", "gpt-x", "paid", AT)
    assert cost(Counts(input=1000, output=500), p) == pytest.approx(1000 * 7e-6 + 500 * 8e-6)
    assert cost(Counts(input=10, cache_read=5), p) is None  # no cache_read unit in that override
    assert cost(Counts(input=10), None) is None


def test_voice_units():
    pr = Pricer({"paid": {"deepgram/nova-3": [{"effective_from": "2026-01-01", "per_second": 0.0001}],
                          "cartesia/sonic-3": [{"effective_from": "2026-01-01", "per_character": 0.00003}]}},
                bundled={}, fetch=None)
    assert cost(Counts(audio_seconds=60), pr.price("deepgram", "nova-3", "paid", AT)) == pytest.approx(0.006)
    assert cost(Counts(characters=1000), pr.price("cartesia", "sonic-3", "paid", AT)) == pytest.approx(0.03)


def test_gemini_live_reasoning_is_billed_as_output():
    u = LLMTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=35, reasoning_tokens=20)
    assert counts_from_pipecat(u, net=False, reasoning_in_output=False).output == 25
    assert counts_from_pipecat(u, net=False).output == 5
