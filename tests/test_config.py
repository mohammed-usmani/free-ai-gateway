from pathlib import Path

import pytest

from gateway.config import Step, available, check_fallbacks, load_catalog, load_keys

ROOT = Path(__file__).resolve().parent.parent


def test_csv_keys_are_trimmed_deduplicated_and_ordered():
    assert load_keys({"GROQ_API_KEYS": " a, b ,,a"}) == {"groq": ["a", "b"]}


def test_single_key_variable_is_a_one_key_list():
    assert load_keys({"MISTRAL_API_KEY": "k"}) == {"mistral": ["k"]}


def test_csv_wins_over_single_key_and_other_vars_are_ignored():
    env = {"GROQ_API_KEYS": "a,b", "GROQ_API_KEY": "c", "PATH": "/bin", "GATEWAY_API_KEY": "g"}
    assert load_keys(env) == {"groq": ["a", "b"]}


def test_multi_word_provider_names():
    assert load_keys({"NVIDIA_NIM_API_KEYS": "n"}) == {"nvidia_nim": ["n"]}


def test_steps_without_keys_are_skipped():
    steps = [Step("groq", "m1"), Step("mistral", "m2")]
    assert available(steps, {"mistral": ["k"]}) == [Step("mistral", "m2")]


def test_shipped_catalog_loads_and_embed_aliases_hold_one_model():
    catalog = load_catalog(ROOT / "providers.free.yaml")
    assert {"chat-smart", "chat-fast", "stt", "tts"} <= set(catalog.aliases)
    embeds = [a for a in catalog.aliases if a.startswith("embed-")]
    assert embeds and all(len(catalog.aliases[a]) == 1 for a in embeds)
    assert catalog.realtime.s2s[0].provider == "gemini"
    assert catalog.realtime.llm_alias in catalog.aliases
    assert catalog.realtime.stt and catalog.realtime.tts


def test_step_options_are_kept():
    catalog = load_catalog(ROOT / "providers.free.yaml")
    assert "voice" in catalog.realtime.s2s[0].options


def test_every_shipped_tts_step_names_a_voice():
    catalog = load_catalog(ROOT / "providers.free.yaml")
    assert all("voice" in s.options for s in catalog.aliases["tts"] + catalog.realtime.tts[1:])


def test_paid_keys_are_separate_from_free_keys():
    env = {"GROQ_API_KEYS": "a", "PAID_GROQ_API_KEYS": "b", "PAID_OPENAI_API_KEY": "o"}
    assert load_keys(env) == {"groq": ["a"]}
    assert load_keys(env, prefix="PAID_") == {"groq": ["b"], "openai": ["o"]}


def write(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return p


CATALOG = """
settings: {cooldown_seconds: 5, num_retries: 2, first_byte_timeout: 8}
aliases:
  chat:
    steps: [{provider: openai, model: gpt-x}]
    fallback_to: free/chat-smart
  embed: [{provider: openai, model: e}]
prices:
  openai/gpt-x:
    - {effective_from: 2026-01-01, input: 0.000001, output: 0.000002}
realtime:
  s2s: []
  stt: []
  llm: chat
  tts: []
  fallback_to: free/realtime
"""


def test_mode_catalog_reads_settings_fallbacks_and_prices(tmp_path):
    c = load_catalog(write(tmp_path, CATALOG))
    assert (c.settings.cooldown_seconds, c.settings.num_retries, c.settings.first_byte_timeout) == (5, 2, 8)
    assert c.aliases["chat"] == [Step("openai", "gpt-x")]
    assert c.fallback_to == {"chat": "free/chat-smart"}
    assert c.prices["openai/gpt-x"][0]["input"] == 0.000001
    assert c.realtime.fallback_to == "free/realtime"


def test_settings_default_when_missing():
    c = load_catalog(ROOT / "providers.free.yaml")
    assert c.settings.cooldown_seconds == 60 and c.settings.num_retries is None


def test_fallback_cycles_and_unknown_targets_are_rejected(tmp_path):
    paid = load_catalog(write(tmp_path, CATALOG))
    free = load_catalog(ROOT / "providers.free.yaml")
    check_fallbacks({"free": free, "paid": paid})
    free.fallback_to["chat-smart"] = "paid/chat"
    paid.fallback_to["chat"] = "free/chat-smart"
    with pytest.raises(ValueError, match="cycle"):
        check_fallbacks({"free": free, "paid": paid})
    paid.fallback_to["chat"] = "free/nope"
    with pytest.raises(ValueError, match="unknown"):
        check_fallbacks({"free": free, "paid": paid})


def test_shipped_paid_catalog_loads_and_fallbacks_are_sound():
    free = load_catalog(ROOT / "providers.free.yaml")
    paid = load_catalog(ROOT / "providers.paid.yaml")
    check_fallbacks({"free": free, "paid": paid})
    assert paid.settings.cooldown_seconds < free.settings.cooldown_seconds
    assert {"chat-best", "chat-smart", "chat-fast", "stt", "tts"} <= set(paid.aliases)
    assert paid.fallback_to["chat-fast"] == "free/chat-fast"
    assert [s.provider for s in paid.realtime.s2s] == ["openai", "gemini"]
    assert paid.realtime.fallback_to == "free/realtime"
    embeds = [a for a in paid.aliases if a.startswith("embed-")]
    assert embeds and all(len(paid.aliases[a]) == 1 and a not in paid.fallback_to for a in embeds)


def test_bad_price_entries_are_rejected_at_load(tmp_path):
    bad = CATALOG.replace("- {effective_from: 2026-01-01, input: 0.000001, output: 0.000002}",
                          "- {input: 0.000001}")
    with pytest.raises(ValueError, match="effective_from"):
        load_catalog(write(tmp_path, bad))
    bad = CATALOG.replace("output: 0.000002}", "output: cheap}")
    with pytest.raises(ValueError, match="output"):
        load_catalog(write(tmp_path, bad))
