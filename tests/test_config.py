from pathlib import Path

from gateway.config import Step, available, load_catalog, load_keys

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
    catalog = load_catalog(ROOT / "providers.yaml")
    assert {"chat-smart", "chat-fast", "stt", "tts"} <= set(catalog.aliases)
    embeds = [a for a in catalog.aliases if a.startswith("embed-")]
    assert embeds and all(len(catalog.aliases[a]) == 1 for a in embeds)
    assert catalog.realtime.s2s[0].provider == "gemini"
    assert catalog.realtime.llm_alias in catalog.aliases
    assert catalog.realtime.stt and catalog.realtime.tts


def test_step_options_are_kept():
    catalog = load_catalog(ROOT / "providers.yaml")
    assert "voice" in catalog.realtime.s2s[0].options


def test_every_shipped_tts_step_names_a_voice():
    catalog = load_catalog(ROOT / "providers.yaml")
    assert all("voice" in s.options for s in catalog.aliases["tts"] + catalog.realtime.tts[1:])
