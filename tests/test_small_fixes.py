"""Tests for the small issues deferred by the two reviews."""

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from litellm.types.utils import Usage

from gateway.app import _stt_counts, create_app
from gateway.config import Catalog, RealtimeChain, Step, known_providers
from gateway.keypool import KeyPool, mask
from gateway.modes import Modes
from gateway.pricing import Counts, Price, Pricer, cost, counts_from_litellm
from gateway.router import Gateway
from gateway.usage import UsageStore

AUTH = {"Authorization": "Bearer gw-secret"}
AT = datetime(2026, 9, 27, tzinfo=timezone.utc)


@pytest.fixture
def setup(upstream_url, tmp_path):
    o = {"api_base": upstream_url}
    catalog = Catalog({"chat": [Step("openai", "m", o)], "embed": [Step("openai", "e", o)]},
                      RealtimeChain([], [], "chat", []))
    usage = UsageStore(str(tmp_path / "s.db"), Pricer({}, bundled={}, fetch=None))
    modes = Modes({"free": Gateway(catalog, {"openai": ["k"]})})
    app = create_app(modes, KeyPool(str(tmp_path / "k.db")), "gw-secret", usage=usage)
    return TestClient(app, raise_server_exceptions=False), usage, modes


def chat(client, headers=None, model="chat"):
    return client.post("/v1/chat/completions", headers={**AUTH, **(headers or {})},
                       json={"model": model, "messages": [{"role": "user", "content": "hi"}]})


# --- realtime session tagging ---

def test_session_tag_is_a_private_token_not_the_session_id(setup):
    client, usage, _ = setup
    usage.register_session("secret-token", "sess_public", "free")
    t = usage.open_turn("free", "realtime", "realtime", session_id="sess_public")
    usage.set_session_turn("secret-token", t)
    assert chat(client, {"x-gateway-session": "sess_public"}).headers["x-gateway-turn-id"] != t
    assert chat(client, {"x-gateway-session": "secret-token"}).headers["x-gateway-turn-id"] == t


def test_session_tag_only_applies_to_chat(setup):
    client, usage, _ = setup
    usage.register_session("tok", "sess", "free")
    t = usage.open_turn("free", "realtime", "realtime", session_id="sess")
    usage.set_session_turn("tok", t)
    r = client.post("/v1/embeddings", headers={**AUTH, "x-gateway-session": "tok"}, json={"model": "embed", "input": "a"})
    assert r.headers["x-gateway-turn-id"] != t
    assert usage.turn(r.headers["x-gateway-turn-id"])["components"][0]["kind"] == "embed"


def test_cascade_call_outside_a_turn_still_belongs_to_the_session(setup):
    client, usage, _ = setup
    usage.register_session("tok", "sess_x", "free")
    turn = usage.turn(chat(client, {"x-gateway-session": "tok"}).headers["x-gateway-turn-id"])
    assert (turn["session_id"], turn["endpoint"], turn["components"][0]["kind"]) == ("sess_x", "realtime", "llm")


# --- API edge cases ---

def test_non_object_json_body_is_400(setup):
    client, _, _ = setup
    r = client.post("/v1/chat/completions", headers=AUTH, content=json.dumps([1, 2]))
    assert r.status_code == 400


def test_internal_key_errors_are_not_reported_as_unknown_model(setup, monkeypatch):
    client, _, modes = setup

    async def broken(*a, **k):
        raise KeyError("some_internal_field")

    monkeypatch.setattr(modes, "call", broken)
    assert chat(client).status_code == 500
    monkeypatch.undo()
    assert chat(client, model="nope").status_code == 404


def test_docs_are_not_public(setup):
    client, _, _ = setup
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_usage_endpoints_validate_input(setup):
    client, _, _ = setup
    assert client.post("/v1/usage/reprice?from=not-a-date", headers=AUTH).status_code == 400
    assert client.get("/v1/usage?from=not-a-date", headers=AUTH).status_code == 400
    assert client.get("/v1/usage/turns?limit=-5", headers=AUTH).status_code == 200


# --- usage accounting ---

def test_turn_without_components_is_unknown_not_free(tmp_path):
    usage = UsageStore(str(tmp_path / "s.db"), Pricer({}, bundled={}, fetch=None))
    t = usage.open_turn("free", "chat", "chat")
    usage.close_turn(t)
    turn = usage.turn(t)
    assert (turn["cost"], turn["cost_complete"]) == (None, False)


def test_transcription_seconds_come_from_usage_when_reported():
    class Resp:
        usage = type("U", (), {"type": "duration", "seconds": 12.5})()

    assert _stt_counts(Resp(), b"not-a-wav").audio_seconds == 12.5


# --- pricing ---

def test_long_prompts_use_tiered_prices():
    p = Price({"input": 1e-6, "output": 2e-6, "input@200000": 3e-6, "output@200000": 4e-6}, "x")
    assert cost(Counts(input=100_000, output=10), p) == pytest.approx(100_000e-6 + 20e-6)
    assert cost(Counts(input=250_000, output=10), p) == pytest.approx(250_000 * 3e-6 + 10 * 4e-6)


def test_litellm_tier_and_one_hour_cache_keys_are_read():
    pr = Pricer({}, bundled={"claude-x": {"litellm_provider": "anthropic", "input_cost_per_token": 1e-6,
                                          "input_cost_per_token_above_200k_tokens": 2e-6,
                                          "cache_creation_input_token_cost": 1.25e-6,
                                          "cache_creation_input_token_cost_above_1hr": 2e-6}}, fetch=None)
    units = pr.price("anthropic", "claude-x", "paid", AT).units
    assert units["input@200000"] == 2e-6 and units["cache_write_1h"] == 2e-6


def test_one_hour_cache_writes_are_their_own_bucket():
    u = Usage(prompt_tokens=10, completion_tokens=1, cache_creation_input_tokens=5,
              cache_creation_token_details={"ephemeral_1h_input_tokens": 3, "ephemeral_5m_input_tokens": 2})
    c = counts_from_litellm(u)
    assert (c.input, c.cache_write, c.cache_write_1h) == (5, 2, 3)


def test_bare_model_names_do_not_cross_providers():
    pr = Pricer({}, bundled={"shared-name": {"litellm_provider": "openai", "input_cost_per_token": 1e-6}}, fetch=None)
    assert pr.price("openai", "shared-name", "free", AT) is not None
    assert pr.price("groq", "shared-name", "free", AT) is None


def test_new_bucket_columns_are_added_to_an_existing_database(tmp_path):
    import sqlite3
    path = str(tmp_path / "old.db")
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE components (id INTEGER PRIMARY KEY, turn_id TEXT, input REAL)")
    db.commit()
    db.close()
    UsageStore(path, Pricer({}, bundled={}, fetch=None))
    cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(components)")}
    assert "cache_write_1h" in cols


# --- keys ---

def test_unrelated_api_key_variables_are_not_providers():
    catalog = Catalog({"chat": [Step("groq", "m")]}, RealtimeChain([], [Step("deepgram", "n")], "chat", []))
    keys = {"groq": ["g"], "langsmith": ["l"], "together_ai": ["t"], "deepgram": ["d"]}
    assert known_providers(keys, [catalog]) == {"groq": ["g"], "together_ai": ["t"], "deepgram": ["d"]}


def test_masks_show_only_the_last_four_characters():
    assert mask("gsk_verysecret_1234") == "…1234"
