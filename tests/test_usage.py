from datetime import datetime, timezone

import pytest

from gateway.pricing import Counts, Pricer
from gateway.usage import UsageStore

AT = datetime(2026, 9, 27, 12, tzinfo=timezone.utc)


def prices(input_price=1e-6):
    return Pricer({"paid": {"openai/m": [{"effective_from": "2026-01-01", "input": input_price, "output": 2e-6}],
                            "deepgram/nova-3": [{"effective_from": "2026-01-01", "per_second": 0.001}]}},
                  bundled={}, fetch=None)


@pytest.fixture
def store(tmp_path):
    return UsageStore(str(tmp_path / "state.db"), prices(), clock=lambda: AT.timestamp())


def test_a_turn_sums_its_priced_components(store):
    t = store.open_turn("paid", "realtime", "realtime", conversation_id="conv1", session_id="s1")
    store.add(t, "stt", "paid", "deepgram", "nova-3", Counts(audio_seconds=10))
    store.add(t, "llm", "paid", "openai", "m", Counts(input=1000, output=100))
    store.close_turn(t)
    turn = store.turn(t)
    assert [c["kind"] for c in turn["components"]] == ["stt", "llm"]
    assert turn["cost"] == pytest.approx(0.01 + 0.001 + 0.0002)
    assert turn["cost_complete"] is True
    assert turn["components"][1]["price_source"] == "override"


def test_unpriced_component_makes_the_turn_incomplete(store):
    t = store.open_turn("paid", "chat", "chat")
    store.add(t, "chat", "paid", "openai", "unknown-model", Counts(input=5))
    store.close_turn(t)
    turn = store.turn(t)
    assert turn["components"][0]["cost"] is None and turn["cost_complete"] is False


def test_provider_reported_cost_wins(store):
    t = store.open_turn("free", "chat", "chat")
    store.add(t, "chat", "free", "openrouter", "x", Counts(input=5), provider_cost=0.0042)
    store.close_turn(t)
    c = store.turn(t)["components"][0]
    assert (c["cost"], c["price_source"]) == (0.0042, "provider")


def test_nothing_measured_is_unknown_not_free(store):
    t = store.open_turn("paid", "stt", "stt")
    store.add(t, "stt", "paid", "openai", "m", Counts())
    assert store.turn(t)["components"][0]["cost"] is None


def test_totals_group_and_turns_by_conversation(store):
    for conv in ("a", "a", "b"):
        t = store.open_turn("paid", "chat", "chat", conversation_id=conv)
        store.add(t, "chat", "paid", "openai", "m", Counts(input=100, output=10, cache_read=0))
        store.close_turn(t)
    rows = store.totals(None, None, "model")
    assert len(rows) == 1
    row = rows[0]
    assert {k: row[k] for k in ("provider", "model", "turns", "calls", "input", "output", "cache_read", "unpriced_calls")} == {
        "provider": "openai", "model": "m", "turns": 3, "calls": 3, "input": 300, "output": 30, "cache_read": 0,
        "unpriced_calls": 0}
    assert row["cost"] == pytest.approx(3 * (100e-6 + 20e-6)) == row["billed_cost"]
    assert row["list_price_estimate"] is None
    assert len(store.turns("a")) == 2
    with pytest.raises(ValueError):
        store.totals(None, None, "nonsense")


def test_reprice_recomputes_from_raw_counts(tmp_path):
    path = str(tmp_path / "state.db")
    old = UsageStore(path, prices(1e-6), clock=lambda: AT.timestamp())
    t = old.open_turn("paid", "chat", "chat")
    old.add(t, "chat", "paid", "openai", "m", Counts(input=1000))
    old.close_turn(t)
    new = UsageStore(path, prices(3e-6), clock=lambda: AT.timestamp())
    assert new.reprice(None, None) == 1
    assert new.turn(t)["cost"] == pytest.approx(0.003)


def test_session_current_turn(store):
    t = store.open_turn("paid", "realtime", "realtime", session_id="s1")
    store.set_session_turn("s1", t)
    assert store.session_turn("s1") == t
    store.set_session_turn("s1", None)
    assert store.session_turn("s1") is None
