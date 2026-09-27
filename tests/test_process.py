import pytest

from gateway.config import load_catalog
from gateway.lock import SingleProcessLock
from gateway.pricing import Counts, Pricer
from gateway.usage import UsageStore


def test_second_process_on_the_same_state_is_refused(tmp_path):
    first = SingleProcessLock(str(tmp_path / "state.db"))
    with pytest.raises(RuntimeError, match="one gateway process"):
        SingleProcessLock(str(tmp_path / "state.db"))
    first.release()
    SingleProcessLock(str(tmp_path / "state.db")).release()


def test_cost_basis_labels_free_as_an_estimate(tmp_path):
    pr = Pricer({"free": {"g/m": [{"effective_from": "2026-01-01", "input": 1e-6}]},
                 "paid": {"o/m": [{"effective_from": "2026-01-01", "input": 2e-6}]}}, bundled={}, fetch=None)
    store = UsageStore(str(tmp_path / "s.db"), pr)
    for mode, prov in (("free", "g"), ("paid", "o")):
        t = store.open_turn(mode, "chat", "chat")
        store.add(t, "chat", mode, prov, "m", Counts(input=1000))
        store.close_turn(t)
    turns = {t["mode"]: t for t in store.turns()}
    assert turns["free"]["cost_basis"] == "list_price_estimate" and turns["paid"]["cost_basis"] == "billed"
    assert turns["free"]["components"][0]["cost_basis"] == "list_price_estimate"
    total = store.totals(None, None, "endpoint")[0]
    assert total["billed_cost"] == pytest.approx(0.002) and total["list_price_estimate"] == pytest.approx(0.001)


def test_paid_first_byte_allows_reasoning_models():
    paid = load_catalog("providers.paid.yaml")
    assert paid.settings.first_byte_timeout >= 30
    assert all(s.options.get("first_byte_timeout", 99) <= 15 for s in paid.aliases["chat-fast"])
