import json

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import Catalog, RealtimeChain, Step
from gateway.keypool import KeyPool
from gateway.modes import Modes
from gateway.pricing import Pricer
from gateway.responses import ResponseStore
from gateway.router import Gateway
from gateway.usage import UsageStore
from tests import fake_upstream

AUTH = {"Authorization": "Bearer s"}


@pytest.fixture
def setup(upstream_url, tmp_path):
    cat = Catalog({"agent": [Step("groq", "m", {"api_base": upstream_url})]}, RealtimeChain([], [], "agent", []))
    usage = UsageStore(str(tmp_path / "s.db"), Pricer({}, bundled={}, fetch=None))
    store = ResponseStore(str(tmp_path / "s.db"))
    app = create_app(Modes({"free": Gateway(cat, {"groq": ["k"]})}), KeyPool(str(tmp_path / "k.db")), "s",
                     usage=usage, responses=store)
    return TestClient(app), usage, store


def create(client, **body):
    return client.post("/v1/responses", headers=AUTH, json={"model": "agent", **body})


def test_response_is_returned_stored_and_retrievable(setup):
    client, usage, _ = setup
    r = create(client, input="hello")
    assert r.status_code == 200
    body = r.json()
    assert body["output"][0]["content"][0]["text"] == "m:k"
    assert client.get(f"/v1/responses/{body['id']}", headers=AUTH).json()["id"] == body["id"]
    assert fake_upstream.bodies[-1].get("store") in (None, False)  # providers never store
    comp = usage.turn(r.headers["x-gateway-turn-id"])["components"][0]
    assert (comp["kind"], comp["input"], comp["output"]) == ("responses", 1, 1)


def test_previous_response_id_rebuilds_the_conversation(setup):
    client, _, _ = setup
    first = create(client, input="my name is Mo").json()
    create(client, input="what is my name?", previous_response_id=first["id"])
    sent = fake_upstream.bodies[-1]["messages"]
    texts = json.dumps(sent)
    assert "my name is Mo" in texts and "m:k" in texts and "what is my name?" in texts


def test_store_false_keeps_nothing(setup):
    client, _, _ = setup
    body = create(client, input="secret", store=False).json()
    assert client.get(f"/v1/responses/{body['id']}", headers=AUTH).status_code == 404
    r = create(client, input="next", previous_response_id=body["id"])
    assert r.status_code == 404


def test_delete_removes_a_stored_response(setup):
    client, _, _ = setup
    body = create(client, input="x").json()
    assert client.delete(f"/v1/responses/{body['id']}", headers=AUTH).json()["deleted"] is True
    assert client.get(f"/v1/responses/{body['id']}", headers=AUTH).status_code == 404


def test_old_responses_are_purged(tmp_path):
    store = ResponseStore(str(tmp_path / "r.db"), clock=lambda: 1000.0)
    store.save("resp_old", [{"role": "user", "content": "x"}], [], {"id": "resp_old"})
    later = ResponseStore(str(tmp_path / "r.db"), clock=lambda: 1000.0 + 31 * 86400, retention_days=30)
    later.purge()
    assert later.get("resp_old") is None


def test_streamed_response_is_stored_when_complete(setup):
    client, _, store = setup
    r = client.post("/v1/responses", headers=AUTH, json={"model": "agent", "input": "hi", "stream": True})
    events = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
    assert events[-1]["type"] == "response.completed"
    assert store.get(events[-1]["response"]["id"]) is not None
