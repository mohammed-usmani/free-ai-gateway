import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import Catalog, RealtimeChain, Step, load_catalog
from gateway.keypool import KeyPool
from gateway.modes import Modes
from gateway.router import Gateway, NoPrivateProvider
from tests import fake_upstream

AUTH = {"Authorization": "Bearer s"}


def catalog(url):
    o = {"api_base": url}
    return Catalog({"chat": [Step("gemini", "trains", {**o, "custom_llm_provider": "openai"}), Step("groq", "private", o)]},
                   RealtimeChain([], [], "chat", []), privacy={"gemini": "yes", "groq": "no"})


async def test_private_requests_skip_providers_that_train(upstream_url):
    g = Gateway(catalog(upstream_url), {"gemini": ["gm"], "groq": ["g"]})
    resp, group, _ = await g.call("acompletion", "chat", private=True, messages=[{"role": "user", "content": "hi"}])
    assert group == "groq/private" and all(k != "gm" for _, k in fake_upstream.calls)
    _, group, _ = await g.call("acompletion", "chat", messages=[{"role": "user", "content": "hi"}])
    assert group == "gemini/trains"


async def test_no_private_provider_is_explicit(upstream_url):
    o = {"api_base": upstream_url}
    cat = Catalog({"chat": [Step("gemini", "t", {**o, "custom_llm_provider": "openai"})]}, RealtimeChain([], [], "chat", []),
                  privacy={"gemini": "opt-out"})
    with pytest.raises(NoPrivateProvider):
        await Gateway(cat, {"gemini": ["gm"]}).call("acompletion", "chat", private=True, messages=[{"role": "user", "content": "x"}])


def test_header_turns_privacy_on_and_503s_when_impossible(upstream_url, tmp_path):
    c = TestClient(create_app(Modes({"free": Gateway(catalog(upstream_url), {"gemini": ["gm"]})}), KeyPool(str(tmp_path / "k.db")), "s"))
    r = c.post("/v1/chat/completions", headers={**AUTH, "x-gateway-private": "true"},
               json={"model": "chat", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503 and r.json()["error"]["type"] == "no_private_provider"


def test_shipped_catalogs_mark_privacy():
    free, paid = load_catalog("providers.free.yaml"), load_catalog("providers.paid.yaml")
    assert free.privacy["groq"] == "no" and free.privacy["gemini"] == "yes" and free.privacy["mistral"] == "opt-out"
    assert paid.privacy["openai"] == "no" and paid.privacy["anthropic"] == "no"


def test_yaml_booleans_are_read_as_privacy_words(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("aliases: {}\nrealtime: {s2s: [], stt: [], llm: x, tts: []}\nproviders:\n  groq: {trains_on_data: no}\n  x: {trains_on_data: yes}\n")
    c = load_catalog(p)
    assert c.privacy == {"groq": "no", "x": "yes"}


def test_realtime_private_options_exclude_training_providers(tmp_path):
    from gateway.realtime.backends import realtime_options_for

    free = Catalog({"chat-fast": []}, RealtimeChain([Step("gemini", "live")], [Step("deepgram", "n"), Step("speechmatics", "e")],
                                                    "chat-fast", [Step("rime", "m", {"voice": "v"})]),
                   privacy={"gemini": "yes", "deepgram": "opt-out", "speechmatics": "no", "rime": "no"})
    keys = {"free": {"gemini": ["g"], "deepgram": ["d"], "speechmatics": ["s"], "rime": ["r"]}}
    pool = KeyPool(str(tmp_path / "k.db"))
    opts = realtime_options_for("realtime", {"free": free}, keys, pool, "http://x/v1", "k", None, private=True)
    built = [o() for o in opts]
    assert [b.name for b in built if b] == ["cascade"]
    procs = built[-1]._build(built[-1], {}, None)
    assert [type(s).__name__ for s in procs[0].services] == ["SpeechmaticsSTTService"]
