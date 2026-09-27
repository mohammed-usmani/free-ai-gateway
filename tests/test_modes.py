import pytest

from gateway.config import Catalog, ModeSettings, RealtimeChain, Step
from gateway.modes import Modes
from gateway.router import Exhausted, Gateway
from tests import fake_upstream


def modes(url, free_keys, paid_keys, paid_fallback=True):
    o = {"api_base": url}
    free = Catalog({"chat": [Step("groq", "free-m", o)]}, RealtimeChain([], [], "chat", []))
    paid = Catalog({"chat": [Step("openai", "paid-m", o)], "solo": [Step("openai", "solo-m", o)]},
                   RealtimeChain([], [], "chat", []),
                   settings=ModeSettings(cooldown_seconds=5, num_retries=0, first_byte_timeout=8),
                   fallback_to={"chat": "free/chat"} if paid_fallback else {})
    return Modes({"free": Gateway(free, free_keys), "paid": Gateway(paid, paid_keys)})


async def ask(m, model):
    return await m.call("acompletion", model, messages=[{"role": "user", "content": "hi"}])


async def test_prefix_picks_the_mode_and_bare_means_free(upstream_url):
    m = modes(upstream_url, {"groq": ["g"]}, {"openai": ["p"]})
    assert (await ask(m, "chat")).resp.choices[0].message.content == "free-m:g"
    assert (await ask(m, "free/chat")).mode == "free"
    answer = await ask(m, "paid/chat")
    assert (answer.mode, answer.resp.choices[0].message.content) == ("paid", "paid-m:p")


async def test_exhausted_paid_chain_falls_back_to_free(upstream_url):
    m = modes(upstream_url, {"groq": ["g"]}, {"openai": ["down"]})
    answer = await ask(m, "paid/chat")
    assert (answer.mode, answer.group, answer.resp.choices[0].message.content) == ("free", "groq/free-m", "free-m:g")


async def test_paid_alias_without_keys_falls_back_too(upstream_url):
    m = modes(upstream_url, {"groq": ["g"]}, {})
    assert (await ask(m, "paid/chat")).mode == "free"


async def test_no_fallback_configured_raises(upstream_url):
    m = modes(upstream_url, {"groq": ["g"]}, {"openai": ["down"]}, paid_fallback=False)
    with pytest.raises(Exhausted):
        await ask(m, "paid/chat")


async def test_free_and_paid_keys_never_mix(upstream_url):
    m = modes(upstream_url, {"groq": ["g"], "openai": ["free-openai"]}, {"openai": ["p"]})
    await ask(m, "paid/chat")
    await ask(m, "chat")
    assert {k for _, k in fake_upstream.calls} == {"p", "g"}


def test_split_keeps_provider_model_passthrough(upstream_url):
    m = modes(upstream_url, {}, {"openai": ["p"]})
    assert m.split("paid/openai/x") == ("paid", "openai/x")
    assert m.split("openai/x") == ("free", "openai/x")


def test_aliases_are_listed_per_mode(upstream_url):
    m = modes(upstream_url, {"groq": ["g"]}, {"openai": ["p"]})
    assert m.aliases() == ["chat", "paid/chat", "paid/solo"]
    assert m.is_alias("paid/chat") and m.is_alias("chat") and not m.is_alias("paid/nope")
