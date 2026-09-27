import litellm
import pytest

from gateway.config import Catalog, RealtimeChain, Step
from gateway.router import Exhausted, Gateway, UnknownModel
from tests import fake_upstream


def gateway(url, keys, extra_aliases=None):
    opts = {"api_base": url}
    aliases = {
        "chat": [Step("groq", "m1", opts), Step("openrouter", "m2", opts)],
        "embed": [Step("openai", "e1", opts)],
        **(extra_aliases or {}),
    }
    return Gateway(Catalog(aliases, RealtimeChain([], [], "chat", [])), keys)


async def chat(gw, model="chat", **kw):
    resp, _, _ = await gw.call("acompletion", model, messages=[{"role": "user", "content": "hi"}], **kw)
    return resp


async def test_first_step_answers_when_healthy(upstream_url):
    gw = gateway(upstream_url, {"groq": ["g1"], "openrouter": ["o1"]})
    assert (await chat(gw)).choices[0].message.content == "m1:g1"


async def test_rate_limited_step_falls_back_to_next_provider(upstream_url):
    gw = gateway(upstream_url, {"groq": ["rl"], "openrouter": ["o1"]})
    assert (await chat(gw)).choices[0].message.content == "m2:o1"


async def test_bad_key_rotates_to_the_next_key_of_the_same_step(upstream_url):
    gw = gateway(upstream_url, {"groq": ["auth", "g2"], "openrouter": ["o1"]})
    contents = {(await chat(gw)).choices[0].message.content for _ in range(3)}
    assert contents == {"m1:g2"}


async def test_bad_request_is_not_fallen_back(upstream_url):
    gw = gateway(upstream_url, {"groq": ["bad"], "openrouter": ["o1"]})
    with pytest.raises(litellm.BadRequestError):
        await chat(gw)
    assert all(key == "bad" for _, key in fake_upstream.calls)


async def test_steps_without_keys_are_left_out(upstream_url):
    gw = gateway(upstream_url, {"openrouter": ["o1"]})
    assert (await chat(gw)).choices[0].message.content == "m2:o1"


def test_unknown_alias_and_keyless_provider_are_rejected(upstream_url):
    gw = gateway(upstream_url, {"groq": ["g1"]})
    with pytest.raises(UnknownModel):
        gw.chain("nope")
    with pytest.raises(UnknownModel):
        gw.chain("mistral/mistral-small-latest")


async def test_provider_model_passthrough_uses_that_providers_keys(upstream_url):
    gw = gateway(upstream_url, {"groq": ["g1"], "openrouter": ["o1"]})
    assert (await chat(gw, "openrouter/m2")).choices[0].message.content == "m2:o1"


async def test_embeddings_use_the_alias(upstream_url):
    gw = gateway(upstream_url, {"openai": ["e1"]})
    resp, group, _ = await gw.call("aembedding", "embed", input=["a", "b"])
    assert group == "openai/e1"
    assert len(resp.data) == 2


def test_aliases_lists_only_usable_ones(upstream_url):
    gw = gateway(upstream_url, {"openrouter": ["o1"]})
    assert gw.aliases() == ["chat"]


async def test_all_steps_failing_raises_exhausted_with_retry_after(upstream_url):
    gw = gateway(upstream_url, {"groq": ["rl"], "openrouter": ["down"]})
    with pytest.raises(Exhausted) as info:
        await chat(gw)
    assert info.value.retry_after > 0


async def test_stream_falls_back_before_the_first_chunk(upstream_url):
    gw = gateway(upstream_url, {"groq": ["rl"], "openrouter": ["o1"]})
    stream = await chat(gw, stream=True)
    text = "".join([c.choices[0].delta.content or "" async for c in stream])
    assert text == "m2:o1"


async def test_call_reports_group_and_skipped_steps(upstream_url):
    gw = gateway(upstream_url, {"groq": ["rl"], "openrouter": ["o1"]})
    _, group, skipped = await gw.call("acompletion", "chat", messages=[{"role": "user", "content": "hi"}])
    assert (group, skipped) == ("openrouter/m2", 1)
