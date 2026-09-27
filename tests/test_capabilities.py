import pytest

from gateway.capabilities import Needs, needs_for, step_can
from gateway.config import Catalog, RealtimeChain, Step
from gateway.router import Gateway, NoCapableStep
from tests import fake_upstream

IMAGE_MSG = [{"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                          {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
TEXT_MSG = [{"role": "user", "content": "hi"}]


def test_needs_are_read_from_the_request():
    n = needs_for("acompletion", {"messages": IMAGE_MSG, "tools": [{"type": "function", "function": {"name": "f"}}],
                                  "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}})
    assert (n.vision, n.tools, n.json_schema) == (True, True, True)
    assert n.prompt_tokens > 0
    assert needs_for("acompletion", {"messages": TEXT_MSG}) == Needs(prompt_tokens=needs_for("acompletion", {"messages": TEXT_MSG}).prompt_tokens)
    assert needs_for("aresponses", {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}]}).vision
    assert needs_for("aanthropic_messages", {"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]}).vision


def test_step_capability_uses_overrides_then_litellm_info():
    assert not step_can(Step("x", "m", {"supports": {"vision": False}}), Needs(vision=True))[0]
    assert step_can(Step("x", "m", {"supports": {"vision": True}}), Needs(vision=True))[0]
    assert step_can(Step("unknownprov", "unknown-model"), Needs(vision=True, tools=True))[0]  # unknown → try
    assert not step_can(Step("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free"), Needs(vision=True))[0]
    ok, reason = step_can(Step("x", "m", {"supports": {"max_input_tokens": 100}}), Needs(prompt_tokens=500))
    assert not ok and "context" in reason


def gw(url, steps, keys):
    return Gateway(Catalog({"chat": steps}, RealtimeChain([], [], "chat", [])), keys)


async def ask(g, messages=TEXT_MSG):
    resp, group, skipped = await g.call("acompletion", "chat", messages=messages)
    return resp.choices[0].message.content, group, skipped


async def test_image_request_skips_text_only_steps(upstream_url):
    o = {"api_base": upstream_url}
    g = gw(upstream_url, [Step("groq", "text-only", {**o, "supports": {"vision": False}}), Step("openrouter", "seer", o)],
           {"groq": ["g"], "openrouter": ["o"]})
    assert await ask(g, IMAGE_MSG) == ("seer:o", "openrouter/seer", 1)
    assert all(k != "g" for _, k in fake_upstream.calls)
    assert (await ask(g))[1] == "groq/text-only"


async def test_no_capable_step_is_a_clear_error(upstream_url):
    o = {"api_base": upstream_url}
    g = gw(upstream_url, [Step("groq", "t", {**o, "supports": {"vision": False}})], {"groq": ["g"]})
    with pytest.raises(NoCapableStep, match="vision"):
        await ask(g, IMAGE_MSG)


async def test_unsupported_feature_and_context_errors_fall_back(upstream_url):
    o = {"api_base": upstream_url}
    for bad_key in ("noimg", "ctx"):
        g = gw(upstream_url, [Step("groq", "a", o), Step("openrouter", "b", o)], {"groq": [bad_key], "openrouter": ["o"]})
        assert (await ask(g))[0] == "b:o"


async def test_plain_bad_requests_still_return(upstream_url):
    import litellm
    o = {"api_base": upstream_url}
    g = gw(upstream_url, [Step("groq", "a", o), Step("openrouter", "b", o)], {"groq": ["bad"], "openrouter": ["o"]})
    with pytest.raises(litellm.BadRequestError):
        await ask(g)


def test_gateway_options_are_not_sent_to_providers(upstream_url):
    g = gw(upstream_url, [Step("groq", "a", {"api_base": upstream_url, "supports": {"vision": False}, "first_byte_timeout": 5})],
           {"groq": ["g"]})
    params = g.router.model_list[0]["litellm_params"]
    assert "supports" not in params and "first_byte_timeout" not in params


def test_app_reports_no_capable_step_as_400(upstream_url, tmp_path):
    from fastapi.testclient import TestClient

    from gateway.app import create_app
    from gateway.keypool import KeyPool
    from gateway.modes import Modes

    g = gw(upstream_url, [Step("groq", "t", {"api_base": upstream_url, "supports": {"vision": False}})], {"groq": ["g"]})
    client = TestClient(create_app(Modes({"free": g}), KeyPool(str(tmp_path / "k.db")), "s"))
    r = client.post("/v1/chat/completions", headers={"Authorization": "Bearer s"}, json={"model": "chat", "messages": IMAGE_MSG})
    assert r.status_code == 400 and r.json()["error"]["type"] == "unsupported_request"


async def test_paid_chain_that_cannot_do_it_falls_back_to_free(upstream_url):
    from gateway.modes import Modes

    o = {"api_base": upstream_url}
    free = Gateway(Catalog({"chat": [Step("openrouter", "seer", o)]}, RealtimeChain([], [], "chat", [])), {"openrouter": ["o"]})
    paid = Gateway(Catalog({"chat": [Step("openai", "blind", {**o, "supports": {"vision": False}})]},
                           RealtimeChain([], [], "chat", []), fallback_to={"chat": "free/chat"}), {"openai": ["p"]})
    answer = await Modes({"free": free, "paid": paid}).call("acompletion", "paid/chat", messages=IMAGE_MSG)
    assert answer.mode == "free"


def test_retired_model_ids_fall_back():
    import litellm
    from gateway.router import should_fall_back

    exc = litellm.BadRequestError("nvidia/x:free is not a valid model ID", "openrouter", "x")
    assert should_fall_back(exc)
