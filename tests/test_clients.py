"""The official OpenAI SDK, pointed at a live gateway, must just work."""

import base64
import socket
import threading
import time

import pytest
import uvicorn
from openai import AsyncOpenAI

from gateway.app import create_app
from gateway.config import Catalog, RealtimeChain, Step
from gateway.keypool import KeyPool
from gateway.realtime.session import RealtimeSession
from gateway.modes import Modes
from gateway.router import Gateway
from tests.test_realtime_session import FakeBackend, option


@pytest.fixture(scope="module")
def gateway_url(upstream_url, tmp_path_factory):
    o = {"api_base": upstream_url}
    catalog = Catalog({"chat-fast": [Step("groq", "m1", o), Step("openrouter", "m2", o)],
                       "embed-x": [Step("openai", "e1", o)],
                       # LiteLLM translates Responses to chat for these; OpenRouter/OpenAI would
                       # need the fake to implement /v1/responses itself.
                       "agent": [Step("groq", "m1", o), Step("sambanova", "m3", o)],
                       "image": [Step("pollinations", "flux", {"custom_llm_provider": "openai", "api_base": upstream_url})],
                       "moderation": [Step("openai", "omni-moderation-latest", o)]},
                      RealtimeChain([], [], "chat-fast", []), model_map={"claude-*": "chat-fast"})
    pool = KeyPool(str(tmp_path_factory.mktemp("k") / "k.db"))
    gw = Gateway(catalog, {"groq": ["rl"], "openrouter": ["o1"], "openai": ["e1"], "pollinations": ["p"], "sambanova": ["s1"]})
    app = create_app(Modes({"free": gw}), pool, "gw-secret",
                     realtime=lambda ws: RealtimeSession(ws, [option([FakeBackend("fake", [])])], pool).run())
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.01)
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True


@pytest.fixture
def client(gateway_url):
    return AsyncOpenAI(base_url=gateway_url, api_key="gw-secret", max_retries=0)


async def test_sdk_chat_with_fallback(client):
    resp = await client.chat.completions.create(model="chat-fast", messages=[{"role": "user", "content": "hi"}])
    assert resp.choices[0].message.content == "m2:o1"


async def test_sdk_streaming_chat(client):
    stream = await client.chat.completions.create(model="chat-fast", stream=True,
                                                  messages=[{"role": "user", "content": "hi"}])
    assert "".join([c.choices[0].delta.content or "" async for c in stream]) == "m2:o1"


async def test_sdk_embeddings(client):
    resp = await client.embeddings.create(model="embed-x", input="hello")
    assert resp.data[0].embedding == [0.1, 0.2, 0.3]


async def test_sdk_realtime_socket(client):
    async with client.realtime.connect(model="realtime") as conn:
        assert (await conn.recv()).type == "session.created"
        await conn.session.update(session={"type": "realtime", "instructions": "be brief"})
        await conn.input_audio_buffer.append(audio=base64.b64encode(b"\x00\x01").decode())
        kinds = []
        async for ev in conn:
            kinds.append(ev.type)
            if ev.type == "response.output_audio.delta":
                assert base64.b64decode(ev.delta) == b"fake"
                break
        assert "session.updated" in kinds


async def test_openai_agents_sdk_runs_on_the_responses_endpoint(gateway_url):
    from agents import Agent, OpenAIResponsesModel, Runner, set_tracing_disabled

    set_tracing_disabled(True)
    client = AsyncOpenAI(base_url=gateway_url, api_key="gw-secret", max_retries=0)
    model = OpenAIResponsesModel(model="agent", openai_client=client)
    result = await Runner.run(Agent(name="a", instructions="be brief", model=model), "hi")
    assert result.final_output == "m3:s1"


async def test_anthropic_sdk_on_the_messages_endpoint(gateway_url):
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(base_url=gateway_url.removesuffix("/v1"), api_key="gw-secret", max_retries=0)
    msg = await client.messages.create(model="claude-sonnet-4-5", max_tokens=50, messages=[{"role": "user", "content": "hi"}])
    # LiteLLM's messages adapter keeps the "openrouter/" prefix on the model name (a LiteLLM quirk);
    # the fake echoes it, a real OpenRouter "not a valid model" 400 now falls back instead.
    assert msg.content[0].text.endswith("m2:o1")
    async with client.messages.stream(model="claude-haiku-4-5", max_tokens=50,
                                      messages=[{"role": "user", "content": "hi"}]) as stream:
        text = "".join([t async for t in stream.text_stream])
    assert text.endswith("m2:o1")


async def test_sdk_images_and_moderation(client):
    img = await client.images.generate(model="image", prompt="a cat")
    assert img.data[0].b64_json
    mod = await client.moderations.create(model="moderation", input="I will kill")
    assert mod.results[0].flagged is True
