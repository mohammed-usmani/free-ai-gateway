"""Live check against a running gateway with real keys: one call per usable alias.

    uv run python scripts/smoke.py [--url http://127.0.0.1:8000/v1]
Reads GATEWAY_API_KEY from the environment (.env is loaded).
"""

import argparse
import asyncio
import base64
import os

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI


async def check(name, coro):
    try:
        print(f"ok    {name:14} {await coro}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  {name:14} {type(exc).__name__}: {str(exc)[:160]}")
        return False


async def chat(client, alias):
    raw = await client.chat.completions.with_raw_response.create(
        model=alias, messages=[{"role": "user", "content": "Reply with one word: pong"}], max_tokens=20)
    return f"{raw.headers.get('x-gateway-model')} → {raw.parse().choices[0].message.content!r}"


async def embed(client, alias):
    raw = await client.embeddings.with_raw_response.create(model=alias, input="hello")
    return f"{raw.headers.get('x-gateway-model')} → {len(raw.parse().data[0].embedding)} dims"


async def speech_roundtrip(client):
    raw = await client.audio.speech.with_raw_response.create(model="tts", voice="alloy", input="Hello from the gateway.", response_format="wav")
    audio = raw.read()
    text = await client.audio.transcriptions.with_raw_response.create(model="stt", file=("hello.wav", audio))
    return (f"tts {raw.headers.get('x-gateway-model')} ({len(audio)} bytes) → "
            f"stt {text.headers.get('x-gateway-model')} {text.parse().text!r}")


async def realtime(client):
    async with client.realtime.connect(model="realtime") as conn:
        await conn.recv()
        await conn.session.update(session={"type": "realtime", "instructions": "Answer in five words."})
        await conn.conversation.item.create(item={"type": "message", "role": "user",
                                                  "content": [{"type": "input_text", "text": "Say hello."}]})
        await conn.response.create()
        audio = 0
        async with asyncio.timeout(30):
            async for ev in conn:
                if ev.type == "response.output_audio.delta":
                    audio += len(base64.b64decode(ev.delta))
                elif ev.type == "response.done" and audio:
                    return f"{audio} bytes of audio"
                elif ev.type == "error":
                    raise RuntimeError(ev.error.message)


async def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()
    client = AsyncOpenAI(base_url=args.url, api_key=os.environ["GATEWAY_API_KEY"])
    health = httpx.get(args.url.removesuffix("/v1") + "/health").json()
    print("providers with keys:", ", ".join(health["providers"]) or "none")
    aliases = health["aliases"]
    results = []
    for alias in aliases:
        if alias.startswith("chat"):
            results.append(await check(alias, chat(client, alias)))
        elif alias.startswith("embed"):
            results.append(await check(alias, embed(client, alias)))
    if "tts" in aliases and "stt" in aliases:
        results.append(await check("tts→stt", speech_roundtrip(client)))
    results.append(await check("realtime", realtime(client)))
    print(f"\n{sum(results)}/{len(results)} passed")


if __name__ == "__main__":
    asyncio.run(main())
