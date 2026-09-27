"""Live check against a running gateway with real keys: one call per usable alias of a mode,
with the tokens and cost the gateway recorded for each call.

    uv run python scripts/smoke.py [--mode free|paid] [--url http://127.0.0.1:8000/v1]
Reads GATEWAY_API_KEY from the environment (.env is loaded).
"""

import argparse
import asyncio
import base64
import os

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI


USAGE: dict = {}


def cost_note(client, headers) -> str:
    turn_id = headers.get("x-gateway-turn-id")
    if not turn_id:
        return ""
    turn = httpx.get(f"{client.base_url}usage/turns/{turn_id}", headers={"Authorization": f"Bearer {client.api_key}"}).json()
    parts = []
    for c in turn.get("components", []):
        counts = {k: c[k] for k in ("input", "cache_read", "output", "audio_seconds", "characters") if c.get(k)}
        cost = "unpriced" if c["cost"] is None else f"${c['cost']:.6f}"
        parts.append(f"{c['kind']} {counts} {cost} [{c['price_source']}]")
    return "  | " + "; ".join(parts)


async def check(name, coro):
    try:
        print(f"ok    {name:22} {await coro}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  {name:22} {type(exc).__name__}: {str(exc)[:160]}")
        return False


async def chat(client, alias):
    raw = await client.chat.completions.with_raw_response.create(
        model=alias, messages=[{"role": "user", "content": "Reply with one word: pong"}], max_tokens=20)
    return f"{raw.headers.get('x-gateway-model')} → {raw.parse().choices[0].message.content!r}" + cost_note(client, raw.headers)


async def embed(client, alias):
    raw = await client.embeddings.with_raw_response.create(model=alias, input="hello")
    return f"{raw.headers.get('x-gateway-model')} → {len(raw.parse().data[0].embedding)} dims" + cost_note(client, raw.headers)


async def speech_roundtrip(client, prefix):
    raw = await client.audio.speech.with_raw_response.create(model=f"{prefix}tts", voice="alloy", input="Hello from the gateway.", response_format="wav")
    audio = raw.read()
    text = await client.audio.transcriptions.with_raw_response.create(model=f"{prefix}stt", file=("hello.wav", audio))
    return (f"tts {raw.headers.get('x-gateway-model')} ({len(audio)} bytes) → "
            f"stt {text.headers.get('x-gateway-model')} {text.parse().text!r}" + cost_note(client, raw.headers)
            + cost_note(client, text.headers))


async def realtime(client, prefix):
    async with client.realtime.connect(model=f"{prefix}realtime") as conn:
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
                    usage = getattr(ev.response, "usage", None)
                    return f"{audio} bytes of audio | turn usage: {usage}"
                elif ev.type == "error":
                    raise RuntimeError(ev.error.message)


async def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--mode", choices=["free", "paid"], default="free")
    args = parser.parse_args()
    client = AsyncOpenAI(base_url=args.url, api_key=os.environ["GATEWAY_API_KEY"])
    health = httpx.get(args.url.removesuffix("/v1") + "/health").json()
    prefix = "" if args.mode == "free" else f"{args.mode}/"
    print("providers with keys:", ", ".join(health["modes"].get(args.mode, {})) or "none")
    print("pricing:", {k: v for k, v in health["pricing"].items() if k != "unpriced"})
    aliases = [a.removeprefix(prefix) for a in health["aliases"] if (a.startswith(prefix) if prefix else "/" not in a)]
    results = []
    for alias in aliases:
        if alias.startswith("chat"):
            results.append(await check(prefix + alias, chat(client, prefix + alias)))
        elif alias.startswith("embed"):
            results.append(await check(prefix + alias, embed(client, prefix + alias)))
    if "tts" in aliases and "stt" in aliases:
        results.append(await check(f"{prefix}tts→stt", speech_roundtrip(client, prefix)))
    results.append(await check(f"{prefix}realtime", realtime(client, prefix)))
    unpriced = httpx.get(args.url.removesuffix("/v1") + "/health").json()["pricing"]["unpriced"]
    if unpriced:
        print("\nno price known (add under prices: in the mode's yaml):", ", ".join(unpriced))
    print(f"\n{sum(results)}/{len(results)} passed")


if __name__ == "__main__":
    asyncio.run(main())
