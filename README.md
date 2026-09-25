# Free AI Gateway

One OpenAI-compatible endpoint over the **free tiers that need no card**: chat LLMs, embeddings,
speech-to-text, text-to-speech and realtime voice. It rotates your keys per provider and falls
back to the next provider when one is down, rate-limited or rejects a key.

Point anything that speaks OpenAI at it (the OpenAI SDKs, LangChain, LiveKit, Pipecat) by
changing `base_url`.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="<GATEWAY_API_KEY>")
client.chat.completions.create(model="chat-smart", messages=[{"role": "user", "content": "hi"}])
```

## Endpoints

| Endpoint | Models (aliases in `providers.yaml`) |
|---|---|
| `POST /v1/chat/completions` (streaming too, tools) | `chat-smart`, `chat-fast` |
| `POST /v1/embeddings` | `embed-gemini`, `embed-mistral`, `embed-bge-m3` |
| `POST /v1/audio/transcriptions` | `stt` |
| `POST /v1/audio/speech` | `tts` |
| `WS /v1/realtime` (OpenAI Realtime protocol, PCM16 24 kHz) | Gemini Live, else STT → `chat-fast` → TTS |
| `GET /v1/models`, `GET /health` | usable aliases; key states (masked) |

Any `provider/model` (e.g. `groq/openai/gpt-oss-20b`) also works: that provider's keys rotate,
without cross-provider fallback.

Responses carry `x-gateway-model` (which step answered) and `x-gateway-fallbacks` (steps skipped).

## How fallback works

- Each alias is an ordered chain of steps; a step is skipped when its provider has no keys.
- Within a step, keys rotate; a key that gets 401/403/404/408/429/5xx cools down (LiteLLM).
- A failing step moves to the next. A **400 is returned to you**, not retried elsewhere.
- Streams fall back only before the first chunk; after that an error event ends the stream.
- Embedding aliases hold one model each: vectors from different models don't mix.
- Realtime: the session starts on the first speech-to-speech step with a usable key, else the
  cascade. If the backend fails mid-session, its key is cooled down (SQLite, survives restarts),
  the next backend starts with your `session.update` config, and the socket stays open.
  Conversation history is not carried over. Cascade STT and TTS fail over inside the session.

## Run

```bash
cp .env.example .env         # set GATEWAY_API_KEY and whichever provider keys you have
uv sync
make run                     # uvicorn on :8000
make smoke                   # one real call per usable alias + a realtime turn
make test                    # offline suite (fake upstream, no keys, no quota)
```

Docker: `docker compose up -d` (binds 127.0.0.1:8000; put a TLS proxy in front for remote use).
Needs about 320 MB RAM idle.

## Providers (checked 2026-09-26)

Text: Mistral (Experiment plan), Groq, NVIDIA NIM, OpenRouter `:free`, Google AI Studio,
SambaNova, Z.ai, Cloudflare Workers AI. Voice: Gemini Live, Deepgram ($200 credit),
AssemblyAI ($50), Speechmatics ($100), Cartesia and ElevenLabs (monthly, non-commercial), Rime.

Left out (card, top-up or retired): Cerebras, GitHub Models, Together, Moonshot, Scaleway,
Nebius, Chutes, OpenAI Realtime, Google Cloud/Azure speech, Alibaba, Gladia, Soniox.

Things to know:
- Keys in one account share one quota (Groq, Google). OpenRouter, Cohere and Hume forbid extra
  accounts to get around limits; spread load across providers instead.
- Gemini's free tier may use your data (human review) and can't serve EEA/UK users.
  Mistral trains on Experiment-plan data unless you opt out.
- Free model lists churn (OpenRouter especially). Model ids are pinned in `providers.yaml`;
  `make smoke` tells you which ones stopped working.

## Layout

```
gateway/config.py            keys from env, providers.yaml
gateway/router.py            LiteLLM key rotation + our step-to-step fallback
gateway/keypool.py           SQLite key health for realtime
gateway/app.py               HTTP + WebSocket API
gateway/realtime/session.py  Realtime protocol, backend failover
gateway/realtime/backends.py Pipecat pipelines: Gemini Live, cascade
```
