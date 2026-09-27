# Free AI Gateway

One OpenAI-compatible endpoint for chat LLMs, embeddings, speech-to-text, text-to-speech and
realtime voice, in two modes:

- **free** — the free tiers that need no card; stretches quota across providers.
- **paid** — OpenAI, Anthropic, Gemini and paid voice; tuned for reliability, and can fall back
  to free as a last resort.

It rotates keys per provider, falls back to the next provider when one is down, rate-limited or
rejects a key, and records **token usage and cost per turn** for every model in both modes.

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

Paid aliases (`providers.paid.yaml`): `paid/chat-best`, `paid/chat-smart`, `paid/chat-fast`,
`paid/embed-openai-small|large`, `paid/stt`, `paid/tts`, and `/v1/realtime?model=paid/realtime`
(OpenAI Realtime → Gemini Live → paid cascade → free realtime). A bare alias means free.

Any `provider/model` (e.g. `groq/openai/gpt-oss-20b`, or `paid/anthropic/claude-sonnet-5`) also
works: that provider's keys rotate, without cross-provider fallback.

Responses carry `x-gateway-mode` and `x-gateway-model` (what answered), `x-gateway-fallbacks`
(steps skipped) and `x-gateway-turn-id`.

## Modes

| | free | paid |
|---|---|---|
| Catalog | `providers.free.yaml` | `providers.paid.yaml` |
| Keys | `GROQ_API_KEYS=…` | `PAID_OPENAI_API_KEYS=…` (never shared with free) |
| 429 cooldown | 60 s | 5 s, 2 quick retries |
| Stream must start within | 20 s | 8 s |
| When the chain is exhausted | 503 | `fallback_to` (e.g. `free/chat-fast`), response says `x-gateway-mode: free` |

## Usage and cost

Every call is a **turn** made of priced **components**. A chat call is one turn with one
component; a realtime cascade response is one turn with `stt` (seconds), `llm` (tokens, from the
provider that really answered) and `tts` (characters); speech-to-speech is one `s2s` component
with text/audio/cached tokens. Tokens are stored as disjoint buckets: uncached input, cache read,
cache write, output (reasoning shown separately), audio in/out.

```bash
curl -H "Authorization: Bearer $GATEWAY_API_KEY" localhost:8000/v1/usage?group_by=model   # totals
curl ... localhost:8000/v1/usage/turns/<x-gateway-turn-id>                               # breakdown
curl ... "localhost:8000/v1/usage/turns?conversation_id=<id>"   # send x-gateway-conversation-id
curl -X POST ... localhost:8000/v1/usage/reprice?from=2026-09-01                         # recompute
```
`group_by`: model, provider, mode, alias, endpoint, kind, day. Realtime clients also get the
turn's breakdown in `response.done.response.usage`. Prompts, outputs and keys are never stored.

**Prices** come from, in order: the cost the provider reports (OpenRouter), your `prices:`
overrides (with `effective_from` dates), LiteLLM's price table re-fetched every
`PRICE_REFRESH_HOURS` (last good copy kept on failure), then the table bundled with LiteLLM.
Unknown price → cost is recorded as `null`, never guessed; `/health` lists those models under
`pricing.unpriced` along with the price table's age. Streaming voice providers aren't in
LiteLLM's table — add their rates under `prices:` to see their cost.

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
make smoke                   # one real call per usable free alias + a realtime turn, with cost
make smoke-paid              # the same for paid mode
make usage                   # usage totals by model
make test                    # offline suite (fake upstream, no keys, no quota)
```

If you run uvicorn on a port other than 8000, set `PORT` (or `GATEWAY_SELF_URL`) to match: the
voice cascade calls the gateway's own chat endpoint there. The startup log prints the URL it uses.

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
gateway/modes.py             free/paid dispatch and cross-mode fallback
gateway/pricing.py           layered prices, token buckets
gateway/usage.py             turns and components in SQLite
gateway/keypool.py           SQLite key health for realtime
gateway/app.py               HTTP + WebSocket API
gateway/realtime/session.py  Realtime protocol, backend failover
gateway/realtime/backends.py Pipecat pipelines: Gemini Live, cascade
```
