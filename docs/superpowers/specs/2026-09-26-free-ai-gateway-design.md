# Free AI Gateway — design

Date: 2026-09-26. Status: approved (design reviewed with the owner in conversation; spec self-reviewed).

## Goal

One self-hosted, OpenAI-compatible service that fronts every useful **no-card free tier** for
chat LLMs, embeddings, speech-to-text, text-to-speech and realtime voice. It rotates keys per
provider and falls back across providers when one is down, rate-limited or rejects a key, so
agent experiments and portfolio demos run without paying.

Callers are the owner's own backends and scripts (one gateway key). Browsers never call it.

## Non-goals

- Browser clients, ephemeral tokens, per-site quotas.
- Multi-account quota farming. Keys are the owner's legitimate keys; see "Terms".
- Translating conversation history between voice backends.
- Paid providers (nothing requiring a card or top-up is in the default catalog).
- Admin UI. Config is files + env.

## Architecture

```
app ──Bearer GATEWAY_API_KEY──▶ FastAPI (one process, uvicorn)
  POST /v1/chat/completions ─┐
  POST /v1/embeddings ───────┤
  POST /v1/audio/transcriptions ─┼─▶ LiteLLM Router (key rotation, fallbacks, cooldowns)
  POST /v1/audio/speech ─────┘
  WS   /v1/realtime ──────────────▶ RealtimeSession
                                     ├─ speech-to-speech backends (Pipecat Gemini Live …)
                                     └─ cascade backend (Pipecat: VAD + STT → LLM → TTS)
                                     key health: SQLite (KeyPool)
  GET  /health, GET /v1/models
```

Decision record (from the design conversation):

- **Service, not library.** Apps decide how to use it; OpenAI formats so LiveKit, Pipecat,
  the OpenAI SDKs and LangChain plug in by changing `base_url`.
- **LiteLLM for request/response endpoints** (chat, embeddings, batch STT, batch TTS). It
  already implements provider formats, key rotation, fallbacks and cooldowns (verified in
  litellm 1.102.1: `_is_cooldown_required` cools down on 401, 404, 408, 429 and 5xx; it
  does not cool down other 4xx). The original design had our own code for batch STT/TTS;
  LiteLLM 1.102 supports transcription for openai/groq/deepgram/elevenlabs/gemini/mistral/
  soniox and speech for openai/elevenlabs/vertex/minimax, so it is used there too and our own
  code is reserved for gaps (ruling: less code, same behaviour).
- **Own code for realtime voice**, running Pipecat pipelines internally so VAD, turn
  detection and provider streaming come from Pipecat.
- **Key health for realtime in SQLite** (stdlib, survives restart, shared by processes).
  LiteLLM keeps its own in-memory cooldowns; a restart costs at most one failed request per
  exhausted key.

## Configuration

- Keys: env vars, comma-separated: `GROQ_API_KEYS=k1,k2`. A provider without keys is
  skipped everywhere. Single-key `*_API_KEY` is accepted as a one-element list.
- Non-key provider settings (e.g. `CLOUDFLARE_ACCOUNT_ID`) are plain env vars.
- `GATEWAY_API_KEY` (required): callers send `Authorization: Bearer <it>`.
- `providers.yaml` (committed, no secrets): for each **task alias** an ordered chain of
  steps `{provider, model, …step options}`. Task aliases: `chat-smart`, `chat-fast`,
  `embed-<name>`, `stt`, `tts`, `realtime`.
- A caller may pass a concrete `provider/model` instead of an alias; it gets that provider's
  key rotation but no cross-provider fallback.

## Fallback rules

| Condition | Behaviour |
|---|---|
| 401 / 403 | key cooled down long (realtime: 24 h in SQLite; keys are identified by hash, so a replaced key starts fresh), warning logged, next key |
| 429 | key cooled down (`retry-after` if given, else configured cooldown), next key, then next step |
| 5xx, connect error, timeout to first byte | short cooldown, next step |
| other 4xx (bad request) | returned to the caller, no fallback |
| every step exhausted | 503 with `retry-after` = soonest key recovery (realtime: `error` event) |

- Streaming chat/TTS fall back only before the first chunk is sent.
- Embedding chains never cross models: each `embed-<name>` alias holds one model, possibly
  on several keys/hosts serving the identical model. Changing model = new alias + re-index.
- Responses carry `x-gateway-provider`, `x-gateway-model` headers where the path allows.
- `GET /health` lists each provider: keys present (masked), realtime key states.

## Realtime voice (`WS /v1/realtime`)

Speaks the server side of the **OpenAI Realtime GA protocol**, reusing Pipecat's event models
(`pipecat.services.openai.realtime.events`). Supported:

- client: `session.update` (instructions, tools, voice, audio format), `input_audio_buffer.append`,
  `input_audio_buffer.commit`/`clear`, `conversation.item.create` (user text message,
  `function_call_output`), `response.create`, `response.cancel`.
- server: `session.created`/`updated`, `input_audio_buffer.speech_started`/`speech_stopped`,
  `response.created`, `response.output_audio.delta`/`done`,
  `response.output_audio_transcript.delta`/`done`, `response.function_call_arguments.done`,
  `response.done`, `conversation.item.input_audio_transcription.completed`, `error`.
- audio: PCM16 mono 24 kHz both ways (OpenAI default).

Backend order (the "realtime" chain in `providers.yaml`):

1. speech-to-speech steps whose provider has a usable key (e.g. Gemini Live),
2. the cascade step: STT → LLM → TTS, each an ordered list with its own key rotation.

The session keeps the client's `session.update` config. When the active backend fails (setup
error, auth/quota error, or its pipeline errors mid-session), the session marks the key in
SQLite, builds the next backend, applies the stored config and keeps the client socket open.
**Conversation history is not carried over**; the next backend answers the next user turn.
The next backend's key is resolved ahead of time; its connection is not pre-opened (an idle
Gemini Live session would consume concurrency quota).

Inside the cascade, STT and TTS lists use Pipecat's `ServiceSwitcher` with the failover
strategy, so a failing STT or TTS provider is swapped without rebuilding the session.

Tool calls: the gateway forwards the model's function call to the client as
`response.function_call_arguments.done` and waits for the client's `function_call_output`
(timeout 30 s), then hands the result back to the backend.

## Provider catalog (researched 2026-09-26, no card required)

Excluded (card, top-up, retired): Cerebras (card since Jul 2026), GitHub Models (retired
2026-07-30), Together, Moonshot, Scaleway, Nebius, Chutes, OpenAI Realtime, Azure/Google Cloud
speech, Alibaba (card bind), Gladia, Soniox (free credits ended), LMNT, PlayHT, Resemble.

Default chains shipped in `providers.yaml` (every step is skipped when its keys are absent):

| Alias | Chain |
|---|---|
| `chat-smart` (tools) | mistral/mistral-medium-latest → groq/openai/gpt-oss-120b → nvidia_nim/moonshotai/kimi-k3 → openrouter/nvidia/nemotron-3-ultra-550b-a55b:free → openrouter/qwen/qwen3.8-27b:free → gemini/gemini-3.7-flash → sambanova/DeepSeek-V3.2 |
| `chat-fast` | groq/openai/gpt-oss-20b → gemini/gemini-3.5-flash-lite → zai/glm-4.7-flash → mistral/mistral-small-latest → cloudflare/@cf/openai/gpt-oss-120b |
| `embed-gemini` | gemini/gemini-embedding-2 |
| `embed-mistral` | mistral/mistral-embed |
| `embed-bge-m3` | cloudflare/@cf/baai/bge-m3 |
| `stt` (batch) | groq/whisper-large-v3-turbo → deepgram/nova-3 → mistral/voxtral-mini-latest → elevenlabs/scribe_v1 |
| `tts` (batch) | gemini/gemini-3.8-flash-lite-tts → groq/canopylabs/orpheus-v1-english (≤200 chars) → elevenlabs/eleven_flash_v2_5 |
| `realtime` s2s | Gemini Live `gemini-3.8-live` (Pipecat `GeminiLiveLLMService`) |
| `realtime` cascade STT | deepgram (nova-3) → assemblyai → speechmatics → cartesia (ink) |
| `realtime` cascade LLM | the gateway's own `chat-fast` alias (loopback), so it inherits LiteLLM fallback |
| `realtime` cascade TTS | deepgram (aura-2) → cartesia (sonic-3) → rime → elevenlabs |

Batch TTS has no Deepgram step: LiteLLM 1.102 has no Deepgram speech adapter (verified in
source), and adding our own is deferred until a need shows up. Model ids are pinned in
`providers.yaml`; free lists (OpenRouter especially) churn, so the smoke test is the check.
Hosted-cascade "agent" APIs (Deepgram Voice Agent, Hume EVI, Ultravox) are not v1 s2s steps:
each has its own protocol and is itself a cascade.

Free-tier caveats worth knowing: Gemini free data may be human-reviewed and can't serve EEA/UK
users; Cartesia and ElevenLabs free tiers are non-commercial; NVIDIA phone verification is
unreliable for +91; Gemini Flash free is ~20 requests/day.

## Terms

Several keys in one account/project share one quota at Groq and Google, and OpenRouter and
Cohere explicitly ban multiple accounts to bypass limits. The gateway rotates whatever keys
are configured; spreading load across different providers is the supported way to raise
free capacity.

## Testing

- Unit: KeyPool (rotation, 401 disable, 429 retry-after, persistence across instances),
  config loading (CSV keys, providers without keys skipped).
- Endpoints via FastAPI TestClient with LiteLLM `mock_response` (success and mocked
  errors): fallback order, 400 not retried, auth required, OpenAI response shape,
  embed alias never crossing models, 503 when exhausted.
- Realtime: scripted client speaking the Realtime protocol against fake Pipecat backends;
  primary dies mid-session → config re-applied, next response from the fallback, socket open.
- `make smoke`: opt-in live checks, one real request per configured provider plus a short
  real realtime session.
- Client checks: OpenAI Python SDK against chat/embeddings/audio; Pipecat's
  `OpenAIRealtimeLLMService` pointed at `ws://…/v1/realtime`.
