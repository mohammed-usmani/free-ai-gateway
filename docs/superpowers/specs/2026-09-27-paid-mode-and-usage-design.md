# Paid mode, live pricing and per-turn usage — design

Date: 2026-09-27. Status: approved in conversation (parts 1–3, pricing layers, turn model); spec self-reviewed.
Builds on `2026-09-26-free-ai-gateway-design.md`.

## Goals

1. A **paid** operating mode next to **free** in the same gateway: same endpoints and fallback
   engine, its own catalog, keys and retry settings, tuned for reliability.
2. **Usage tracking for every model in both modes** (totals only, no per-caller keys): input,
   output, cached-read, cache-write, reasoning and audio tokens; STT seconds; TTS characters;
   cost in USD.
3. **Prices kept current** from layered sources, with raw counts stored so cost can be
   recomputed when a price was wrong.
4. **One record per turn** with a component breakdown — for realtime cascade: STT + LLM + TTS as
   one turn.

## Modes

- Catalogs: `providers.free.yaml` (today's `providers.yaml`, renamed) and `providers.paid.yaml`.
  Each has `settings:`, `aliases:`, `realtime:`; the paid one also `prices:`.
- Alias entry: either a list of steps, or `{steps: [...], fallback_to: "<mode>/<alias>"}`.
- Settings per mode: `cooldown_seconds` (free 60, paid 5), `num_retries` (free: keys−1,
  paid 2), `first_byte_timeout` (free 20 s, paid 8 s).
- Keys: free = `<PROVIDER>_API_KEYS` (unchanged); paid = `PAID_<PROVIDER>_API_KEYS`. They
  never mix.
- Model names: `paid/<alias>`, `free/<alias>`, bare `<alias>` = free. `provider/model`
  passthrough stays free-mode; `paid/provider/model` uses paid keys.
- One `Gateway` (LiteLLM Router, own cooldowns) per mode; a `Modes` dispatcher resolves the
  prefix and follows `fallback_to` (any mode) after a chain is exhausted; responses then carry
  `x-gateway-mode` of the mode that answered. Cycles are rejected at load.
- Realtime: `/v1/realtime?model=paid/realtime` (default free). Paid s2s adds an OpenAI Realtime
  backend (Pipecat `OpenAIRealtimeLLMService`) beside Gemini Live; paid cascade LLM is
  `paid/chat-fast`. `realtime.fallback_to` appends the other mode's options.

Default paid catalog (ids from LiteLLM 1.102.1's price table): see `providers.paid.yaml`.
chat-best: claude-opus-5-5 → gpt-5.5 → gemini-3.1-pro-preview (→ paid/chat-smart);
chat-smart: claude-sonnet-5 → gpt-5.6 → gemini-3.1-pro-preview (→ free/chat-smart);
chat-fast: claude-haiku-4-5 → gpt-5.4-mini → gemini-3.8-flash (→ free/chat-fast);
embed-openai-small/large (no fallback); stt: gpt-4o-transcribe → deepgram nova-3 →
elevenlabs scribe_v2 → gemini-3.5-transcribe (→ free/stt); tts: gpt-4o-mini-tts →
elevenlabs flash v2.5 → gemini-3.1-flash-tts-preview (→ free/tts); realtime s2s:
gpt-realtime-2.1 → gemini-3.1-flash-live-preview; cascade STT deepgram → assemblyai, TTS
cartesia → elevenlabs → deepgram (→ free/realtime).

## Pricing

`Pricer.price(provider, model, at) -> Price | None`, first hit wins:

1. **Provider-reported cost** on the response (`usage.cost`, e.g. OpenRouter) — used as the
   cost directly, source `provider`.
2. **Overrides**: `prices:` in the mode's yaml, keyed `provider/model`, each with
   `effective_from` (date) and unit fields; the latest entry not after `at` applies. Source
   `override`.
3. **Live LiteLLM map**: fetched from LiteLLM's GitHub `model_prices_and_context_window.json`
   at startup and every `PRICE_REFRESH_HOURS` (6); on failure the last good map is kept. Source
   `litellm-live@<fetched_at>`.
4. **Bundled map** of the pinned LiteLLM. Source `litellm-bundled`.

Unit fields: input, output, cache_read, cache_write (per token); audio_input, audio_output (per
token); per_second (input audio); per_character (TTS). A model with no price is recorded with
`cost = NULL` and listed in `/health.pricing.unpriced`. `/health.pricing` also shows map age
and source. Paid voice providers (Deepgram, Cartesia, ElevenLabs, AssemblyAI) are not in the
LiteLLM map; they stay unpriced until the owner adds overrides.

Token normalisation: counts are stored as disjoint buckets — `input` (uncached text),
`cache_read`, `cache_write`, `output` (incl. reasoning), `reasoning` (informational),
`audio_input`, `audio_output`, `audio_cache_read`. LiteLLM usage is gross
(`prompt_tokens` includes cache; verified for Anthropic in 1.102.1 transformation):
`input = prompt − cache_read − cache_write − audio_input`. Pipecat `LLMTokenUsage` is net for
Anthropic/Bedrock and gross otherwise (documented in pipecat metrics); normalise per service.

Cost = Σ bucket × unit price; each component stores unit prices used + source, so
`POST /v1/usage/reprice?from=&to=` recomputes from raw counts with current prices.

## Turns and usage

SQLite `state.db` (key health moves here too):

- `turns(turn_id, conversation_id, session_id, mode, endpoint, alias, started_at, ended_at,
  cost, cost_complete)`
- `components(turn_id, kind [chat|embed|stt|tts|s2s|llm], provider, model, attempt [ok|failed],
  input, cache_read, cache_write, output, reasoning, audio_input, audio_output,
  audio_cache_read, audio_seconds, characters, unit_prices JSON, price_source, cost, latency_ms)`

- HTTP endpoints: each request = one turn with one component (the step that answered; mode of
  the answering gateway). Header `x-gateway-turn-id`; optional request header
  `x-gateway-conversation-id` groups turns. Streams: the gateway sets
  `stream_options.include_usage=true` and records from the final chunk after the stream ends.
- Realtime s2s: a turn per response, one `s2s` component from Pipecat
  `LLMUsageMetricsData`.
- Realtime cascade: a turn from the caller's finished utterance to `response.done`;
  components `stt` (audio seconds from `STTUsageMetricsData` accumulated since the previous
  turn), `llm` (recorded by the chat endpoint: the loopback LLM sends
  `x-gateway-session: <session id>`, and the endpoint attaches its component to that session's
  current turn instead of opening a new one — so provider/model are the ones that really
  answered), `tts` (characters from `TTSUsageMetricsData`). Pipecat LLM usage metrics are
  ignored in the cascade to avoid double counting.
- Mid-turn backend failover: components already measured stay, marked `attempt=failed`.
- `response.done.usage` carries the turn's totals and components.
- Reads (gateway key required): `GET /v1/usage?from&to&group_by=model|provider|mode|alias|day|endpoint`,
  `GET /v1/usage/turns/{id}`, `GET /v1/usage/turns?conversation_id=&limit=`,
  `POST /v1/usage/reprice`.
- Never stored: prompts, outputs, keys.

## Testing

Offline (fake upstreams): prefix routing, bare = free, `fallback_to` across modes with
`x-gateway-mode`, paid/free key isolation, cycle rejection; usage rows for chat, streamed chat,
embeddings, STT, TTS; cached/reasoning/provider-cost fields from a fake upstream; Anthropic-style
gross→bucket normalisation; pricer layer order, override effective dates, live-map refresh
failure keeps last good map, unpriced → NULL; reprice; turn grouping; realtime s2s turn from
metrics and cascade turn combining stt + loopback llm + tts; `response.done.usage`.
Live: `make smoke --mode paid` prints tokens and cost per call.
