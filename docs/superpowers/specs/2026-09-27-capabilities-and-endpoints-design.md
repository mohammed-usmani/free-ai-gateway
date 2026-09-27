# Capability routing, quota-aware cooldowns and new endpoints — design

Date: 2026-09-27. Status: approved in conversation; spec self-reviewed. Builds on the free and
paid-mode specs. Decisions by the owner: Responses API stores items only when `store` is true
(OpenAI default) with retention; response caching is opt-in.

## Fixes
1. **Capability-aware chains.** Needs derived per request: `vision` (image content parts),
   `tools`, `json_schema` (response_format json_schema), `prompt_tokens` (litellm.token_counter).
   A step is skipped when LiteLLM model info (or a per-step `supports:` override) says it
   lacks a needed capability or its `max_input_tokens` is smaller than the prompt; unknown info
   → the step is tried. Context-window errors and 400s whose message says a feature is
   unsupported ("does not support", "not supported", "unsupported", image/tool/json phrases)
   fall back; other 400s return to the caller.
2. **Daily-limit cooldowns.** A LiteLLM failure callback classifies 429s: Gemini `quotaId`
   containing `PerDay` (reset midnight America/Los_Angeles); Groq body "per day (RPD|TPD)" or
   `x-ratelimit-reset-requests`; OpenRouter "free-models-per-day" or `X-RateLimit-Reset`
   (ms epoch); Cloudflare error code 4006 (reset 00:00 UTC); SambaNova
   `x-ratelimit-remaining-requests-day: 0` with `x-ratelimit-reset-requests-day`. Daily →
   the deployment is cooled down in the router until the reset and the cooldown is written to
   SQLite; on start, unexpired cooldowns are re-applied. Otherwise LiteLLM's normal cooldown.
3. **Streaming TTS.** `/v1/audio/speech` streams: OpenAI-compatible providers (openai, groq)
   via a streamed POST to `<base>/audio/speech`, ElevenLabs via its `/stream` endpoint; other
   providers (Gemini) go through LiteLLM and are sent whole. Fallback before the first chunk;
   keys rotate through the SQLite key pool (`<mode>:<provider>`).
4. **Single process.** An exclusive `flock` on `<state.db>.lock` at startup; a second process
   exits with a message naming the lock. Documented; Redis is the scale-out path.
5. **First-byte timeout.** Paid default 30 s; `first_byte_timeout` may be set per step.
6. **Cost basis.** Usage rows and totals carry `cost_basis`: `billed` (paid) or
   `list_price_estimate` (free).

## New features
- **`/v1/responses`**: `router.aresponses` through Modes/fallback/usage; SSE passthrough when
  `stream`. `store` (default true): the gateway persists input + output items (table
  `responses`), resolves `previous_response_id` itself (prepends stored items) and calls
  providers with `store=false`. `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}`. Rows
  older than `RESPONSES_RETENTION_DAYS` (30) are purged at start and daily. Usage from
  `response.usage` (input/output/cached/reasoning).
- **`/v1/messages`** (Anthropic): `router.aanthropic_messages`; Anthropic SSE when streaming;
  auth by `Authorization: Bearer` or `x-api-key`; yaml `model_map` (glob → alias, e.g.
  `claude-*: paid/chat-smart`) applied before routing.
- **Caching**: LiteLLM disk cache (`diskcache`), mode default-off; enabled per request with
  `x-gateway-cache: true` or per alias `cache: true`; `CACHE_DIR`, `CACHE_TTL` (24 h). Hits are
  recorded with `attempt=cached`, cost 0.
- **Privacy routing**: yaml `providers: {<name>: {trains_on_data: no|opt-out|yes}}` per mode;
  `x-gateway-private: true` (realtime `?private=true`) keeps only `no`. Nothing left → 503
  `no_private_provider`.
- **Streaming STT**: `/v1/realtime?intent=transcription` (or `session.type: "transcription"`):
  STT-only Pipecat pipeline with failover; emits
  `conversation.item.input_audio_transcription.delta/completed`; usage per utterance.
- **Rerank** `/v1/rerank` (Cohere/Jina shape): free jina → voyage → cohere trial → nvidia_nim;
  paid cohere v4 pro → voyage → jina. Usage bucket `searches` priced by
  `input_cost_per_query`; token-priced rerankers use input tokens.
- **Images** `/v1/images/generations`: free Cloudflare flux-1-schnell (custom LiteLLM
  provider `cloudflare_images`, `CLOUDFLARE_ACCOUNT_ID`) → Pollinations (`openai/` with
  api_base); paid gemini-3.1-flash-lite-image → gpt-image-2.5-flare → gemini-3.1-flash-image.
  Usage bucket `images` priced by `output_cost_per_image`; token-billed image models from usage.
- **Moderation** `/v1/moderations`: OpenAI `omni-moderation-latest` in both modes (only
  provider LiteLLM supports; free with an OpenAI key).
- **Metrics**: every provider attempt (via LiteLLM success/failure callbacks, plus steps
  skipped for capability/privacy/cooldown) → SQLite `attempts`. `GET /v1/metrics?window=24h`
  per provider/model: requests, success rate, p50/p95 latency, errors by class, fallback rate,
  cache hit rate; `GET /metrics` Prometheus text.
- **Pricing health** `GET /v1/pricing/health` (+ summary in `/health`, smoke): per catalog
  model: source, unpriced, identical-to-sibling (same provider, identical input/output/cache
  units, different model), live price differs from bundled, override older than 90 days.

## Testing
Offline with fake upstreams for each item above; real-client checks with the OpenAI Agents
SDK (`/v1/responses`), Anthropic SDK (`/v1/messages`), OpenAI SDK (images, moderation,
transcription socket).
