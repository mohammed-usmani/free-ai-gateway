# Free AI Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A self-hosted OpenAI-compatible gateway over no-card free tiers with key rotation and cross-provider fallback for chat, embeddings, STT, TTS and realtime voice.

**Architecture:** FastAPI app. Request/response endpoints go through a LiteLLM `Router` built from `providers.yaml` × CSV env keys. `WS /v1/realtime` speaks the OpenAI Realtime GA protocol and runs a Pipecat pipeline per backend (Gemini Live, else cascade), swapping backends mid-session on failure; realtime key health lives in SQLite.

**Tech Stack:** Python 3.12, uv, FastAPI, uvicorn, litellm==1.102.1, pipecat-ai==1.11.0, PyYAML, sqlite3, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-09-26-free-ai-gateway-design.md`

## Global Constraints

- No card-requiring provider in the default `providers.yaml`.
- Keys only from env (`<PROVIDER>_API_KEYS` CSV, or `<PROVIDER>_API_KEY`); never logged unmasked, never committed.
- Every endpoint requires `Authorization: Bearer $GATEWAY_API_KEY`.
- OpenAI wire formats (chat, embeddings, audio, Realtime GA events).
- Bad requests (4xx other than 401/403/404/408/429) are returned, not retried.
- Embedding aliases hold exactly one model.
- No Claude/AI attribution in commits.

## Review Focus

1. A provider whose keys are all cooling down must be skipped, not retried in a tight loop.
2. Mid-stream chat failure must not splice a second provider's output into the same stream.
3. Realtime backend swap must keep the client socket and re-apply `session.update` config.
4. A client closing the realtime socket must tear down the Pipecat task (no leaked provider sessions).
5. Keys must never appear in `/health`, logs or error bodies.

## File structure

```
gateway/
  config.py        keys from env, providers.yaml → Catalog
  keypool.py       SQLite key health (realtime)
  router.py        Catalog + keys → litellm.Router, alias → first model group
  app.py           FastAPI: auth, /v1/chat/completions, /v1/embeddings, /v1/audio/*, /v1/models, /health, /v1/realtime
  realtime/
    events.py      client-event parsing, server-event builders
    session.py     RealtimeSession: socket loop, backend lifecycle, failover
    backends.py    Pipecat pipelines: Gemini Live, cascade; EventSink processor
providers.yaml
tests/…
Dockerfile, docker-compose.yml, .env.example, README.md, scripts/smoke.py
```

---

### Task 1: Scaffold + config

**Files:** Create `pyproject.toml`, `gateway/__init__.py`, `gateway/config.py`, `providers.yaml`, `tests/test_config.py`

**Interfaces — Produces:**
- `load_keys(env: Mapping[str,str]) -> dict[str, list[str]]` — provider name lower-case; `FOO_API_KEYS` CSV (trimmed, empties dropped, de-duplicated in order) else `FOO_API_KEY`.
- `@dataclass Step(provider: str, model: str, options: dict)`; `@dataclass Catalog(aliases: dict[str, list[Step]], realtime: RealtimeChain)`; `RealtimeChain(s2s: list[Step], stt: list[Step], llm_alias: str, tts: list[Step])`.
- `load_catalog(path) -> Catalog`; `available(steps, keys) -> list[Step]` (steps whose provider has keys; providers listed in `KEYLESS` need none).

- [ ] Tests: CSV parsing (`" a, b ,,a"` → `["a","b"]`), single-key fallback, provider without keys skipped by `available`, catalog loads shipped `providers.yaml` and every embed alias has exactly one step.
- [ ] Run → FAIL (module missing). Implement. Run → PASS. Commit "Load provider keys and fallback chains from config".

### Task 2: KeyPool (SQLite)

**Files:** `gateway/keypool.py`, `tests/test_keypool.py`

**Produces:** `KeyPool(path: str, clock=time.time)`:
- `pick(provider, keys) -> str | None` — round-robin over keys not cooling down (rotation cursor per provider in memory).
- `report(provider, key, status: int | None, retry_after: float | None = None)` — 401/403 → 24 h; 429 → retry_after or 60 s; None/5xx/timeout → 15 s. Keys stored as sha256 prefix.
- `soonest(provider, keys) -> float | None` seconds until a key recovers.
- `states(provider, keys) -> list[dict]` masked (`sk-…abcd`), state, until.

- [ ] Tests: rotation alternates; 401 key skipped for 24 h then returns (fake clock); 429 honours retry_after; state shared by two `KeyPool` instances on one file; all cooling → `pick` None and `soonest` > 0; masked output never contains the full key.
- [ ] FAIL → implement → PASS → commit "Track key health in SQLite".

### Task 3: LiteLLM router

**Files:** `gateway/router.py`, `tests/test_router.py`

**Consumes:** `Catalog`, `load_keys`, `available`.
**Produces:** `Gateway(catalog, keys)` with `.router: litellm.Router`, `.resolve(model: str) -> str` (alias → first group name `alias#0`; `provider/model` passthrough if that provider has keys; unknown → `KeyError`), `.aliases() -> list[str]`.

Build rules: for each alias, each available step `i` becomes model group `f"{alias}#{i}"` with one deployment per key (`litellm_params: {model: f"{provider}/{model}", api_key, **options}`); `fallbacks=[{f"{alias}#0": [f"{alias}#1", …]}]`; `num_retries=len(keys)-1` per group via retry across deployments; `cooldown_time=60`; `allowed_fails=0`.

- [ ] Tests using `mock_response` in `litellm_params` (test builds catalog with fake steps): first step mocked `litellm.RateLimitError` → answer from step 2; 401 on key1 → key2 of same step answers; `mock_response` BadRequest → raised, not fallen back; resolve unknown alias → KeyError.
- [ ] FAIL → implement → PASS → commit "Route chat and embeddings through LiteLLM with fallbacks".

### Task 4: HTTP API

**Files:** `gateway/app.py`, `tests/test_app.py`

**Produces:** `create_app(gateway: Gateway, pool: KeyPool, gateway_key: str, realtime_factory=None) -> FastAPI`; module-level `app` built from env for uvicorn.

Endpoints: `POST /v1/chat/completions` (JSON or SSE when `stream`), `POST /v1/embeddings`, `POST /v1/audio/transcriptions` (multipart), `POST /v1/audio/speech` (binary), `GET /v1/models`, `GET /health` (no auth). Error map: litellm `BadRequestError`→400, `AuthenticationError` after fallbacks→503, `RateLimitError`→429/503 with `retry-after`, unknown model→404, others→502; body `{"error": {"message", "type"}}` with keys scrubbed. Headers `x-gateway-model` (deployment model from response `_hidden_params`).

- [ ] Tests (TestClient, mock steps): 401 without bearer; chat JSON shape; streamed chat yields `data:` chunks ending `[DONE]`; embeddings shape; 400 passthrough; all-steps-failing → 503 with retry-after; `/v1/models` lists aliases; `/health` masks keys.
- [ ] FAIL → implement → PASS → commit "Serve OpenAI-compatible chat, embeddings and audio endpoints".

### Task 5: Realtime session core (fake backends)

**Files:** `gateway/realtime/events.py`, `gateway/realtime/session.py`, `tests/test_realtime_session.py`

**Produces:**
- `class Backend(Protocol)`: `name: str`; `async start(config: dict, emit: Callable[[dict], Awaitable]) -> None`; `async send_audio(pcm16_24k: bytes)`; `async send_text(text)`; `async function_result(call_id, output)`; `async respond()`; `async cancel()`; `async stop()`; `failed: asyncio.Event`; `failure: tuple[str, str, int|None] | None` (provider, key, status).
- `BackendOption = Callable[[], Backend | None]` (returns None when no usable key).
- `RealtimeSession(ws, options: list[BackendOption], pool: KeyPool)`, `async run()`: send `session.created`; loop client events; start first option lazily at first event; on `failed` report key to pool, `stop()` it, start next option with stored config, emit `session.updated`; none left → `error` event + close 1011.

- [ ] Tests with a `FakeBackend`: session.update stored and passed to start; audio forwarded as bytes; backend A fails mid-session → B started with same config, socket still open, next `response.output_audio.delta` from B; option returning None is skipped; all exhausted → error event; client disconnect → current backend `stop()` called.
- [ ] FAIL → implement → PASS → commit "Run realtime sessions with mid-session backend failover".

### Task 6: Pipecat backends

**Files:** `gateway/realtime/backends.py`, `tests/test_backends.py`

**Produces:** `PipecatBackend(Backend)` wrapping a `PipelineTask` with no transport: input frames queued with `task.queue_frame(InputAudioRawFrame(audio, 24000, 1))`; an `EventSink` processor at the tail turns `TTSAudioRawFrame/OutputAudioRawFrame` → `response.output_audio.delta` (resampled to 24 kHz), `TTSTextFrame/LLMTextFrame` → transcript delta, `TranscriptionFrame` → input transcription, `UserStartedSpeakingFrame`/`InterruptionFrame` → `input_audio_buffer.speech_started`, `LLMFullResponseEnd` → `response.done`; fatal `ErrorFrame` → `failed`. Catch-all function handler forwards calls as `response.function_call_arguments.done` and awaits `function_result` (30 s).
Factories: `gemini_live_option(step, keys, pool)`, `cascade_option(chain, keys, pool, llm_base_url, gateway_key)` — STT/TTS lists wrapped in `ServiceSwitcher(..., ServiceSwitcherStrategyFailover)`, LLM `OpenAILLMService(base_url=llm_base_url, api_key=gateway_key, model=chain.llm_alias)`, Silero VAD in `LLMUserAggregatorParams`.

- [ ] Tests: `EventSink` maps each frame type to the right event (pipecat `run_test`); a cascade built with fake STT/LLM/TTS processors turns queued audio + a TranscriptionFrame into audio delta events; factories return None when provider has no usable key; tool call round-trip through the catch-all handler.
- [ ] FAIL → implement → PASS → commit "Build Gemini Live and cascade voice backends on Pipecat".

### Task 7: Wire, package, document

**Files:** `gateway/app.py` (realtime route), `Dockerfile`, `docker-compose.yml`, `.env.example`, `.gitignore`, `README.md`, `scripts/smoke.py`, `Makefile`

- [ ] Test: `/v1/realtime` rejects missing bearer (close 1008), accepts `?` subprotocol-less bearer header; end-to-end with fake options.
- [ ] Docker image builds; container `/health` returns 200 with no keys configured.
- [ ] Live check: OpenAI SDK against chat/embeddings using whatever keys the owner configures (`make smoke`), skipped providers reported.
- [ ] Commit "Package the gateway with Docker and document setup".

### Task 8: Final review

Fresh reviewer over the whole branch; fix Critical/Important with a failing test first.
