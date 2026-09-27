# Paid Mode and Usage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Paid mode beside free, live-updated pricing, and per-turn usage with component breakdown.
**Architecture:** One `Gateway` per mode behind a `Modes` dispatcher; a `Pricer` with layered price sources; a SQLite `UsageStore` of turns and components fed by the HTTP endpoints and realtime Pipecat metrics.
**Tech Stack:** as before (FastAPI, litellm 1.102.1, pipecat-ai 1.11.0, sqlite3, httpx).
**Spec:** `docs/superpowers/specs/2026-09-27-paid-mode-and-usage-design.md`

## Global Constraints
- Bare alias = free; free env key names unchanged; paid keys `PAID_<PROVIDER>_API_KEYS`.
- Prompts, outputs and keys are never stored.
- Unknown price → cost NULL, never a guess.
- All existing tests keep passing (free behaviour unchanged).
- No Claude/AI attribution in commits.

## Review Focus
1. Stream usage must be recorded even when the client disconnects mid-stream.
2. Cascade LLM usage must not be counted twice (loopback row + Pipecat metrics).
3. `fallback_to` must not loop between modes.
4. Price refresh failure must not break requests or wipe the last good map.
5. Token buckets must not double-count cached tokens (gross vs net sources).

### Task 1: Mode config
Files: `gateway/config.py`, `providers.free.yaml` (renamed), `tests/test_config.py`
Produces: `load_keys(env, prefix="")` (prefix "PAID_" strips it; free ignores PAID_ vars); `Catalog.settings: ModeSettings(cooldown_seconds, num_retries: int|None, first_byte_timeout)`; `Catalog.fallback_to: dict[str,str]`; `Catalog.prices: dict[str, list[dict]]`; `RealtimeChain.fallback_to: str|None`; `load_catalog` accepts list-or-dict alias entries; `check_fallbacks(catalogs: dict[str,Catalog])` raises on cycles/unknown targets.
- [ ] tests → RED → implement → GREEN → commit "Load per-mode catalogs, settings and paid keys"

### Task 2: Modes dispatcher
Files: `gateway/router.py`, `gateway/modes.py`, `gateway/app.py`, tests
Produces: `Gateway(catalog, keys, mode)` uses catalog.settings; `Modes(gateways: dict[str,Gateway])` with `split(model)->(mode, name)`, `async call(method, model, **kw) -> Answer(resp, mode, group, skipped)` following fallback_to on Exhausted; `aliases()` prefixed. App uses Modes; header `x-gateway-mode`.
- [ ] tests (prefix routing, bare=free, fallback_to with header, key isolation) → RED → GREEN → commit "Route requests to free or paid mode with cross-mode fallback"

### Task 3: Pricing
Files: `gateway/pricing.py`, tests
Produces: `Price(units: dict, source: str, guessed: bool)`; `Pricer(overrides_by_mode, fetch=callable, bundled=dict)`, `.price(provider, model, at, mode)`, `async refresh()`, `.status()`; `Counts` dataclass of buckets; `counts_from_litellm(usage)`, `counts_from_pipecat(token_usage, service)`; `cost(counts, price) -> float|None`.
- [ ] tests (layer order, effective dates, refresh failure keeps map, gross→buckets, anthropic-net pipecat, unpriced None) → RED → GREEN → commit "Price usage from overrides and a refreshed LiteLLM map"

### Task 4: Usage store + HTTP recording
Files: `gateway/usage.py`, `gateway/app.py`, tests
Produces: `UsageStore(path)`: `open_turn(...)->turn_id`, `add(turn_id, Component)`, `close_turn(turn_id)`, `totals(from,to,group_by)`, `turn(id)`, `turns(conversation_id, limit)`, `reprice(pricer, from, to)`; session registry `current_turn(session_id)`. App records every endpoint; stream include_usage; `x-gateway-turn-id`, `x-gateway-conversation-id`, `x-gateway-session` attach; `/v1/usage*` routes.
- [ ] tests → RED → GREEN → commit "Record usage per turn and serve usage reports"

### Task 5: Realtime modes and turns
Files: `gateway/realtime/backends.py`, `gateway/realtime/session.py`, `gateway/app.py`, tests
Produces: `openai_realtime_option`; `realtime_options(..., mode, session_id)`; EventSink turn accounting from MetricsFrames; cascade loopback header; `response.done.usage`; `/v1/realtime?model=paid/realtime`.
- [ ] tests → RED → GREEN → commit "Account realtime turns and add paid voice backends"

### Task 6: Paid catalog, smoke, docs
Files: `providers.paid.yaml`, `scripts/smoke.py`, `README.md`, `.env.example`
- [ ] test: shipped paid catalog loads, fallbacks acyclic → commit "Add the paid catalog and document both modes"

### Task 7: Final review (fresh reviewer, one fix pass)
