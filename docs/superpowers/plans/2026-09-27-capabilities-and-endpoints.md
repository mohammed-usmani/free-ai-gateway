# Capabilities and Endpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the four design problems and two smaller ones, and add Responses, Messages, caching, privacy routing, streaming STT, rerank, images, moderation, metrics and pricing health.
**Architecture:** Everything routes through `Modes`/`Gateway.call`, now filtered by request needs and privacy; LiteLLM callbacks feed quota cooldowns and an attempts log; new endpoints are thin FastAPI routes over router methods.
**Tech Stack:** as before + diskcache.
**Spec:** `docs/superpowers/specs/2026-09-27-capabilities-and-endpoints-design.md`

## Global Constraints
- Existing behaviour and tests keep passing; bare alias = free.
- Prompts/outputs stored only by `/v1/responses` with store=true; keys never stored.
- Unknown capability → try the step; unknown price → NULL.
- No Claude/AI attribution in commits.

## Review Focus
1. A request needing vision must never be answered by a text-only model silently (skip or error, not a wrong answer).
2. Daily cooldowns must not bench keys for per-minute 429s.
3. Stored responses must honour store=false, DELETE and retention.
4. Private requests must never reach a provider marked yes/opt-out/unknown.
5. Callbacks must not break requests when SQLite or classification fails.

### Task 1: Capability-aware chains — `gateway/capabilities.py`, router, tests
### Task 2: Quota classifier + persisted cooldowns + attempts log — `gateway/quota.py`, `gateway/observe.py`, tests
### Task 3: Metrics endpoints + pricing health — app, `gateway/pricing.py`, tests
### Task 4: Single-process lock, first-byte per step, cost_basis — app/build, usage, tests
### Task 5: Streaming TTS — `gateway/speech.py`, app, tests
### Task 6: Privacy routing — config/modes/router/realtime, tests
### Task 7: Caching — router/app, tests
### Task 8: /v1/responses with storage — `gateway/responses.py`, app, tests
### Task 9: /v1/messages + model_map + x-api-key — app, config, tests
### Task 10: Rerank, images (cloudflare custom provider), moderation — app, `gateway/providers_extra.py`, tests
### Task 11: Streaming STT sessions — realtime, tests
### Task 12: Catalogs, docs, smoke, real-client checks
### Task 13: Final review
Each task: failing tests first, implement, suite green, commit, ledger line.
