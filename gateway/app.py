"""OpenAI-compatible HTTP API over the gateway."""

import asyncio
import io
import logging
from contextlib import asynccontextmanager
import json
import os
import secrets
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

import httpx
import litellm
import openai
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from gateway.config import Step, check_fallbacks, known_providers, load_catalog, load_keys
from gateway.keypool import KeyPool, mask
from gateway.lock import SingleProcessLock
from gateway.realtime.backends import RealtimeContext, realtime_options_for
from gateway.realtime.session import RealtimeSession
from gateway.modes import Answer, Modes
from gateway.observe import Observer
from gateway.pricing import Counts, Pricer, counts_from_litellm
from gateway.responses import ResponseStore, as_items
from gateway.usage import UsageStore
from gateway.router import Exhausted, Gateway, NoCapableStep, NoPrivateProvider, UnknownModel, should_fall_back, status_of
from gateway.speech import MEDIA_TYPES, SpeechError, can_stream, open_stream


# Request fields LiteLLM would treat as overrides of the configured deployment: they could
# send our provider keys to another host, swap the key, or re-enable fallback on 400s.
_NOT_FROM_CLIENTS = {
    "api_base", "base_url", "api_key", "api_version", "custom_llm_provider", "fallbacks",
    "context_window_fallbacks", "content_policy_fallbacks", "num_retries", "max_retries", "timeout",
    "mock_response", "metadata", "extra_headers", "headers", "model_list", "user_config",
}


def _client_params(body: dict) -> dict:
    return {k: v for k, v in body.items() if k not in _NOT_FROM_CLIENTS and not k.startswith("litellm")}


async def _json_object(request: Request) -> dict:
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(400, "body must be JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be a JSON object")
    return body


def enable_cache(directory: str, ttl: float):
    """LiteLLM's disk cache in opt-in mode: only calls passing cache={"use-cache": True} use it."""
    from litellm.caching.caching import Cache, CacheMode

    litellm.cache = Cache(type="disk", disk_cache_dir=directory, mode=CacheMode.default_off, ttl=ttl)


def _cache_wanted(request, modes, model: str) -> bool:
    flag = (request.headers.get("x-gateway-cache") or "").lower()
    if flag in ("0", "false", "no"):
        return False
    if flag in ("1", "true", "yes"):
        return True
    mode, name = modes.split(model)
    return name in modes.gateways[mode].catalog.cache_aliases


def _private(request) -> bool:
    return (request.headers.get("x-gateway-private") or request.query_params.get("private") or "").lower() in ("1", "true", "yes")


def _health_summary(report: list[dict]) -> dict:
    return {"models": len(report), "unpriced": sum("unpriced" in r["flags"] for r in report),
            "suspicious": [f'{r["mode"]}:{r["provider"]}/{r["model"]}: {", ".join(r["flags"])}'
                           for r in report if r["flags"] and r["flags"] != ["unpriced"]]}


_ANTHROPIC_ERRORS = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
                     404: "not_found_error", 429: "rate_limit_error", 503: "overloaded_error"}


def _error(status: int, message: str, kind: str, headers: dict | None = None, request: Request | None = None) -> JSONResponse:
    if request is not None and request.url.path.startswith("/v1/messages"):  # Anthropic clients parse this shape
        body = {"type": "error", "error": {"type": _ANTHROPIC_ERRORS.get(status, "api_error"), "message": message}}
        return JSONResponse(body, status, headers)
    return JSONResponse({"error": {"message": message, "type": kind}}, status, headers)


log = logging.getLogger(__name__)

# Providers whose responses carry the real charge. LiteLLM also stamps its *own* estimate into
# usage.cost on streams, so for anyone else that field is not the provider's number.
PROVIDER_REPORTED_COST = {"openrouter"}


def reported_cost(provider: str, usage) -> float | None:
    value = getattr(usage, "cost", None) if provider in PROVIDER_REPORTED_COST else None
    if isinstance(value, dict):
        value = value.get("total_cost")
    return float(value) if isinstance(value, (int, float)) else None


async def stream_and_record(resp, on_done, scrub):
    """SSE out of a committed LiteLLM stream. `on_done(usage, text)` runs however the stream
    ends — finished, failed, or the client went away (then usage is None and the text so far
    is used to estimate). The upstream stream is closed so the provider stops generating."""
    final_usage, text = None, []
    try:
        async for chunk in resp:
            if getattr(chunk, "usage", None):
                final_usage = chunk.usage
            for choice in getattr(chunk, "choices", None) or []:
                if content := getattr(getattr(choice, "delta", None), "content", None):
                    text.append(content)
            yield f"data: {chunk.model_dump_json()}\n\n"
    except Exception as exc:  # noqa: BLE001 — committed stream: report, don't splice
        yield f"data: {json.dumps({'error': {'message': scrub(str(exc)), 'type': 'provider_error'}})}\n\n"
    finally:
        if aclose := getattr(resp, "aclose", None):
            try:
                await aclose()
            except Exception:  # noqa: BLE001
                pass
        on_done(final_usage, "".join(text))
    yield "data: [DONE]\n\n"


def _estimate(model: str, messages, text: str) -> Counts:
    try:
        prompt = litellm.token_counter(model=model, messages=messages or [])
        output = litellm.token_counter(model=model, text=text) if text else 0
    except Exception:  # noqa: BLE001 — an estimate is best effort
        return Counts()
    return Counts(input=prompt, output=output)


def _responses_counts(u) -> Counts:
    if u is None:
        return Counts()
    get = (lambda o, k: (o or {}).get(k) if isinstance(o, dict) else getattr(o, k, None))
    cached = get(get(u, "input_tokens_details"), "cached_tokens") or 0
    reasoning = get(get(u, "output_tokens_details"), "reasoning_tokens") or 0
    return Counts(input=max(0, (get(u, "input_tokens") or 0) - cached), cache_read=cached,
                  output=get(u, "output_tokens") or 0, reasoning=reasoning)


def _anthropic_counts(u) -> Counts:
    """Anthropic usage is net: input_tokens excludes cache reads and writes."""
    u = u or {}
    return Counts(input=int(u.get("input_tokens") or 0), cache_read=int(u.get("cache_read_input_tokens") or 0),
                  cache_write=int(u.get("cache_creation_input_tokens") or 0), output=int(u.get("output_tokens") or 0))


def _rerank_counts(resp) -> Counts:
    """Search-billed rerankers (Cohere) report search units; token-billed ones report tokens."""
    meta = getattr(resp, "meta", None) or (resp.get("meta") if isinstance(resp, dict) else None) or {}
    billed = meta.get("billed_units") or {}
    if billed.get("search_units"):
        return Counts(searches=int(billed["search_units"]))
    tokens = billed.get("total_tokens") or (meta.get("tokens") or {}).get("input_tokens")
    return Counts(input=int(tokens)) if tokens else Counts(searches=1)


def _image_counts(resp) -> Counts:
    u = getattr(resp, "usage", None)
    out = getattr(u, "output_tokens", None) if u is not None else None
    if out:  # token-priced image models (gpt-image)
        return Counts(input=int(getattr(u, "input_tokens", 0) or 0), image_output=int(out))
    return Counts(images=len(getattr(resp, "data", None) or []))


def _was_cached(resp) -> bool:
    return bool((getattr(resp, "_hidden_params", None) or {}).get("cache_hit"))


def _audio_seconds(resp, data: bytes) -> float | None:
    if getattr(resp, "duration", None):
        return float(resp.duration)
    try:
        with wave.open(io.BytesIO(data)) as w:
            return w.getnframes() / w.getframerate()
    except (wave.Error, EOFError):
        return None


def _stt_counts(resp, data: bytes) -> Counts:
    """Token-billed transcription reports usage; per-second models report seconds or are
    measured from the audio."""
    u = getattr(resp, "usage", None)
    if getattr(u, "seconds", None):
        return Counts(audio_seconds=float(u.seconds))
    tokens_in = getattr(u, "input_tokens", None) if u is not None else None
    if tokens_in:
        audio = getattr(getattr(u, "input_token_details", None), "audio_tokens", 0) or 0
        return Counts(input=tokens_in - audio, audio_input=audio, output=getattr(u, "output_tokens", 0) or 0)
    return Counts(audio_seconds=_audio_seconds(resp, data) or 0)


def create_app(modes: Modes, pool: KeyPool, gateway_key: str, realtime=None, usage: UsageStore | None = None,
               lifespan=None, responses: ResponseStore | None = None) -> FastAPI:
    app = FastAPI(title="Free AI Gateway", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    all_keys = modes.keys()
    usage = usage or UsageStore(":memory:", Pricer({}, bundled={}, fetch=None))
    app.state.usage = usage
    responses = responses or ResponseStore(":memory:")

    def scrub(text: str) -> str:
        for key in (k for k in all_keys if len(k) >= 8):  # real keys; short test strings would mangle text
            text = text.replace(key, mask(key))
        return text

    def require_key(request: Request):
        # Bearer for OpenAI-style clients, x-api-key for Anthropic-style ones.
        given = request.headers.get("authorization", "").removeprefix("Bearer ") or request.headers.get("x-api-key", "")
        if not secrets.compare_digest(given.encode(), gateway_key.encode()):
            raise HTTPException(401, "invalid gateway key")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return _error(exc.status_code, str(exc.detail), "invalid_request_error", request=request)

    @app.exception_handler(UnknownModel)
    async def unknown_model(request: Request, exc: UnknownModel):
        return _error(404, f"model {exc.args[0]!r} is not configured or has no keys", "model_not_found", request=request)

    @app.exception_handler(NoPrivateProvider)
    async def no_private(request: Request, exc: NoPrivateProvider):
        return _error(503, str(exc), "no_private_provider", request=request)

    @app.exception_handler(NoCapableStep)
    async def no_capable_step(request: Request, exc: NoCapableStep):
        return _error(400, str(exc), "unsupported_request", request=request)

    @app.exception_handler(Exhausted)
    async def exhausted(request: Request, exc: Exhausted):
        headers = {"retry-after": str(int(exc.retry_after))} if exc.retry_after else None
        return _error(503, scrub(f"every provider failed; last error: {exc}"), "all_providers_failed", headers, request)

    # Exception subclasses (not bare Exception) so Starlette answers instead of re-raising.
    @app.exception_handler(openai.APIError)
    async def provider_error(request: Request, exc: Exception):
        status = status_of(exc)
        return _error(status if isinstance(status, int) else 502, scrub(str(exc)), "provider_error", request=request)

    def headers(answer: Answer) -> dict:
        h = {"x-gateway-mode": answer.mode, "x-gateway-model": answer.group, "x-gateway-fallbacks": str(answer.skipped)}
        if _was_cached(answer.resp):
            h["x-gateway-cache"] = "hit"
        return h

    def cache_args(request: Request, model: str) -> dict:
        # The router passes caching=False unless told otherwise, so opted-in calls say so.
        if litellm.cache is not None and _cache_wanted(request, modes, model):
            return {"caching": True, "cache": {"use-cache": True}}
        return {}

    auth = [Depends(require_key)]

    def begin(request: Request, endpoint: str, model: str) -> tuple[str, bool, str | None]:
        """The turn this call belongs to. A realtime cascade's chat call carries its session's
        private token: it joins the session's current turn, or (between turns) a new turn of
        that session. Anything else opens its own turn. Returns (turn_id, closes_here, kind)."""
        token = request.headers.get("x-gateway-session") if endpoint == "chat" else None
        if token and (turn := usage.session_turn(token)):
            return turn, False, "llm"
        if token and (info := usage.session_of(token)):
            return usage.open_turn(info[1], "realtime", "realtime", session_id=info[0]), True, "llm"
        mode, alias = modes.split(model)
        return usage.open_turn(mode, endpoint, alias, request.headers.get("x-gateway-conversation-id")), True, None

    def record(turn: tuple[str, bool, str | None], kind: str, answer: Answer, counts: Counts, started: float,
               usage_obj=None, attempt: str = "ok", free: bool = False):
        """Never lets bookkeeping fail a request the provider already answered."""
        turn_id, is_new, kind_override = turn
        provider, _, name = answer.group.partition("/")
        try:
            usage.add(turn_id, kind_override or kind, answer.mode, provider, name, counts, attempt=attempt,
                      fallbacks=answer.skipped,
                      latency_ms=int((time.monotonic() - started) * 1000),
                      provider_cost=0.0 if free else reported_cost(provider, usage_obj))
            if is_new:
                usage.close_turn(turn_id)
        except Exception:  # noqa: BLE001
            log.exception("could not record usage for turn %s", turn_id)

    @app.post("/v1/chat/completions", dependencies=auth)
    async def chat(request: Request):
        body = _client_params(await _json_object(request))
        model = body.pop("model", "")
        if body.get("stream"):
            body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
        started = time.monotonic()
        answer = await modes.call("acompletion", model, private=_private(request), **body, **cache_args(request, model))
        resp = answer.resp
        turn = begin(request, "chat", model)
        if not body.get("stream"):
            record(turn, "chat", answer, counts_from_litellm(resp.usage), started, resp.usage,
                   attempt="cached" if _was_cached(resp) else "ok")
            return JSONResponse(resp.model_dump(), headers={**headers(answer), "x-gateway-turn-id": turn[0]})

        def on_done(final_usage, text: str):
            if final_usage:
                record(turn, "chat", answer, counts_from_litellm(final_usage), started, final_usage)
            else:  # interrupted: the provider still bills what it generated, so estimate it
                record(turn, "chat", answer, _estimate(answer.group, body.get("messages"), text), started,
                       attempt="interrupted")

        return StreamingResponse(stream_and_record(resp, on_done, scrub), media_type="text/event-stream",
                                 headers={**headers(answer), "x-gateway-turn-id": turn[0]})

    @app.post("/v1/embeddings", dependencies=auth)
    async def embeddings(request: Request):
        body = _client_params(await _json_object(request))
        model = body.pop("model", "")
        started = time.monotonic()
        answer = await modes.call("aembedding", model, private=_private(request), **body, **cache_args(request, model))
        turn = begin(request, "embed", model)
        u = answer.resp.usage
        record(turn, "embed", answer, Counts(input=int(getattr(u, "prompt_tokens", 0) or 0)), started,
               attempt="cached" if _was_cached(answer.resp) else "ok")
        return JSONResponse(answer.resp.model_dump(), headers={**headers(answer), "x-gateway-turn-id": turn[0]})

    @app.post("/v1/responses", dependencies=auth)
    async def create_response(request: Request):
        """OpenAI Responses API on any provider. With store (the default) the gateway keeps the
        conversation so previous_response_id works; providers are always called with store=false."""
        body = _client_params(await _json_object(request))
        model = body.pop("model", "")
        store = body.pop("store", True) is not False
        previous = body.pop("previous_response_id", None)
        history = []
        if previous:
            history = await asyncio.to_thread(responses.history, previous)
            if history is None:
                raise HTTPException(404, f"no stored response {previous!r} (it may have expired or used store=false)")
        new_items = as_items(body.pop("input", None))
        items = history + new_items
        started = time.monotonic()
        answer = await modes.call("aresponses", model, private=_private(request), input=items, store=False,
                                  **body, **cache_args(request, model))
        turn = begin(request, "responses", model)

        def finish(resp):
            record(turn, "responses", answer, _responses_counts(getattr(resp, "usage", None)), started)
            if store and resp is not None:
                data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
                responses.save(data["id"], items, data.get("output") or [], data)

        if not body.get("stream"):
            finish(answer.resp)
            return JSONResponse(answer.resp.model_dump(), headers={**headers(answer), "x-gateway-turn-id": turn[0]})

        async def events():
            final = None
            try:
                async for ev in answer.resp:
                    kind = str(getattr(getattr(ev, "type", None), "value", getattr(ev, "type", "")))
                    if kind == "response.completed":
                        final = getattr(ev, "response", None)
                    yield f"event: {kind}\ndata: {ev.model_dump_json()}\n\n"
            except Exception as exc:  # noqa: BLE001 — committed stream: report, don't splice
                yield f"event: error\ndata: {json.dumps({'type': 'error', 'message': scrub(str(exc))})}\n\n"
            finally:
                try:
                    finish(final)
                except Exception:  # noqa: BLE001
                    log.exception("could not store streamed response")
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={**headers(answer), "x-gateway-turn-id": turn[0]})

    @app.get("/v1/responses/{response_id}", dependencies=auth)
    async def get_response(response_id: str):
        stored = await asyncio.to_thread(responses.get, response_id)
        if not stored:
            raise HTTPException(404, "no such stored response")
        return stored["response"]

    @app.delete("/v1/responses/{response_id}", dependencies=auth)
    async def delete_response(response_id: str):
        if not await asyncio.to_thread(responses.delete, response_id):
            raise HTTPException(404, "no such stored response")
        return {"id": response_id, "object": "response", "deleted": True}

    @app.post("/v1/messages", dependencies=auth)
    async def messages(request: Request):
        """Anthropic Messages API (Claude Code, the Anthropic SDK) on any provider."""
        body = _client_params(await _json_object(request))
        model = modes.map_model(body.pop("model", ""))
        started = time.monotonic()
        answer = await modes.call("aanthropic_messages", model, private=_private(request), **body,
                                  **cache_args(request, model))
        turn = begin(request, "messages", model)
        if not body.get("stream"):
            data = answer.resp if isinstance(answer.resp, dict) else answer.resp.model_dump()
            data = {k: v for k, v in data.items() if not k.startswith("_")}
            record(turn, "messages", answer, _anthropic_counts(data.get("usage")), started)
            return JSONResponse(data, headers={**headers(answer), "x-gateway-turn-id": turn[0]})

        async def relay():
            seen = {}
            try:
                async for chunk in answer.resp:
                    raw = chunk if isinstance(chunk, (bytes, str)) else json.dumps(chunk)
                    text = raw.decode() if isinstance(raw, bytes) else raw
                    for line in text.splitlines():  # usage arrives on message_start / message_delta
                        if line.startswith("data: ") and '"usage"' in line:
                            try:
                                event = json.loads(line[6:])
                                usage = event.get("usage") or (event.get("message") or {}).get("usage") or {}
                                seen.update({k: v for k, v in usage.items() if v})
                            except ValueError:
                                pass
                    yield raw
            finally:
                record(turn, "messages", answer, _anthropic_counts(seen), started)

        return StreamingResponse(relay(), media_type="text/event-stream",
                                 headers={**headers(answer), "x-gateway-turn-id": turn[0]})

    @app.post("/v1/rerank", dependencies=auth)
    async def rerank(request: Request):
        body = _client_params(await _json_object(request))
        model = body.pop("model", "rerank")
        started = time.monotonic()
        answer = await modes.call("arerank", model, private=_private(request), **body)
        turn = begin(request, "rerank", model)
        record(turn, "rerank", answer, _rerank_counts(answer.resp), started)
        data = answer.resp.model_dump() if hasattr(answer.resp, "model_dump") else dict(answer.resp)
        return JSONResponse({k: v for k, v in data.items() if not k.startswith("_")},
                            headers={**headers(answer), "x-gateway-turn-id": turn[0]})

    @app.post("/v1/images/generations", dependencies=auth)
    async def images(request: Request):
        body = _client_params(await _json_object(request))
        model = body.pop("model", "image")
        started = time.monotonic()
        answer = await modes.call("aimage_generation", model, private=_private(request), **body)
        turn = begin(request, "image", model)
        record(turn, "image", answer, _image_counts(answer.resp), started)
        return JSONResponse(answer.resp.model_dump(), headers={**headers(answer), "x-gateway-turn-id": turn[0]})

    @app.post("/v1/moderations", dependencies=auth)
    async def moderations(request: Request):
        body = _client_params(await _json_object(request))
        model = body.pop("model", "moderation")
        started = time.monotonic()
        answer = await modes.call("amoderation", model, private=_private(request), **body)
        turn = begin(request, "moderation", model)
        record(turn, "moderation", answer, Counts(), started, free=True)  # OpenAI moderation is free
        return JSONResponse(answer.resp.model_dump(), headers={**headers(answer), "x-gateway-turn-id": turn[0]})

    @app.post("/v1/audio/transcriptions", dependencies=auth)
    async def transcriptions(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form("stt"),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
    ):
        data = await file.read()
        extra = {k: v for k, v in {"language": language, "prompt": prompt}.items() if v}
        started = time.monotonic()
        answer = await modes.call("atranscription", model, private=_private(request), file=(file.filename or "audio.wav", data), **extra)
        turn = begin(request, "stt", model)
        record(turn, "stt", answer, _stt_counts(answer.resp, data), started)
        return JSONResponse({"text": answer.resp.text}, headers={**headers(answer), "x-gateway-turn-id": turn[0]})

    @app.post("/v1/audio/speech", dependencies=auth)
    async def speech(request: Request):
        """Streams audio: tries each step until one starts sending, then relays chunks."""
        body = _client_params(await _json_object(request))
        model = body.pop("model", "tts")
        text, fmt = body.get("input", ""), body.get("response_format") or "mp3"
        started = time.monotonic()
        mode, name = modes.split(model)
        target: str | None = model
        skipped = 0
        last: Exception | None = None
        while target:
            mode, name = modes.split(target)
            gateway = modes.gateways[mode]
            try:
                groups = gateway.chain(name)
            except UnknownModel:
                groups = []
            for group in groups:
                step = gateway._steps.get(group) or Step(*group.split("/", 1))
                if _private(request) and not gateway.catalog.is_private(step.provider):
                    continue
                voice = step.options.get("voice") if modes.is_alias(target) else (body.get("voice") or step.options.get("voice"))
                try:
                    first, rest, close = await _start_speech(gateway, mode, group, step, text, voice, fmt, body)
                except HTTPException:
                    raise
                except Exception as exc:  # noqa: BLE001 — classified below
                    if not should_fall_back(exc):
                        raise HTTPException(400, scrub(str(exc)))
                    last, skipped = exc, skipped + 1
                    continue
                answer = Answer(None, mode, group, skipped)
                turn = begin(request, "tts", model)

                async def relay():
                    try:
                        yield first
                        async for chunk in rest:
                            yield chunk
                    finally:
                        await close()
                        record(turn, "tts", answer, Counts(characters=len(text)), started)

                return StreamingResponse(relay(), media_type=MEDIA_TYPES.get(fmt, "application/octet-stream"),
                                         headers={**headers(answer), "x-gateway-turn-id": turn[0]})
            target = gateway.catalog.fallback_to.get(name)
        if last is None:
            if _private(request):
                raise NoPrivateProvider(f"no text-to-speech provider for {model!r} is known not to train on API data")
            raise UnknownModel(model)
        raise Exhausted(last, modes.gateways[mode].settings.cooldown_seconds)

    async def _start_speech(gateway, mode, group, step, text, voice, fmt, body):
        """Returns (first chunk, remaining chunks, close) once the provider started sending."""
        if can_stream(step.provider):
            keys = gateway.keys.get(step.provider, [])
            key = pool.pick(f"{mode}:{step.provider}", keys)
            if not key:
                raise SpeechError(429, "every key is cooling down")
            http = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10))
            try:
                resp, chunks = await asyncio.wait_for(
                    open_stream(http, step.provider, step.model, key, text, voice, fmt, step.options.get("api_base")),
                    gateway.settings.first_byte_timeout)
                first = await chunks.__anext__()
            except SpeechError as exc:
                pool.report(f"{mode}:{step.provider}", key, exc.status_code)
                await http.aclose()
                raise
            except Exception:
                pool.report(f"{mode}:{step.provider}", key, None)
                await http.aclose()
                raise

            async def close():
                await resp.aclose()
                await http.aclose()

            return first, chunks, close
        resp = await gateway.router.aspeech(model=group, input=text, voice=voice,
                                            **{k: v for k, v in body.items() if k not in ("input", "voice")})

        async def nothing():
            return
            yield  # pragma: no cover

        async def done():
            return None

        return resp.content, nothing(), done

    @app.get("/v1/models", dependencies=auth)
    async def models():
        return {"object": "list", "data": [{"id": a, "object": "model", "owned_by": "gateway"} for a in modes.aliases()]}

    def _ts(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            pass
        try:
            when = datetime.fromisoformat(value)
        except ValueError:
            raise HTTPException(400, f"not a date or timestamp: {value!r}")
        return (when if when.tzinfo else when.replace(tzinfo=timezone.utc)).timestamp()

    # Reads and reprices can scan many rows: keep them off the event loop.
    @app.get("/v1/usage", dependencies=auth)
    async def usage_totals(group_by: str = "model", start: str | None = Query(None, alias="from"),
                           end: str | None = Query(None, alias="to")):
        try:
            rows = await asyncio.to_thread(usage.totals, _ts(start), _ts(end), group_by)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"object": "list", "group_by": group_by, "data": rows}

    @app.get("/v1/usage/turns", dependencies=auth)
    async def usage_turns(conversation_id: str | None = None, limit: int = 50):
        rows = await asyncio.to_thread(usage.turns, conversation_id, max(1, min(limit, 500)))
        return {"object": "list", "data": rows}

    @app.get("/v1/usage/turns/{turn_id}", dependencies=auth)
    async def usage_turn(turn_id: str):
        turn = await asyncio.to_thread(usage.turn, turn_id)
        if not turn:
            raise HTTPException(404, "no such turn")
        return turn

    @app.post("/v1/usage/reprice", dependencies=auth)
    async def usage_reprice(start: str | None = Query(None, alias="from"), end: str | None = Query(None, alias="to")):
        return {"repriced": await asyncio.to_thread(usage.reprice, _ts(start), _ts(end))}

    def catalog_models():
        seen = []
        for mode, g in modes.gateways.items():
            c = g.catalog
            for step in [s for steps in c.aliases.values() for s in steps] + c.realtime.s2s:
                item = (mode, step.provider, step.model.removeprefix("models/"))
                if item not in seen:
                    seen.append(item)
        return seen

    @app.get("/v1/pricing/health", dependencies=auth)
    async def pricing_health():
        return {"pricing": usage.pricer.status(),
                "models": usage.pricer.health(catalog_models(), datetime.now(timezone.utc))}

    @app.get("/v1/metrics", dependencies=auth)
    async def metrics(window: str = "24h"):
        unit = {"m": 60, "h": 3600, "d": 86400}.get(window[-1:], 1)
        try:
            seconds = float(window[:-1] if window[-1:] in "mhd" else window) * unit
        except ValueError:
            raise HTTPException(400, "window like 30m, 24h, 7d or seconds")
        return await asyncio.to_thread(usage.metrics, seconds)

    @app.get("/metrics", dependencies=auth, response_class=PlainTextResponse)
    async def prometheus():
        data = await asyncio.to_thread(usage.metrics, 86400)
        lines = ["# HELP gateway_attempts_total Provider attempts in the last 24h.",
                 "# TYPE gateway_attempts_total gauge"]
        for p in data["providers"]:
            labels = f'mode="{p["mode"]}",provider="{p["provider"]}",model="{p["model"]}"'
            ok = round((p["success_rate"] or 0) * p["requests"])
            lines.append(f'gateway_attempts_total{{{labels},outcome="ok"}} {ok}')
            lines.append(f'gateway_attempts_total{{{labels},outcome="error"}} {p["requests"] - ok}')
            lines.append(f'gateway_attempts_total{{{labels},outcome="skipped"}} {p["skipped"]}')
            for q in ("p50", "p95"):
                if p[f"{q}_ms"] is not None:
                    lines.append(f'gateway_latency_ms{{{labels},quantile="{q}"}} {p[f"{q}_ms"]}')
        for a in data["aliases"]:
            lines.append(f'gateway_fallback_rate{{mode="{a["mode"]}",alias="{a["alias"]}"}} {a["fallback_rate"]}')
        return "\n".join(lines) + "\n"

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "aliases": modes.aliases(),
            "pricing": {**usage.pricer.status(), "health": _health_summary(
                usage.pricer.health(catalog_models(), datetime.now(timezone.utc)))},
            "modes": {
                mode: {p: {"keys": len(k), "realtime": pool.states(f"{mode}:{p}", k)} for p, k in sorted(g.keys.items())}
                for mode, g in modes.gateways.items()
            },
        }

    if realtime is not None:
        @app.websocket("/v1/realtime")
        async def realtime_ws(ws: WebSocket):
            header = ws.headers.get("authorization", "")
            if not secrets.compare_digest(header.encode(), f"Bearer {gateway_key}".encode()):
                await ws.close(code=1008)
                return
            await realtime(ws)

    return app


def build() -> FastAPI:
    """uvicorn factory: `uvicorn gateway.app:build --factory`."""
    load_dotenv()
    gateway_key = os.environ["GATEWAY_API_KEY"]
    root = Path(__file__).parent.parent
    files = {"free": os.environ.get("PROVIDERS_FILE", root / "providers.free.yaml"),
             "paid": os.environ.get("PAID_PROVIDERS_FILE", root / "providers.paid.yaml")}
    catalogs = {m: load_catalog(f) for m, f in files.items() if Path(f).exists()}
    check_fallbacks(catalogs)
    mode_keys = {m: known_providers(load_keys(os.environ, prefix="" if m == "free" else "PAID_"), catalogs.values())
                 for m in catalogs}
    state_db = os.environ.get("STATE_DB") or os.environ.get("KEYPOOL_DB", "state.db")
    process_lock = SingleProcessLock(state_db)  # refuses a second worker/instance on this state
    pool = KeyPool(state_db)
    pricer = Pricer({m: c.prices for m, c in catalogs.items()})
    usage = UsageStore(state_db, pricer)
    observer = Observer(pool, sink=usage.log_attempt)
    modes = Modes({m: Gateway(c, mode_keys[m], mode=m, observer=observer) for m, c in catalogs.items()})
    # The cascade's LLM step calls this gateway's own chat endpoint, so it gets fallback too.
    self_url = os.environ.get("GATEWAY_SELF_URL", f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/v1")
    log.info("voice cascade reaches the chat endpoint at %s (set PORT or GATEWAY_SELF_URL if that's wrong)", self_url)

    async def realtime(ws: WebSocket):
        session_id = f"sess_{uuid.uuid4().hex[:20]}"
        token = secrets.token_urlsafe(24)
        model = ws.query_params.get("model", "realtime")
        usage.register_session(token, session_id, modes.split(model)[0])
        options = realtime_options_for(model, catalogs, mode_keys, pool, self_url, gateway_key,
                                       RealtimeContext(usage, session_id, token), private=_private(ws))
        try:
            await RealtimeSession(ws, options, pool, session_id=session_id).run()
        finally:
            usage.unregister_session(token)

    enable_cache(os.environ.get("CACHE_DIR", ".cache/responses"), float(os.environ.get("CACHE_TTL", 86400)))
    refresh_hours = float(os.environ.get("PRICE_REFRESH_HOURS", "6"))

    @asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(pricer.refresh_forever(refresh_hours)) if refresh_hours > 0 else None
        yield
        if task:
            task.cancel()
        process_lock.release()

    responses = ResponseStore(state_db, retention_days=float(os.environ.get("RESPONSES_RETENTION_DAYS", 30)))
    return create_app(modes, pool, gateway_key, realtime, usage, lifespan, responses)
