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

import litellm
import openai
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from gateway.config import check_fallbacks, known_providers, load_catalog, load_keys
from gateway.keypool import KeyPool, mask
from gateway.realtime.backends import RealtimeContext, realtime_options_for
from gateway.realtime.session import RealtimeSession
from gateway.modes import Answer, Modes
from gateway.observe import Observer
from gateway.pricing import Counts, Pricer, counts_from_litellm
from gateway.usage import UsageStore
from gateway.router import Exhausted, Gateway, NoCapableStep, UnknownModel, status_of


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


def _health_summary(report: list[dict]) -> dict:
    return {"models": len(report), "unpriced": sum("unpriced" in r["flags"] for r in report),
            "suspicious": [f'{r["mode"]}:{r["provider"]}/{r["model"]}: {", ".join(r["flags"])}'
                           for r in report if r["flags"] and r["flags"] != ["unpriced"]]}


def _error(status: int, message: str, kind: str, headers: dict | None = None) -> JSONResponse:
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
               lifespan=None) -> FastAPI:
    app = FastAPI(title="Free AI Gateway", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    all_keys = modes.keys()
    usage = usage or UsageStore(":memory:", Pricer({}, bundled={}, fetch=None))
    app.state.usage = usage

    def scrub(text: str) -> str:
        for key in all_keys:
            text = text.replace(key, mask(key))
        return text

    def require_key(request: Request):
        header = request.headers.get("authorization", "")
        if not secrets.compare_digest(header.encode(), f"Bearer {gateway_key}".encode()):
            raise HTTPException(401, "invalid gateway key")

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException):
        return _error(exc.status_code, str(exc.detail), "invalid_request_error")

    @app.exception_handler(UnknownModel)
    async def unknown_model(_: Request, exc: UnknownModel):
        return _error(404, f"model {exc.args[0]!r} is not configured or has no keys", "model_not_found")

    @app.exception_handler(NoCapableStep)
    async def no_capable_step(_: Request, exc: NoCapableStep):
        return _error(400, str(exc), "unsupported_request")

    @app.exception_handler(Exhausted)
    async def exhausted(_: Request, exc: Exhausted):
        headers = {"retry-after": str(int(exc.retry_after))} if exc.retry_after else None
        return _error(503, scrub(f"every provider failed; last error: {exc}"), "all_providers_failed", headers)

    # Exception subclasses (not bare Exception) so Starlette answers instead of re-raising.
    @app.exception_handler(openai.APIError)
    async def provider_error(_: Request, exc: Exception):
        status = status_of(exc)
        return _error(status if isinstance(status, int) else 502, scrub(str(exc)), "provider_error")

    def headers(answer: Answer) -> dict:
        return {"x-gateway-mode": answer.mode, "x-gateway-model": answer.group,
                "x-gateway-fallbacks": str(answer.skipped)}

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
               usage_obj=None, attempt: str = "ok"):
        """Never lets bookkeeping fail a request the provider already answered."""
        turn_id, is_new, kind_override = turn
        provider, _, name = answer.group.partition("/")
        try:
            usage.add(turn_id, kind_override or kind, answer.mode, provider, name, counts, attempt=attempt,
                      fallbacks=answer.skipped,
                      latency_ms=int((time.monotonic() - started) * 1000),
                      provider_cost=reported_cost(provider, usage_obj))
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
        answer = await modes.call("acompletion", model, **body)
        resp = answer.resp
        turn = begin(request, "chat", model)
        if not body.get("stream"):
            record(turn, "chat", answer, counts_from_litellm(resp.usage), started, resp.usage)
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
        answer = await modes.call("aembedding", model, **body)
        turn = begin(request, "embed", model)
        u = answer.resp.usage
        record(turn, "embed", answer, Counts(input=int(getattr(u, "prompt_tokens", 0) or 0)), started)
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
        answer = await modes.call("atranscription", model, file=(file.filename or "audio.wav", data), **extra)
        turn = begin(request, "stt", model)
        record(turn, "stt", answer, _stt_counts(answer.resp, data), started)
        return JSONResponse({"text": answer.resp.text}, headers={**headers(answer), "x-gateway-turn-id": turn[0]})

    @app.post("/v1/audio/speech", dependencies=auth)
    async def speech(request: Request):
        body = _client_params(await _json_object(request))
        model = body.pop("model", "tts")
        # Voices are per provider: an alias uses each step's configured voice.
        if modes.is_alias(model):
            body.pop("voice", None)
        started = time.monotonic()
        answer = await modes.call("aspeech", model, **body)
        turn = begin(request, "tts", model)
        record(turn, "tts", answer, Counts(characters=len(body.get("input", ""))), started)
        return Response(answer.resp.content, media_type="audio/mpeg" if body.get("response_format", "mp3") == "mp3" else "application/octet-stream",
                        headers={**headers(answer), "x-gateway-turn-id": turn[0]})

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
                                       RealtimeContext(usage, session_id, token))
        try:
            await RealtimeSession(ws, options, pool, session_id=session_id).run()
        finally:
            usage.unregister_session(token)

    refresh_hours = float(os.environ.get("PRICE_REFRESH_HOURS", "6"))

    @asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(pricer.refresh_forever(refresh_hours)) if refresh_hours > 0 else None
        yield
        if task:
            task.cancel()

    return create_app(modes, pool, gateway_key, realtime, usage, lifespan)
