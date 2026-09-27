"""OpenAI-compatible HTTP API over the gateway."""

import asyncio
import io
from contextlib import asynccontextmanager
import json
import os
import secrets
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

import openai
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse, Response, StreamingResponse

from gateway.config import check_fallbacks, load_catalog, load_keys
from gateway.keypool import KeyPool, mask
from gateway.realtime.backends import RealtimeContext, realtime_options_for
from gateway.realtime.session import RealtimeSession
from gateway.modes import Answer, Modes
from gateway.pricing import Counts, Pricer, counts_from_litellm
from gateway.usage import UsageStore
from gateway.router import Exhausted, Gateway, status_of


# Request fields LiteLLM would treat as overrides of the configured deployment: they could
# send our provider keys to another host, swap the key, or re-enable fallback on 400s.
_NOT_FROM_CLIENTS = {
    "api_base", "base_url", "api_key", "api_version", "custom_llm_provider", "fallbacks",
    "context_window_fallbacks", "content_policy_fallbacks", "num_retries", "max_retries", "timeout",
    "mock_response", "metadata", "extra_headers", "headers", "model_list", "user_config",
}


def _client_params(body: dict) -> dict:
    return {k: v for k, v in body.items() if k not in _NOT_FROM_CLIENTS and not k.startswith("litellm")}


def _error(status: int, message: str, kind: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status, headers)


def _audio_seconds(resp, data: bytes) -> float | None:
    if getattr(resp, "duration", None):
        return float(resp.duration)
    try:
        with wave.open(io.BytesIO(data)) as w:
            return w.getnframes() / w.getframerate()
    except (wave.Error, EOFError):
        return None


def _stt_counts(resp, data: bytes) -> Counts:
    """Token-billed transcription reports usage; per-second models are measured from the audio."""
    u = getattr(resp, "usage", None)
    tokens_in = getattr(u, "input_tokens", None) if u is not None else None
    if tokens_in:
        audio = getattr(getattr(u, "input_token_details", None), "audio_tokens", 0) or 0
        return Counts(input=tokens_in - audio, audio_input=audio, output=getattr(u, "output_tokens", 0) or 0)
    return Counts(audio_seconds=_audio_seconds(resp, data) or 0)


def create_app(modes: Modes, pool: KeyPool, gateway_key: str, realtime=None, usage: UsageStore | None = None,
               lifespan=None) -> FastAPI:
    app = FastAPI(title="Free AI Gateway", lifespan=lifespan)
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

    @app.exception_handler(KeyError)
    async def unknown_model(_: Request, exc: KeyError):
        return _error(404, f"model {exc.args[0]!r} is not configured or has no keys", "model_not_found")

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

    def begin(request: Request, endpoint: str, model: str) -> tuple[str, bool]:
        """The turn this call belongs to: the realtime session's current turn when the call
        comes from a cascade (x-gateway-session), else a new turn. Returns (turn_id, is_new)."""
        session = request.headers.get("x-gateway-session")
        if session and (turn := usage.session_turn(session)):
            return turn, False
        mode, alias = modes.split(model)
        return usage.open_turn(mode, endpoint, alias, request.headers.get("x-gateway-conversation-id")), True

    def record(turn: tuple[str, bool], kind: str, answer: Answer, counts: Counts, started: float, provider_cost=None):
        turn_id, is_new = turn
        provider, _, name = answer.group.partition("/")
        usage.add(turn_id, kind if is_new else "llm", answer.mode, provider, name, counts,
                  latency_ms=int((time.monotonic() - started) * 1000), provider_cost=provider_cost)
        if is_new:
            usage.close_turn(turn_id)

    @app.post("/v1/chat/completions", dependencies=auth)
    async def chat(request: Request):
        body = _client_params(await request.json())
        model = body.pop("model", "")
        if body.get("stream"):
            body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
        started = time.monotonic()
        answer = await modes.call("acompletion", model, **body)
        resp = answer.resp
        turn = begin(request, "chat", model)
        if not body.get("stream"):
            record(turn, "chat", answer, counts_from_litellm(resp.usage), started, getattr(resp.usage, "cost", None))
            return JSONResponse(resp.model_dump(), headers={**headers(answer), "x-gateway-turn-id": turn[0]})

        async def sse():
            final_usage = None
            try:
                async for chunk in resp:
                    if getattr(chunk, "usage", None):
                        final_usage = chunk.usage
                    yield f"data: {chunk.model_dump_json()}\n\n"
            except Exception as exc:  # noqa: BLE001 — committed stream: report, don't splice
                yield f"data: {json.dumps({'error': {'message': scrub(str(exc)), 'type': 'provider_error'}})}\n\n"
            finally:  # runs on client disconnect too: what was generated is billed
                counts = counts_from_litellm(final_usage) if final_usage else Counts()
                record(turn, "chat", answer, counts, started, getattr(final_usage, "cost", None))
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={**headers(answer), "x-gateway-turn-id": turn[0]})

    @app.post("/v1/embeddings", dependencies=auth)
    async def embeddings(request: Request):
        body = _client_params(await request.json())
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
        body = _client_params(await request.json())
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
            return datetime.fromisoformat(value).replace(tzinfo=datetime.fromisoformat(value).tzinfo or timezone.utc).timestamp()

    @app.get("/v1/usage", dependencies=auth)
    async def usage_totals(group_by: str = "model", start: str | None = Query(None, alias="from"),
                           end: str | None = Query(None, alias="to")):
        try:
            return {"object": "list", "group_by": group_by, "data": usage.totals(_ts(start), _ts(end), group_by)}
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/v1/usage/turns", dependencies=auth)
    async def usage_turns(conversation_id: str | None = None, limit: int = 50):
        return {"object": "list", "data": usage.turns(conversation_id, min(limit, 500))}

    @app.get("/v1/usage/turns/{turn_id}", dependencies=auth)
    async def usage_turn(turn_id: str):
        turn = usage.turn(turn_id)
        if not turn:
            raise HTTPException(404, "no such turn")
        return turn

    @app.post("/v1/usage/reprice", dependencies=auth)
    async def usage_reprice(start: str | None = Query(None, alias="from"), end: str | None = Query(None, alias="to")):
        return {"repriced": usage.reprice(_ts(start), _ts(end))}

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "aliases": modes.aliases(),
            "pricing": usage.pricer.status(),
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
    mode_keys = {m: load_keys(os.environ, prefix="" if m == "free" else "PAID_") for m in catalogs}
    modes = Modes({m: Gateway(c, mode_keys[m]) for m, c in catalogs.items()})
    state_db = os.environ.get("STATE_DB") or os.environ.get("KEYPOOL_DB", "state.db")
    pool = KeyPool(state_db)
    pricer = Pricer({m: c.prices for m, c in catalogs.items()})
    usage = UsageStore(state_db, pricer)
    # The cascade's LLM step calls this gateway's own chat endpoint, so it gets fallback too.
    self_url = os.environ.get("GATEWAY_SELF_URL", f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/v1")

    async def realtime(ws: WebSocket):
        session_id = f"sess_{uuid.uuid4().hex[:20]}"
        options = realtime_options_for(ws.query_params.get("model", "realtime"), catalogs, mode_keys, pool,
                                       self_url, gateway_key, RealtimeContext(usage, session_id))
        try:
            await RealtimeSession(ws, options, pool, session_id=session_id).run()
        finally:
            usage.set_session_turn(session_id, None)

    refresh_hours = float(os.environ.get("PRICE_REFRESH_HOURS", "6"))

    @asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(pricer.refresh_forever(refresh_hours)) if refresh_hours > 0 else None
        yield
        if task:
            task.cancel()

    return create_app(modes, pool, gateway_key, realtime, usage, lifespan)
