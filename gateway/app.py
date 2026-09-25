"""OpenAI-compatible HTTP API over the gateway."""

import json
import os
import secrets
from pathlib import Path

import openai
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse, Response, StreamingResponse

from gateway.config import load_catalog, load_keys
from gateway.keypool import KeyPool, mask
from gateway.realtime.backends import realtime_options
from gateway.realtime.session import RealtimeSession
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


def create_app(gateway: Gateway, pool: KeyPool, gateway_key: str, realtime=None) -> FastAPI:
    app = FastAPI(title="Free AI Gateway")
    all_keys = [k for keys in gateway.keys.values() for k in keys]

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

    def headers(group: str, skipped: int) -> dict:
        return {"x-gateway-model": group, "x-gateway-fallbacks": str(skipped)}

    auth = [Depends(require_key)]

    @app.post("/v1/chat/completions", dependencies=auth)
    async def chat(request: Request):
        body = _client_params(await request.json())
        model = body.pop("model", "")
        resp, group, skipped = await gateway.call("acompletion", model, **body)
        if not body.get("stream"):
            return JSONResponse(resp.model_dump(), headers=headers(group, skipped))

        async def sse():
            try:
                async for chunk in resp:
                    yield f"data: {chunk.model_dump_json()}\n\n"
            except Exception as exc:  # noqa: BLE001 — committed stream: report, don't splice
                yield f"data: {json.dumps({'error': {'message': scrub(str(exc)), 'type': 'provider_error'}})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream", headers=headers(group, skipped))

    @app.post("/v1/embeddings", dependencies=auth)
    async def embeddings(request: Request):
        body = _client_params(await request.json())
        model = body.pop("model", "")
        resp, group, skipped = await gateway.call("aembedding", model, **body)
        return JSONResponse(resp.model_dump(), headers=headers(group, skipped))

    @app.post("/v1/audio/transcriptions", dependencies=auth)
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form("stt"),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
    ):
        audio = (file.filename or "audio.wav", await file.read())
        extra = {k: v for k, v in {"language": language, "prompt": prompt}.items() if v}
        resp, group, skipped = await gateway.call("atranscription", model, file=audio, **extra)
        return JSONResponse({"text": resp.text}, headers=headers(group, skipped))

    @app.post("/v1/audio/speech", dependencies=auth)
    async def speech(request: Request):
        body = _client_params(await request.json())
        model = body.pop("model", "tts")
        # Voices are per provider: an alias uses each step's configured voice.
        if model in gateway.aliases():
            body.pop("voice", None)
        resp, group, skipped = await gateway.call("aspeech", model, **body)
        return Response(resp.content, media_type="audio/mpeg" if body.get("response_format", "mp3") == "mp3" else "application/octet-stream",
                        headers=headers(group, skipped))

    @app.get("/v1/models", dependencies=auth)
    async def models():
        return {"object": "list", "data": [{"id": a, "object": "model", "owned_by": "gateway"} for a in gateway.aliases()]}

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "aliases": gateway.aliases(),
            "providers": {
                p: {"keys": len(k), "realtime": pool.states(p, k)} for p, k in sorted(gateway.keys.items())
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
    keys = load_keys(os.environ)
    catalog = load_catalog(os.environ.get("PROVIDERS_FILE", Path(__file__).parent.parent / "providers.yaml"))
    gateway = Gateway(catalog, keys)
    pool = KeyPool(os.environ.get("KEYPOOL_DB", "keypool.db"))
    # The cascade's LLM step calls this gateway's own chat endpoint, so it gets fallback too.
    self_url = os.environ.get("GATEWAY_SELF_URL", f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/v1")
    options = realtime_options(catalog.realtime, keys, pool, self_url, gateway_key)

    async def realtime(ws: WebSocket):
        await RealtimeSession(ws, options, pool).run()

    return create_app(gateway, pool, gateway_key, realtime)
