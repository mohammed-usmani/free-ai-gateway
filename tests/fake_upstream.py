"""A fake OpenAI-compatible provider. The API key decides what it does:
rl → 429, auth → 401, bad → 400, down → 500, anything else → 200 naming model and key."""

import json
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

ERRORS = {"day": 429, "rl": 429, "auth": 401, "bad": 400, "down": 500, "noimg": 400, "ctx": 400}
MESSAGES = {"day": "Quota exceeded for metric: generate_content_free_tier_requests, quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier","noimg": "This model does not support image input.",
            "ctx": "This model's maximum context length is 8192 tokens. However, your messages resulted in 9000 tokens."}

app = FastAPI()
calls: list[tuple[str, str]] = []


def _fail(request: Request):
    key = request.headers.get("authorization", "").removeprefix("Bearer ")
    calls.append((request.url.path, key))
    status = ERRORS.get(key)
    if status:
        headers = {"retry-after": "1"} if status == 429 else {}
        code = "context_length_exceeded" if key == "ctx" else None
        body = {"error": {"message": MESSAGES.get(key, f"fake {status}"), "type": "invalid_request_error" if status == 400 else "x", "code": code}}
        return key, JSONResponse(body, status, headers)
    return key, None


@app.post("/v1/chat/completions")
async def chat(request: Request):
    key, err = _fail(request)
    if err:
        return err
    body = await request.json()
    text = f"{body['model']}:{key}"
    if body.get("stream"):
        def gen():
            for part in (text[: len(text) // 2], text[len(text) // 2 :]):
                chunk = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
                         "choices": [{"index": 0, "delta": {"role": "assistant", "content": part}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
            if (body.get("stream_options") or {}).get("include_usage"):
                final = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": body["model"], "choices": [],
                         "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}}
                yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return {"id": "c", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, "service_tier": "on_demand"}


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    key, err = _fail(request)
    if err:
        return err
    body = await request.json()
    inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
    return {"object": "list", "model": body["model"], "usage": {"prompt_tokens": 1, "total_tokens": 1},
            "data": [{"object": "embedding", "index": i, "embedding": [0.1, 0.2, 0.3]} for i, _ in enumerate(inputs)]}


@app.post("/v1/audio/transcriptions")
async def transcriptions(request: Request):
    key, err = _fail(request)
    if err:
        return err
    form = await request.form()
    return {"text": f"{form['model']}:{key}"}


@app.post("/v1/audio/speech")
async def speech(request: Request):
    key, err = _fail(request)
    if err:
        return err
    body = await request.json()

    def chunks():
        yield b"RIFF"
        yield f"fake-audio:{body['model']}:{key}".encode()

    return StreamingResponse(chunks(), media_type="audio/wav")


@app.post("/v1/text-to-speech/{voice}/stream")
async def elevenlabs_stream(voice: str, request: Request):
    key = request.headers.get("xi-api-key", "")
    calls.append((request.url.path, key))
    if key in ERRORS:
        return JSONResponse({"detail": {"status": "error"}}, ERRORS[key])
    body = await request.json()
    return StreamingResponse(iter([b"ID3", f"eleven:{voice}:{body['model_id']}".encode()]), media_type="audio/mpeg")


def start() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.01)
    return f"http://127.0.0.1:{port}/v1"
