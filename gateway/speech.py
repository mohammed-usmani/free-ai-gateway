"""Streaming text-to-speech. LiteLLM reads a whole clip before returning, so providers that can
stream are called directly: OpenAI-compatible `/audio/speech` (OpenAI, Groq) and ElevenLabs'
`/stream` endpoint. Others go through LiteLLM and arrive as one chunk. Fallback happens before
the first chunk; keys rotate through the key pool as `<mode>:<provider>`."""

from collections.abc import AsyncIterator

import httpx

OPENAI_COMPATIBLE = {"openai": "https://api.openai.com/v1", "groq": "https://api.groq.com/openai/v1"}
ELEVENLABS = "https://api.elevenlabs.io"
MEDIA_TYPES = {"mp3": "audio/mpeg", "wav": "audio/wav", "opus": "audio/ogg", "aac": "audio/aac",
               "flac": "audio/flac", "pcm": "audio/pcm"}
_ELEVEN_FORMATS = {"mp3": "mp3_44100_128", "pcm": "pcm_24000", "opus": "opus_48000_64"}


class SpeechError(Exception):
    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status_code = status


def can_stream(provider: str) -> bool:
    return provider in OPENAI_COMPATIBLE or provider == "elevenlabs"


async def open_stream(client: httpx.AsyncClient, provider: str, model: str, key: str, text: str,
                      voice: str, fmt: str, api_base: str | None) -> tuple[httpx.Response, AsyncIterator[bytes]]:
    """Start the request and return once the provider accepted it (errors raise SpeechError)."""
    if provider == "elevenlabs":
        url = f"{api_base or ELEVENLABS}/v1/text-to-speech/{voice}/stream"
        req = client.build_request("POST", url, headers={"xi-api-key": key},
                                   params={"output_format": _ELEVEN_FORMATS.get(fmt, "mp3_44100_128")},
                                   json={"text": text, "model_id": model})
    else:
        url = f"{api_base or OPENAI_COMPATIBLE[provider]}/audio/speech"
        req = client.build_request("POST", url, headers={"Authorization": f"Bearer {key}"},
                                   json={"model": model, "input": text, "voice": voice, "response_format": fmt})
    resp = await client.send(req, stream=True)
    if resp.status_code >= 400:
        body = (await resp.aread()).decode(errors="replace")[:500]
        await resp.aclose()
        raise SpeechError(resp.status_code, body)
    return resp, resp.aiter_bytes()
