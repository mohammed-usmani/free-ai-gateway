import asyncio

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from pipecat.frames.frames import InputAudioRawFrame, InterimTranscriptionFrame, MetricsFrame, TranscriptionFrame
from pipecat.metrics.metrics import STTUsage, STTUsageMetricsData
from pipecat.processors.frame_processor import FrameProcessor

from gateway.config import Catalog, RealtimeChain, Step
from gateway.keypool import KeyPool
from gateway.pricing import Pricer
from gateway.realtime.backends import PipecatBackend, TranscriptTap, TurnAccounting, realtime_options_for
from gateway.realtime.session import RealtimeSession
from gateway.usage import UsageStore


class FakeSTT(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            await self.push_frame(MetricsFrame(data=[STTUsageMetricsData(processor=self.name, value=STTUsage(audio_seconds=2.0))]))
            await self.push_frame(InterimTranscriptionFrame(text="hel", user_id="u", timestamp="t"))
            await self.push_frame(TranscriptionFrame(text="hello", user_id="u", timestamp="t"))
        else:
            await self.push_frame(frame, direction)


@pytest.fixture
def usage(tmp_path):
    return UsageStore(str(tmp_path / "s.db"), Pricer({"free": {"deepgram/nova-3": [
        {"effective_from": "2026-01-01", "per_second": 0.001}]}}, bundled={}, fetch=None))


async def test_transcription_backend_streams_deltas_and_records_each_utterance(usage):
    acct = TurnAccounting(usage, "sess", "free")

    def build(backend, config, emit):
        stt = FakeSTT()
        acct.register(stt, "stt", "deepgram", "nova-3")
        return [stt, TranscriptTap(emit, acct, transcription=True)]

    events = []

    async def emit(e):
        events.append(e)

    backend = PipecatBackend("transcription", "free:transcription", "-", build, accounting=acct)
    await backend.start({"type": "transcription"}, emit)
    await asyncio.sleep(0.2)
    await backend.send_audio(b"\x00\x00")
    for _ in range(50):
        if any(e["type"].endswith("completed") for e in events):
            break
        await asyncio.sleep(0.02)
    await backend.stop()
    kinds = [e["type"] for e in events]
    assert kinds.index("conversation.item.input_audio_transcription.delta") < kinds.index(
        "conversation.item.input_audio_transcription.completed")
    turns = usage.turns()
    assert len(turns) == 1 and turns[0]["components"][0]["audio_seconds"] == 2.0
    assert turns[0]["cost"] == pytest.approx(0.002)


def test_transcription_options_are_stt_only(tmp_path):
    chain = RealtimeChain([Step("gemini", "live")], [Step("deepgram", "nova-3")], "chat-fast", [Step("deepgram", "aura")])
    cats = {"free": Catalog({"chat-fast": []}, chain)}
    opts = realtime_options_for("realtime", cats, {"free": {"gemini": ["g"], "deepgram": ["d"]}}, KeyPool(str(tmp_path / "k.db")),
                                "http://x/v1", "k", None, transcription=True)
    backends = [o() for o in opts]
    assert [b.name for b in backends] == ["transcription"]
    procs = backends[0]._build(backends[0], {"type": "transcription"}, None)
    assert [type(p).__name__ for p in procs] == ["ServiceSwitcher", "TranscriptTap"]


def test_session_type_transcription_picks_transcription_options(tmp_path):
    picked = []

    class Dummy:
        name = "t"
        failure = None

        def __init__(self):
            self.failed = asyncio.Event()

        async def start(self, config, emit):
            picked.append(config.get("type"))

        async def update(self, config): ...

        async def send_audio(self, pcm): ...
        async def stop(self): ...

    def options_for(config):
        return [lambda: Dummy()] if config.get("type") == "transcription" else []

    app = FastAPI()

    @app.websocket("/rt")
    async def rt(ws: WebSocket):
        await RealtimeSession(ws, options_for, KeyPool(str(tmp_path / "k.db"))).run()

    with TestClient(app).websocket_connect("/rt") as ws:
        ws.receive_json()
        ws.send_json({"type": "session.update", "session": {"type": "transcription"}})
        ws.receive_json()
        ws.send_json({"type": "input_audio_buffer.append", "audio": "AAA="})
        ws.send_json({"type": "session.update", "session": {"x": 1}})
        ws.receive_json()
    assert picked == ["transcription"]
