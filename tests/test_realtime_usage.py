import asyncio

import pytest
from pipecat.frames.frames import (
    InputAudioRawFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData, STTUsage, STTUsageMetricsData, TTSUsageMetricsData
from pipecat.processors.frame_processor import FrameProcessor

from gateway.config import Catalog, RealtimeChain, Step
from gateway.keypool import KeyPool
from gateway.pricing import Counts, Pricer
from gateway.realtime.backends import (
    EventSink,
    PipecatBackend,
    RealtimeContext,
    TranscriptTap,
    TurnAccounting,
    realtime_options,
    realtime_options_for,
)
from gateway.usage import UsageStore


@pytest.fixture
def usage(tmp_path):
    pricer = Pricer({"paid": {
        "gemini/live": [{"effective_from": "2026-01-01", "input": 1e-6, "output": 2e-6, "audio_input": 3e-6, "audio_output": 4e-6}],
        "deepgram/nova-3": [{"effective_from": "2026-01-01", "per_second": 0.001}],
        "cartesia/sonic-3": [{"effective_from": "2026-01-01", "per_character": 0.0001}],
        "openai/m": [{"effective_from": "2026-01-01", "input": 1e-6, "output": 1e-6}],
    }}, bundled={}, fetch=None)
    return UsageStore(str(tmp_path / "state.db"), pricer)


async def run(backend, audio=b"\x00\x00"):
    events = []

    async def emit(e):
        events.append(e)

    await backend.start({}, emit)
    await asyncio.sleep(0.2)
    await backend.send_audio(audio)
    for _ in range(150):
        if any(e["type"] == "response.done" for e in events):
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.1)
    await backend.stop()
    return events


class SpeechModel(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            await self.push_frame(TTSStartedFrame())
            await self.push_frame(TTSAudioRawFrame(audio=b"\x01", sample_rate=24000, num_channels=1))
            usage = LLMTokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150,
                                  input_audio_tokens=60, output_audio_tokens=40)
            await self.push_frame(MetricsFrame(data=[LLMUsageMetricsData(processor=self.name, model="live", value=usage)]))
            await self.push_frame(TTSStoppedFrame())
        else:
            await self.push_frame(frame, direction)


async def test_s2s_turn_is_recorded_and_sent_in_response_done(usage):
    acct = TurnAccounting(usage, "sess1", "paid")

    def build(backend, config, emit):
        model = SpeechModel()
        acct.register(model, "s2s", "gemini", "live")
        return [model, EventSink(emit, acct)]

    events = await run(PipecatBackend("g", "paid:gemini", "k", build))
    done = next(e for e in events if e["type"] == "response.done")
    turn = usage.turn(done["response"]["usage"]["turn_id"])
    comp = turn["components"][0]
    assert (comp["kind"], comp["input"], comp["audio_input"], comp["output"], comp["audio_output"]) == ("s2s", 40, 60, 10, 40)
    assert done["response"]["usage"]["cost"] == pytest.approx(40e-6 + 60 * 3e-6 + 10 * 2e-6 + 40 * 4e-6)
    assert turn["session_id"] == "sess1" and turn["mode"] == "paid"


async def test_cascade_turn_combines_stt_llm_and_tts(usage):
    acct = TurnAccounting(usage, "sess2", "paid")

    class STT(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, InputAudioRawFrame):
                await self.push_frame(MetricsFrame(data=[STTUsageMetricsData(processor=self.name, value=STTUsage(audio_seconds=3.0))]))
                await self.push_frame(TranscriptionFrame(text="hi", user_id="u", timestamp="t"))
            else:
                await self.push_frame(frame, direction)

    class LLM(FrameProcessor):  # stands in for the loopback call the chat endpoint records
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame):
                turn = usage.session_turn("sess2")
                usage.add(turn, "llm", "paid", "openai", "m", Counts(input=10, output=5))
                await self.push_frame(MetricsFrame(data=[LLMUsageMetricsData(
                    processor=self.name, value=LLMTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))]))
                await self.push_frame(LLMTextFrame("hello"))
            else:
                await self.push_frame(frame, direction)

    class TTS(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMTextFrame):
                await self.push_frame(MetricsFrame(data=[TTSUsageMetricsData(processor=self.name, value=5)]))
                for f in (TTSStartedFrame(), TTSTextFrame("hello", aggregated_by="sentence"),
                          TTSAudioRawFrame(audio=b"\x02", sample_rate=24000, num_channels=1), TTSStoppedFrame()):
                    await self.push_frame(f)
            else:
                await self.push_frame(frame, direction)

    def build(backend, config, emit):
        stt, llm, tts = STT(), LLM(), TTS()
        acct.register(stt, "stt", "deepgram", "nova-3")
        acct.register(llm, "llm", "cascade", "paid/chat-fast")  # recorded by the chat endpoint, not metrics
        acct.register(tts, "tts", "cartesia", "sonic-3")
        return [stt, TranscriptTap(emit, acct), llm, tts, EventSink(emit, acct)]

    events = await run(PipecatBackend("cascade", "paid:cascade", "-", build))
    done = next(e for e in events if e["type"] == "response.done")
    turn = usage.turn(done["response"]["usage"]["turn_id"])
    assert sorted(c["kind"] for c in turn["components"]) == ["llm", "stt", "tts"]  # llm once, not twice
    assert turn["cost"] == pytest.approx(3 * 0.001 + 15e-6 + 5 * 0.0001)
    assert usage.session_turn("sess2") is None


def test_options_follow_mode_keys_and_realtime_fallback(tmp_path, usage):
    pool = KeyPool(str(tmp_path / "k.db"))
    free = Catalog({"chat-fast": []}, RealtimeChain([Step("gemini", "live")], [Step("deepgram", "nova-3")], "chat-fast",
                                                    [Step("deepgram", "aura")]))
    paid = Catalog({"chat-fast": []}, RealtimeChain([Step("openai", "gpt-realtime-2.1"), Step("gemini", "live")], [], "chat-fast", [],
                                                    fallback_to="free/realtime"))
    catalogs = {"free": free, "paid": paid}
    keys = {"free": {"gemini": ["fg"], "deepgram": ["d"]}, "paid": {"openai": ["po"], "gemini": ["pg"]}}
    ctx = RealtimeContext(usage, "s")
    names = [(o() or type("N", (), {"name": None})).name
             for o in realtime_options_for("paid/realtime", catalogs, keys, pool, "http://x/v1", "g", ctx)]
    assert names == ["openai:gpt-realtime-2.1", "gemini:live", None, "gemini:live", "cascade"]
    backend = realtime_options_for("paid/realtime", catalogs, keys, pool, "http://x/v1", "g", ctx)[1]()
    assert (backend.provider, backend.key) == ("paid:gemini", "pg")
    assert [o().name for o in realtime_options_for("realtime", catalogs, keys, pool, "http://x/v1", "g", ctx)] == ["gemini:live", "cascade"]


def test_openai_realtime_backend_builds(tmp_path, usage):
    pool = KeyPool(str(tmp_path / "k.db"))
    chain = RealtimeChain([Step("openai", "gpt-realtime-2.1", {"voice": "marin"})], [], "chat-fast", [])
    backend = realtime_options(chain, {"openai": ["k"]}, pool, "http://x/v1", "g", mode="paid", ctx=RealtimeContext(usage, "s"))[0]()
    processors = backend._build(backend, {"instructions": "hi", "tools": []}, None)
    assert type(processors[0]).__name__ == "OpenAIRealtimeLLMService"


class HalfAnswer(FrameProcessor):
    """Starts speaking and reports usage, then the call ends before the response finishes."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            await self.push_frame(TTSStartedFrame())
            usage = LLMTokenUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14)
            await self.push_frame(MetricsFrame(data=[LLMUsageMetricsData(processor=self.name, model="live", value=usage)]))
        else:
            await self.push_frame(frame, direction)


async def test_hang_up_mid_response_still_records_the_turn(usage):
    acct = TurnAccounting(usage, "sess3", "paid")

    def build(backend, config, emit):
        model = HalfAnswer()
        acct.register(model, "s2s", "gemini", "live")
        return [model, EventSink(emit, acct)]

    backend = PipecatBackend("g", "paid:gemini", "k", build, accounting=acct)
    events = []

    async def emit(e):
        events.append(e)

    await backend.start({}, emit)
    await asyncio.sleep(0.2)
    await backend.send_audio(b"\x00")
    await asyncio.sleep(0.3)
    backend.fail(429)
    await backend.stop()
    turns = usage.turns()
    assert len(turns) == 1 and turns[0]["ended_at"] is not None
    comp = turns[0]["components"][0]
    assert (comp["kind"], comp["input"], comp["attempt"]) == ("s2s", 10, "failed")
