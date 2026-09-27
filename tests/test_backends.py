import asyncio
import base64
import json

import pytest
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

from gateway.config import RealtimeChain, Step
from gateway.keypool import KeyPool
from gateway.realtime.backends import (
    EventSink,
    TranscriptTap,
    PipecatBackend,
    cascade_option,
    gemini_live_option,
    realtime_options,
    status_from_error,
    tools_schema,
)


class FakeVoiceModel(FrameProcessor):
    """Answers any audio with a spoken reply; audio b"die" makes it unusable (bad key)."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            if frame.audio == b"die":
                await self.push_error("401 API key not valid", force_treat_as_permanent=True)
                return
            await self.push_frame(UserStartedSpeakingFrame())
            await self.push_frame(TranscriptionFrame(text="hi", user_id="u", timestamp="t"))
            await self.push_frame(TTSStartedFrame())
            await self.push_frame(TTSTextFrame("hello", aggregated_by="word"))
            await self.push_frame(TTSAudioRawFrame(audio=b"\x01\x02", sample_rate=24000, num_channels=1))
            await self.push_frame(TTSStoppedFrame())
        else:
            await self.push_frame(frame, direction)


RUNNING: list[PipecatBackend] = []


@pytest.fixture(autouse=True)
async def _stop_backends():
    yield
    while RUNNING:
        await RUNNING.pop().stop()


async def started(build=lambda b, c, e: [FakeVoiceModel(), EventSink(e)]):
    events = []

    async def emit(e):
        events.append(e)

    backend = PipecatBackend("fake", "gemini", "k1", build)
    RUNNING.append(backend)
    await backend.start({"instructions": "x"}, emit)
    await asyncio.sleep(0.2)
    return backend, events


async def wait_for(events, kind, timeout=3.0):
    for _ in range(int(timeout / 0.02)):
        if any(e["type"] == kind for e in events):
            return [e for e in events if e["type"] == kind]
        await asyncio.sleep(0.02)
    raise AssertionError(f"no {kind} in {[e['type'] for e in events]}")


async def test_audio_turns_into_realtime_events_in_order():
    backend, events = await started()
    await backend.send_audio(b"\x00\x00")
    await wait_for(events, "response.done")
    kinds = [e["type"] for e in events]
    assert kinds.index("input_audio_buffer.speech_started") < kinds.index("response.created")
    assert kinds.index("response.created") < kinds.index("response.output_audio.delta") < kinds.index("response.done")
    delta = next(e for e in events if e["type"] == "response.output_audio.delta")
    assert base64.b64decode(delta["delta"]) == b"\x01\x02"
    assert next(e for e in events if e["type"] == "response.output_audio_transcript.delta")["delta"] == "hello"
    assert (await wait_for(events, "conversation.item.input_audio_transcription.completed"))[0]["transcript"] == "hi"
    await backend.stop()


async def test_unusable_processor_marks_backend_failed_with_status():
    backend, _ = await started()
    await backend.send_audio(b"die")
    await asyncio.wait_for(backend.failed.wait(), 3)
    assert backend.failure == ("gemini", "k1", 401)
    await backend.stop()


async def test_tool_call_round_trip_through_the_client():
    backend, events = await started()
    results = []

    class Params:
        tool_call_id, function_name, arguments = "call_1", "get_time", {"tz": "IST"}

        async def result_callback(self, result):
            results.append(result)

    call = asyncio.create_task(backend.on_function_call(Params()))
    ev = (await wait_for(events, "response.function_call_arguments.done"))[0]
    assert (ev["call_id"], ev["name"], json.loads(ev["arguments"])) == ("call_1", "get_time", {"tz": "IST"})
    await backend.function_result("call_1", '{"time": "10:00"}')
    await asyncio.wait_for(call, 2)
    assert results == [{"time": "10:00"}]
    await backend.stop()


def test_error_status_is_read_from_the_message():
    assert status_from_error(ErrorFrame(error="429 RESOURCE_EXHAUSTED quota")) == 429
    assert status_from_error(ErrorFrame(error="API key not valid")) == 401
    assert status_from_error(ErrorFrame(error="socket closed")) is None


def test_realtime_tools_become_a_pipecat_schema():
    schema = tools_schema([{"type": "function", "name": "get_time", "description": "d",
                            "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}}])
    fn = schema.standard_tools[0]
    assert (fn.name, fn.required, list(fn.properties)) == ("get_time", ["tz"], ["tz"])
    assert tools_schema(None) is None


@pytest.fixture
def pool(tmp_path):
    return KeyPool(str(tmp_path / "k.db"))


CHAIN = RealtimeChain(
    s2s=[Step("gemini", "models/gemini-3.8-live", {"voice": "Charon"})],
    stt=[Step("deepgram", "nova-3"), Step("cartesia", "ink-whisper")],
    llm_alias="chat-fast",
    tts=[Step("deepgram", "aura-2-thalia-en"), Step("cartesia", "sonic-3", {"voice": "v"})],
)


def test_options_are_none_without_usable_keys(pool):
    assert gemini_live_option(CHAIN.s2s[0], {}, pool)() is None
    assert cascade_option(CHAIN, {"deepgram": []}, pool, "http://x/v1", "g")() is None
    pool.report("free:gemini", "k1", 401)
    assert gemini_live_option(CHAIN.s2s[0], {"gemini": ["k1"]}, pool)() is None


def test_gemini_option_rotates_keys(pool):
    option = gemini_live_option(CHAIN.s2s[0], {"gemini": ["k1", "k2"]}, pool)
    assert [option().key, option().key] == ["k1", "k2"]


def test_cascade_builds_real_services_with_failover(pool):
    backend = cascade_option(CHAIN, {"deepgram": ["d"], "cartesia": ["c"]}, pool, "http://x/v1", "g")()
    processors = backend._build(backend, {"instructions": "be brief", "tools": []}, None)
    names = [type(p).__name__ for p in processors]
    assert names == ["ServiceSwitcher", "TranscriptTap", "LLMUserAggregator", "OpenAILLMService",
                     "ServiceSwitcher", "EventSink", "LLMAssistantAggregator"]
    assert [type(s).__name__ for s in processors[0].services] == ["DeepgramSTTService", "CartesiaSTTService"]
    assert [type(s).__name__ for s in processors[4].services] == ["DeepgramTTSService", "CartesiaTTSService"]


def test_options_order_is_s2s_then_cascade(pool):
    options = realtime_options(CHAIN, {"gemini": ["k"], "deepgram": ["d"]}, pool, "http://x/v1", "g")
    assert [o().name for o in options] == ["gemini:models/gemini-3.8-live", "cascade"]


async def test_cascade_shape_carries_a_turn_from_audio_to_audio():
    from pipecat.frames.frames import (
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        UserStoppedSpeakingFrame,
    )
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

    class STT(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, InputAudioRawFrame):
                await self.push_frame(UserStartedSpeakingFrame())
                await self.push_frame(TranscriptionFrame(text=frame.audio.decode(), user_id="u", timestamp="t"))
                await self.push_frame(UserStoppedSpeakingFrame())
            else:
                await self.push_frame(frame, direction)

    class LLM(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if type(frame).__name__ == "LLMContextFrame":
                said = frame.context.get_messages()[-1]["content"]
                for f in (LLMFullResponseStartFrame(), LLMTextFrame(f"you said {said}"), LLMFullResponseEndFrame()):
                    await self.push_frame(f)
            else:
                await self.push_frame(frame, direction)

    class TTS(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMTextFrame):
                for f in (TTSStartedFrame(), TTSTextFrame(frame.text, aggregated_by="sentence"),
                          TTSAudioRawFrame(audio=b"\x03\x04", sample_rate=24000, num_channels=1), TTSStoppedFrame()):
                    await self.push_frame(f)
            else:
                await self.push_frame(frame, direction)

    def build(backend, config, emit):
        agg = LLMContextAggregatorPair(LLMContext([{"role": "system", "content": config["instructions"]}]))
        return [STT(), TranscriptTap(emit), agg.user(), LLM(), TTS(), EventSink(emit), agg.assistant()]

    backend, events = await started(build)
    await backend.send_audio(b"hello")
    await wait_for(events, "response.done")
    assert next(e for e in events if e["type"] == "response.output_audio_transcript.delta")["delta"] == "you said hello"
    assert base64.b64decode(next(e for e in events if e["type"] == "response.output_audio.delta")["delta"]) == b"\x03\x04"
    assert [e["transcript"] for e in events if e["type"].endswith("transcription.completed")] == ["hello"]
    await backend.stop()


class GenaiStyleError(Exception):
    code = 1011  # a websocket close code, not an HTTP status


class QuotaDrop(FrameProcessor):
    """What Gemini Live does on quota loss: a non-permanent error carrying a close code."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            await self.push_error("1011 RESOURCE_EXHAUSTED: quota exceeded", exception=GenaiStyleError())
            return
        await self.push_frame(frame, direction)


async def test_quota_error_fails_the_backend_even_when_not_permanent():
    backend, _ = await started(lambda b, c, e: [QuotaDrop(), EventSink(e)])
    await backend.send_audio(b"x")
    await asyncio.wait_for(backend.failed.wait(), 3)
    assert backend.failure == ("gemini", "k1", 429)


async def test_pipeline_ending_on_its_own_fails_the_backend():
    backend, _ = await started()
    await backend._worker.cancel()
    await asyncio.wait_for(backend.failed.wait(), 5)


async def test_stop_does_not_count_as_failure():
    backend, _ = await started()
    await backend.stop()
    await asyncio.sleep(0.1)
    assert not backend.failed.is_set()


def test_failed_cascade_cools_down(pool):
    option = cascade_option(CHAIN, {"deepgram": ["d"]}, pool, "http://x/v1", "g")
    assert option() is not None
    pool.report("free:cascade", "-", None)
    assert option() is None


def test_close_codes_are_not_mistaken_for_http_status():
    assert status_from_error(ErrorFrame(error="socket", exception=GenaiStyleError())) is None
