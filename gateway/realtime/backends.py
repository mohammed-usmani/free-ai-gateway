"""Realtime backends as Pipecat pipelines with no transport.

Client audio is queued into the pipeline; `EventSink` at the tail turns Pipecat frames
into OpenAI Realtime server events. A pipeline error that leaves a processor unusable
(bad key, quota, dead provider) marks the backend failed, and the session moves on.
"""

import asyncio
import base64
import json
import logging
import re
import uuid
from collections.abc import Callable

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from gateway.config import RealtimeChain, Step
from gateway.keypool import KeyPool
from gateway.realtime.session import Emit, event

log = logging.getLogger(__name__)

RATE = 24000  # OpenAI Realtime default PCM16 rate, both directions
TOOL_TIMEOUT = 30


def status_from_error(frame: ErrorFrame) -> int | None:
    status = getattr(frame.exception, "status_code", None) or getattr(frame.exception, "code", None)
    if isinstance(status, int) and 100 <= status < 600:  # not websocket close codes (1008, 1011)
        return status
    text = f"{getattr(frame, 'category', '')} {frame.error}".lower()
    if re.search(r"\b(401|403)\b|auth|api key|permission", text):
        return 401
    if re.search(r"\b429\b|quota|rate.?limit|resource.?exhausted", text):
        return 429
    return None


def tools_schema(tools: list[dict] | None) -> ToolsSchema | None:
    """OpenAI Realtime tool definitions → Pipecat's provider-neutral schema."""
    if not tools:
        return None
    functions = []
    for t in tools:
        params = t.get("parameters") or {}
        functions.append(FunctionSchema(
            name=t["name"], description=t.get("description", ""),
            properties=params.get("properties", {}), required=params.get("required", []),
        ))
    return ToolsSchema(standard_tools=functions)


class TranscriptTap(FrameProcessor):
    """Emits the caller's transcript. Sits right after STT: the user aggregator consumes it."""

    def __init__(self, emit: Emit):
        super().__init__()
        self._emit = emit

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            await self._emit(event("conversation.item.input_audio_transcription.completed",
                                   item_id=f"item_{uuid.uuid4().hex[:20]}", content_index=0, transcript=frame.text))
        await self.push_frame(frame, direction)


class EventSink(TranscriptTap):
    """Pipecat frames → OpenAI Realtime server events. Sits right after the voice output
    (TTS or the speech-to-speech model): the assistant aggregator consumes spoken text."""

    def __init__(self, emit: Emit):
        super().__init__(emit)
        self._response: str | None = None
        self._item = ""
        self._audio_sent = False

    async def _open(self):
        if not self._response:
            self._response = f"resp_{uuid.uuid4().hex[:20]}"
            self._item = f"item_{uuid.uuid4().hex[:20]}"
            self._audio_sent = False
            await self._emit(event("response.created", response={"id": self._response, "status": "in_progress"}))

    async def _close(self, status: str = "completed"):
        if self._response:
            ids = {"response_id": self._response, "item_id": self._item, "output_index": 0, "content_index": 0}
            await self._emit(event("response.output_audio.done", **ids))
            await self._emit(event("response.output_audio_transcript.done", **ids))
            await self._emit(event("response.done", response={"id": self._response, "status": status}))
            self._response = None

    def _ids(self) -> dict:
        return {"response_id": self._response, "item_id": self._item, "output_index": 0, "content_index": 0}

    async def process_frame(self, frame, direction: FrameDirection):
        await FrameProcessor.process_frame(self, frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            await self._emit(event("input_audio_buffer.speech_started", audio_start_ms=0, item_id=""))
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._emit(event("input_audio_buffer.speech_stopped", audio_end_ms=0, item_id=""))
        elif isinstance(frame, InterruptionFrame):
            await self._close("cancelled")
        elif isinstance(frame, TranscriptionFrame):
            await TranscriptTap.process_frame(self, frame, direction)
            return
        elif isinstance(frame, TTSStartedFrame):
            await self._open()
        elif isinstance(frame, OutputAudioRawFrame):
            await self._open()
            self._audio_sent = True
            await self._emit(event("response.output_audio.delta", **self._ids(),
                                   delta=base64.b64encode(frame.audio).decode()))
        elif isinstance(frame, TTSTextFrame):
            await self._open()
            await self._emit(event("response.output_audio_transcript.delta", **self._ids(), delta=frame.text))
        elif isinstance(frame, TTSStoppedFrame):
            await self._close()
        elif isinstance(frame, LLMFullResponseEndFrame) and self._response and not self._audio_sent:
            await self._close()
        await self.push_frame(frame, direction)


class PipecatBackend:
    """One realtime backend: a Pipecat pipeline built by `build(backend, config, emit)`.

    The build places `EventSink(emit)` (and `TranscriptTap(emit)` when an aggregator would
    swallow transcripts) where the frames it reads still flow."""

    def __init__(self, name: str, provider: str, key: str,
                 build: Callable[["PipecatBackend", dict, Emit], list[FrameProcessor]]):
        self.name, self.provider, self.key = name, provider, key
        self._build = build
        self.failed = asyncio.Event()
        self.failure: tuple[str, str, int | None] | None = None
        self._worker: PipelineWorker | None = None
        self._runner_task: asyncio.Task | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._emit: Emit | None = None
        self._config: dict = {}
        self._stopping = False

    async def start(self, config: dict, emit: Emit):
        self._emit, self._config = emit, dict(config)
        processors = self._build(self, self._config, emit)
        self._worker = PipelineWorker(
            Pipeline(processors),
            params=PipelineParams(audio_in_sample_rate=RATE, audio_out_sample_rate=RATE),
            idle_timeout_secs=None,
            enable_rtvi=False,
        )

        @self._worker.event_handler("on_pipeline_error")
        async def on_error(_worker, frame: ErrorFrame):
            # Gemini Live reports quota loss and reconnect give-ups as non-permanent errors,
            # so key/quota errors and give-ups count as failures too, not only unusable ones.
            status = status_from_error(frame)
            gave_up = "max consecutive" in str(frame.error).lower()
            if frame.fatal or (frame.processor and not frame.processor.is_usable) or status in (401, 403, 429) or gave_up:
                self.fail(status)

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self._worker)
        self._stopping = False
        self._runner_task = asyncio.create_task(runner.run())
        # A pipeline that ends without us stopping it (setup timeout, cancel on error) is dead.
        self._runner_task.add_done_callback(lambda _t: None if self._stopping else self.fail(None))

    def fail(self, status: int | None):
        if not self.failed.is_set():
            self.failure = (self.provider, self.key, status)
            self.failed.set()

    async def update(self, config: dict):
        # Instructions and tools are fixed when a pipeline starts, so restart it.
        await self.stop()
        await self.start(config, self._emit)

    async def send_audio(self, pcm16_24k: bytes):
        await self._worker.queue_frame(InputAudioRawFrame(audio=pcm16_24k, sample_rate=RATE, num_channels=1))

    async def send_text(self, text: str):
        await self._worker.queue_frame(LLMMessagesAppendFrame([{"role": "user", "content": text}], run_llm=False))

    async def respond(self):
        await self._worker.queue_frame(LLMRunFrame())

    async def cancel(self):
        await self._worker.queue_frame(InterruptionFrame())

    async def function_result(self, call_id: str, output: str):
        future = self._pending.pop(call_id, None)
        if future and not future.done():
            future.set_result(output)

    async def on_function_call(self, params):
        """Catch-all tool handler: the client runs the tool and sends the output back."""
        future = asyncio.get_running_loop().create_future()
        self._pending[params.tool_call_id] = future
        await self._emit(event("response.function_call_arguments.done", call_id=params.tool_call_id,
                               name=params.function_name, arguments=json.dumps(params.arguments),
                               response_id="", item_id="", output_index=0))
        try:
            output = await asyncio.wait_for(future, TOOL_TIMEOUT)
        except TimeoutError:
            output = json.dumps({"error": "tool result timed out"})
        finally:
            self._pending.pop(params.tool_call_id, None)
        try:
            result = json.loads(output)
        except (TypeError, ValueError):
            result = output
        await params.result_callback(result)

    async def stop(self):
        self._stopping = True
        for future in self._pending.values():
            future.cancel()
        self._pending.clear()
        if self._worker:
            await self._worker.cancel()
        if self._runner_task:
            try:
                await asyncio.wait_for(self._runner_task, 10)
            except (TimeoutError, asyncio.CancelledError):
                self._runner_task.cancel()
        self._worker = self._runner_task = None


# --- factories -------------------------------------------------------------------------


def gemini_live_option(step: Step, keys: dict[str, list[str]], pool: KeyPool):
    def option():
        key = pool.pick(step.provider, keys.get(step.provider, []))
        if not key:
            return None

        def build(backend: PipecatBackend, config: dict, emit: Emit):
            from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

            llm = GeminiLiveLLMService(
                api_key=key,
                model=step.model,
                voice_id=config.get("voice") if config.get("voice") not in (None, "alloy") else step.options.get("voice", "Charon"),
                system_instruction=config.get("instructions"),
                tools=tools_schema(config.get("tools")),
            )
            llm.register_function(None, backend.on_function_call)
            return [llm, EventSink(emit)]

        return PipecatBackend(f"{step.provider}:{step.model}", step.provider, key, build)

    return option


def _stt(step: Step, key: str):
    p = step.provider
    if p == "deepgram":
        from pipecat.services.deepgram.stt import DeepgramSTTService
        return DeepgramSTTService(api_key=key)
    if p == "assemblyai":
        from pipecat.services.assemblyai.stt import AssemblyAISTTService
        return AssemblyAISTTService(api_key=key)
    if p == "speechmatics":
        from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
        return SpeechmaticsSTTService(api_key=key)
    if p == "cartesia":
        from pipecat.services.cartesia.stt import CartesiaSTTService
        return CartesiaSTTService(api_key=key)
    raise ValueError(f"no streaming STT adapter for {p}")


def _tts(step: Step, key: str):
    p, voice = step.provider, step.options.get("voice")
    if p == "deepgram":
        from pipecat.services.deepgram.tts import DeepgramTTSService
        return DeepgramTTSService(api_key=key, voice=step.model)
    if p == "cartesia":
        from pipecat.services.cartesia.tts import CartesiaTTSService
        return CartesiaTTSService(api_key=key, voice_id=voice, model=step.model)
    if p == "rime":
        from pipecat.services.rime.tts import RimeTTSService
        return RimeTTSService(api_key=key, voice_id=voice, model=step.model)
    if p == "elevenlabs":
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
        return ElevenLabsTTSService(api_key=key, voice_id=voice, model=step.model)
    raise ValueError(f"no streaming TTS adapter for {p}")


def _pick_services(steps: list[Step], keys, pool: KeyPool, make) -> list:
    services = []
    for step in steps:
        key = pool.pick(step.provider, keys.get(step.provider, []))
        if key:
            services.append(make(step, key))
    return services


def cascade_option(chain: RealtimeChain, keys: dict[str, list[str]], pool: KeyPool,
                   llm_base_url: str, gateway_key: str):
    """STT → the gateway's own chat alias → TTS. STT and TTS fail over inside the session."""

    def option():
        if not any(keys.get(s.provider) for s in chain.stt) or not any(keys.get(s.provider) for s in chain.tts):
            return None
        if pool.pick("cascade", ["-"]) is None:  # the cascade failed recently; don't loop on it
            return None

        def build(backend: PipecatBackend, config: dict, emit: Emit):
            from pipecat.audio.vad.silero import SileroVADAnalyzer
            from pipecat.pipeline.service_switcher import ServiceSwitcher, ServiceSwitcherStrategyFailover
            from pipecat.processors.aggregators.llm_context import LLMContext
            from pipecat.processors.aggregators.llm_response_universal import (
                LLMContextAggregatorPair,
                LLMUserAggregatorParams,
            )
            from pipecat.services.openai.llm import OpenAILLMService

            stts = _pick_services(chain.stt, keys, pool, _stt)
            ttss = _pick_services(chain.tts, keys, pool, _tts)
            if not stts or not ttss:
                raise RuntimeError("every STT or TTS key is cooling down")
            llm = OpenAILLMService(api_key=gateway_key, base_url=llm_base_url, model=chain.llm_alias)
            llm.register_function(None, backend.on_function_call)
            messages = [{"role": "system", "content": config["instructions"]}] if config.get("instructions") else []
            tools = tools_schema(config.get("tools"))
            context = LLMContext(messages, tools) if tools else LLMContext(messages)
            aggregators = LLMContextAggregatorPair(
                context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
            )
            return [
                ServiceSwitcher(stts, ServiceSwitcherStrategyFailover),
                TranscriptTap(emit),
                aggregators.user(),
                llm,
                ServiceSwitcher(ttss, ServiceSwitcherStrategyFailover),
                EventSink(emit),
                aggregators.assistant(),
            ]

        return PipecatBackend("cascade", "cascade", "-", build)

    return option


def realtime_options(chain: RealtimeChain, keys, pool: KeyPool, llm_base_url: str, gateway_key: str):
    return [gemini_live_option(s, keys, pool) for s in chain.s2s if s.provider == "gemini"] + [
        cascade_option(chain, keys, pool, llm_base_url, gateway_key)
    ]
