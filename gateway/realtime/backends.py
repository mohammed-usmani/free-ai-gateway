"""Realtime backends as Pipecat pipelines with no transport.

Client audio is queued into the pipeline; `EventSink` at the tail turns Pipecat frames
into OpenAI Realtime server events. A pipeline error that leaves a processor unusable
(bad key, quota, dead provider) marks the backend failed, and the session moves on.
"""

import asyncio
import base64
from dataclasses import dataclass
import json
import logging
import re
import uuid
from collections.abc import Callable

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    ErrorFrame,
    InterimTranscriptionFrame,
    MetricsFrame,
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

from pipecat.metrics.metrics import LLMUsageMetricsData, STTUsageMetricsData, TTSUsageMetricsData

from gateway.config import Catalog, RealtimeChain, Step
from gateway.keypool import KeyPool
from gateway.pricing import Counts, counts_from_pipecat
from gateway.realtime.session import Emit, event
from gateway.usage import UsageStore

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


@dataclass
class RealtimeContext:
    usage: UsageStore
    session_id: str
    token: str | None = None  # private tag for the cascade's loopback chat call


class TurnAccounting:
    """Collects one realtime session's usage into turns.

    Services are registered with what they are (`s2s`, `stt`, `llm`, `tts`) and which
    provider/model to price them as. Pipecat usage metrics are added to the open turn when
    it closes; the cascade's LLM usage is recorded by the chat endpoint (it knows which
    provider really answered), so its Pipecat metrics are ignored here.
    """

    def __init__(self, usage: UsageStore, session_id: str, mode: str, token: str | None = None):
        self.usage, self.session_id, self.mode = usage, session_id, mode
        self.token = token or session_id
        self._services: dict[str, tuple[str, str, str]] = {}
        self._pending: dict[tuple[str, str, str], Counts] = {}
        self._turn: str | None = None
        self._last: str | None = None

    def register(self, service: FrameProcessor, kind: str, provider: str, model: str):
        self._services[service.name] = (kind, provider, model)

    def on_metrics(self, data):
        entry = self._services.get(data.processor)
        if not entry:
            return
        kind = entry[0]
        if isinstance(data, LLMUsageMetricsData) and kind == "s2s":
            # OpenAI Realtime and Gemini Live report gross input; Gemini's thinking sits outside output.
            counts = counts_from_pipecat(data.value, net=False, reasoning_in_output=entry[1] != "gemini")
        elif isinstance(data, STTUsageMetricsData):
            counts = Counts(audio_seconds=data.value.audio_seconds)
        elif isinstance(data, TTSUsageMetricsData):
            counts = Counts(characters=int(data.value))
        else:
            return
        if self._turn is None and kind != "stt" and self._last:
            # Arrived after response.done (s2s usage can trail the audio): add to that turn.
            self.usage.add(self._last, kind, self.mode, entry[1], entry[2], counts)
            return
        self._pending[entry] = self._pending.get(entry, Counts()) + counts

    def open(self) -> str:
        if self._turn is None:
            self._turn = self.usage.open_turn(self.mode, "realtime", "realtime", session_id=self.session_id)
            self.usage.set_session_turn(self.token, self._turn)
        return self._turn

    def close(self, attempt: str = "ok") -> dict | None:
        if self._turn is None:
            if not self._pending:
                return None
            self.open()  # usage measured outside any response: still billed, still recorded
        turn, self._turn, self._last = self._turn, None, self._turn
        self.usage.set_session_turn(self.token, None)
        for (kind, provider, model), counts in self._pending.items():
            self.usage.add(turn, kind, self.mode, provider, model, counts, attempt=attempt)
        self._pending.clear()
        self.usage.close_turn(turn)
        return self.usage.turn(turn)


def _usage_summary(turn: dict | None) -> dict | None:
    if not turn:
        return None
    keep = ("kind", "mode", "provider", "model", "input", "cache_read", "cache_write", "output", "reasoning",
            "audio_input", "audio_output", "audio_cache_read", "audio_seconds", "characters", "cost", "price_source")
    return {"turn_id": turn["turn_id"], "cost": turn["cost"], "cost_complete": turn["cost_complete"],
            "components": [{k: c[k] for k in keep} for c in turn["components"]]}


class TranscriptTap(FrameProcessor):
    """Emits the caller's transcript (interim deltas, then the final text). Sits right after
    STT: the user aggregator consumes it. In a transcription-only session every final
    utterance is its own turn."""

    def __init__(self, emit: Emit, accounting: TurnAccounting | None = None, transcription: bool = False):
        super().__init__()
        self._emit = emit
        self._accounting = accounting
        self._transcription = transcription
        self._item: str | None = None

    def _item_id(self) -> str:
        self._item = self._item or f"item_{uuid.uuid4().hex[:20]}"
        return self._item

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, MetricsFrame) and self._transcription and self._accounting:
            for data in frame.data:  # no EventSink in a transcription session: count STT here
                self._accounting.on_metrics(data)
        elif isinstance(frame, InterimTranscriptionFrame):
            await self._emit(event("conversation.item.input_audio_transcription.delta",
                                   item_id=self._item_id(), content_index=0, delta=frame.text))
        elif isinstance(frame, TranscriptionFrame):
            if self._accounting:
                self._accounting.open()  # the caller finished speaking: this turn starts
            await self._emit(event("conversation.item.input_audio_transcription.completed",
                                   item_id=self._item_id(), content_index=0, transcript=frame.text))
            self._item = None
            if self._transcription and self._accounting:
                self._accounting.close()
        await self.push_frame(frame, direction)


class EventSink(TranscriptTap):
    """Pipecat frames → OpenAI Realtime server events. Sits right after the voice output
    (TTS or the speech-to-speech model): the assistant aggregator consumes spoken text."""

    def __init__(self, emit: Emit, accounting: TurnAccounting | None = None):
        super().__init__(emit, accounting)
        self._response: str | None = None
        self._item = ""
        self._audio_sent = False

    async def _open(self):
        if not self._response:
            self._response = f"resp_{uuid.uuid4().hex[:20]}"
            self._item = f"item_{uuid.uuid4().hex[:20]}"
            self._audio_sent = False
            if self._accounting:
                self._accounting.open()
            await self._emit(event("response.created", response={"id": self._response, "status": "in_progress"}))

    async def _close(self, status: str = "completed"):
        """Finish the open response (if any) for the client and close its turn."""
        if self._response:
            ids = {"response_id": self._response, "item_id": self._item, "output_index": 0, "content_index": 0}
            await self._emit(event("response.output_audio.done", **ids))
            await self._emit(event("response.output_audio_transcript.done", **ids))
            usage = (_usage_summary(self._accounting.close(attempt="failed" if status == "failed" else "ok"))
                     if self._accounting else None)
            await self._emit(event("response.done", response={"id": self._response, "status": status, "usage": usage}))
            self._response = None

    def _ids(self) -> dict:
        return {"response_id": self._response, "item_id": self._item, "output_index": 0, "content_index": 0}

    async def process_frame(self, frame, direction: FrameDirection):
        await FrameProcessor.process_frame(self, frame, direction)
        if isinstance(frame, MetricsFrame):
            if self._accounting:
                for data in frame.data:
                    self._accounting.on_metrics(data)
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._emit(event("input_audio_buffer.speech_started", audio_start_ms=0, item_id=""))
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._emit(event("input_audio_buffer.speech_stopped", audio_end_ms=0, item_id=""))
        elif isinstance(frame, InterruptionFrame):
            await self._close("cancelled")
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
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
                 build: Callable[["PipecatBackend", dict, Emit], list[FrameProcessor]],
                 accounting: "TurnAccounting | None" = None):
        self.name, self.provider, self.key = name, provider, key
        self.accounting = accounting
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
        self._sinks = [p for p in processors if isinstance(p, EventSink)]
        self._worker = PipelineWorker(
            Pipeline(processors),
            params=PipelineParams(audio_in_sample_rate=RATE, audio_out_sample_rate=RATE,
                                  enable_metrics=True, enable_usage_metrics=True),
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
        for sink in getattr(self, "_sinks", []):  # a response cut off here still gets its response.done
            try:
                await sink._close("failed" if self.failed.is_set() else "cancelled")
            except Exception:  # noqa: BLE001 — the client may already be gone
                log.debug("could not close the open response", exc_info=True)
        self._sinks = []
        if self.accounting:  # hang-up or failover mid-response: keep what was measured
            try:
                self.accounting.close(attempt="failed" if self.failed.is_set() else "ok")
            except Exception:  # noqa: BLE001 — bookkeeping must not block teardown
                log.exception("could not record realtime usage")


# --- factories ---------------------------------------------------------------------------
# An option builds a backend with a fresh key for one session, or returns None when it has
# no usable key. Key health lives in the pool under "<mode>:<provider>".


def _voice(config: dict, step: Step, default: str) -> str:
    voice = config.get("voice") or ((config.get("audio") or {}).get("output") or {}).get("voice")
    return voice or step.options.get("voice", default)


def _accounting(ctx: RealtimeContext | None, mode: str) -> TurnAccounting | None:
    return TurnAccounting(ctx.usage, ctx.session_id, mode, ctx.token) if ctx else None


def gemini_live_option(step: Step, keys: dict[str, list[str]], pool: KeyPool, mode: str = "free",
                       ctx: RealtimeContext | None = None):
    pool_name = f"{mode}:{step.provider}"

    def option():
        key = pool.pick(pool_name, keys.get(step.provider, []))
        if not key:
            return None
        acct = _accounting(ctx, mode)

        def build(backend: PipecatBackend, config: dict, emit: Emit):
            from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

            voice = _voice(config, step, "Charon")
            llm = GeminiLiveLLMService(
                api_key=key, tools=tools_schema(config.get("tools")),
                settings=GeminiLiveLLMService.Settings(model=step.model, voice="Charon" if voice == "alloy" else voice,
                                                       system_instruction=config.get("instructions")),
            )
            llm.register_function(None, backend.on_function_call)
            if acct:
                acct.register(llm, "s2s", step.provider, step.model.removeprefix("models/"))
            return [llm, EventSink(emit, acct)]

        return PipecatBackend(f"{step.provider}:{step.model}", pool_name, key, build, acct)

    return option


def openai_realtime_option(step: Step, keys: dict[str, list[str]], pool: KeyPool, mode: str = "free",
                           ctx: RealtimeContext | None = None):
    pool_name = f"{mode}:{step.provider}"

    def option():
        key = pool.pick(pool_name, keys.get(step.provider, []))
        if not key:
            return None
        acct = _accounting(ctx, mode)

        def build(backend: PipecatBackend, config: dict, emit: Emit):
            from pipecat.services.openai.realtime.events import AudioConfiguration, AudioOutput, SessionProperties
            from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

            props = SessionProperties(
                instructions=config.get("instructions"), tools=tools_schema(config.get("tools")),
                audio=AudioConfiguration(output=AudioOutput(voice=_voice(config, step, "marin"))),
            )
            llm = OpenAIRealtimeLLMService(
                api_key=key, settings=OpenAIRealtimeLLMService.Settings(model=step.model, session_properties=props))
            llm.register_function(None, backend.on_function_call)
            if acct:
                acct.register(llm, "s2s", step.provider, step.model)
            return [llm, EventSink(emit, acct)]

        return PipecatBackend(f"{step.provider}:{step.model}", pool_name, key, build, acct)

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
        from pipecat.services.deepgram.tts import DeepgramTTSService as C
        return C(api_key=key, settings=C.Settings(voice=step.model))  # Deepgram's voice id is the model
    if p == "cartesia":
        from pipecat.services.cartesia.tts import CartesiaTTSService as C
    elif p == "rime":
        from pipecat.services.rime.tts import RimeTTSService as C
    elif p == "elevenlabs":
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService as C
    else:
        raise ValueError(f"no streaming TTS adapter for {p}")
    return C(api_key=key, settings=C.Settings(model=step.model, voice=voice))


def _pick_services(steps: list[Step], keys, pool: KeyPool, mode: str, make, acct, kind) -> list:
    services = []
    for step in steps:
        key = pool.pick(f"{mode}:{step.provider}", keys.get(step.provider, []))
        if key:
            service = make(step, key)
            if acct:
                acct.register(service, kind, step.provider, step.model)
            services.append(service)
    return services


def cascade_option(chain: RealtimeChain, keys: dict[str, list[str]], pool: KeyPool,
                   llm_base_url: str, gateway_key: str, mode: str = "free", ctx: RealtimeContext | None = None,
                   private: bool = False):
    """STT → the gateway's own chat alias → TTS. STT and TTS fail over inside the session."""
    pool_name = f"{mode}:cascade"
    llm_model = chain.llm_alias if mode == "free" else f"{mode}/{chain.llm_alias}"

    def option():
        if not any(keys.get(s.provider) for s in chain.stt) or not any(keys.get(s.provider) for s in chain.tts):
            return None
        if pool.pick(pool_name, ["-"]) is None:  # the cascade failed recently; don't loop on it
            return None
        acct = _accounting(ctx, mode)

        def build(backend: PipecatBackend, config: dict, emit: Emit):
            from pipecat.audio.vad.silero import SileroVADAnalyzer
            from pipecat.pipeline.service_switcher import ServiceSwitcher, ServiceSwitcherStrategyFailover
            from pipecat.processors.aggregators.llm_context import LLMContext
            from pipecat.processors.aggregators.llm_response_universal import (
                LLMContextAggregatorPair,
                LLMUserAggregatorParams,
            )
            from pipecat.services.openai.llm import OpenAILLMService

            stts = _pick_services(chain.stt, keys, pool, mode, _stt, acct, "stt")
            ttss = _pick_services(chain.tts, keys, pool, mode, _tts, acct, "tts")
            if not stts or not ttss:
                raise RuntimeError("every STT or TTS key is cooling down")
            # The LLM step calls this gateway, tagged with the session so its usage (with the
            # provider that really answered) lands on the session's current turn.
            headers = {"x-gateway-session": ctx.token or ctx.session_id} if ctx else {}
            if private:
                headers["x-gateway-private"] = "true"
            llm = OpenAILLMService(api_key=gateway_key, base_url=llm_base_url, default_headers=headers,
                                   settings=OpenAILLMService.Settings(model=llm_model))
            llm.register_function(None, backend.on_function_call)
            if acct:
                acct.register(llm, "llm", "cascade", llm_model)
            messages = [{"role": "system", "content": config["instructions"]}] if config.get("instructions") else []
            tools = tools_schema(config.get("tools"))
            context = LLMContext(messages, tools) if tools else LLMContext(messages)
            aggregators = LLMContextAggregatorPair(
                context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
            )
            return [
                ServiceSwitcher(stts, ServiceSwitcherStrategyFailover),
                TranscriptTap(emit, acct),
                aggregators.user(),
                llm,
                ServiceSwitcher(ttss, ServiceSwitcherStrategyFailover),
                EventSink(emit, acct),
                aggregators.assistant(),
            ]

        return PipecatBackend("cascade", pool_name, "-", build, acct)

    return option


def transcription_option(chain: RealtimeChain, keys, pool: KeyPool, mode: str = "free",
                         ctx: RealtimeContext | None = None):
    """Speech-to-text only (OpenAI transcription sessions): STT fails over inside the session."""
    pool_name = f"{mode}:transcription"

    def option():
        if not any(keys.get(s.provider) for s in chain.stt) or pool.pick(pool_name, ["-"]) is None:
            return None
        acct = _accounting(ctx, mode)

        def build(backend: PipecatBackend, config: dict, emit: Emit):
            from pipecat.pipeline.service_switcher import ServiceSwitcher, ServiceSwitcherStrategyFailover

            stts = _pick_services(chain.stt, keys, pool, mode, _stt, acct, "stt")
            if not stts:
                raise RuntimeError("every STT key is cooling down")
            return [ServiceSwitcher(stts, ServiceSwitcherStrategyFailover), TranscriptTap(emit, acct, transcription=True)]

        return PipecatBackend("transcription", pool_name, "-", build, acct)

    return option


_S2S = {"gemini": gemini_live_option, "openai": openai_realtime_option}


def realtime_options(chain: RealtimeChain, keys, pool: KeyPool, llm_base_url: str, gateway_key: str,
                     mode: str = "free", ctx: RealtimeContext | None = None, allowed=lambda provider: True,
                     private: bool = False):
    """`allowed` filters providers (privacy); the cascade keeps only allowed STT/TTS steps."""
    cascade = RealtimeChain(chain.s2s, [s for s in chain.stt if allowed(s.provider)], chain.llm_alias,
                            [s for s in chain.tts if allowed(s.provider)], chain.fallback_to)
    return [_S2S[s.provider](s, keys, pool, mode, ctx) for s in chain.s2s if s.provider in _S2S and allowed(s.provider)] + [
        cascade_option(cascade, keys, pool, llm_base_url, gateway_key, mode, ctx, private)
    ]


def realtime_options_for(model: str, catalogs: dict[str, Catalog], mode_keys: dict[str, dict[str, list[str]]],
                         pool: KeyPool, llm_base_url: str, gateway_key: str, ctx: RealtimeContext | None,
                         private: bool = False, transcription: bool = False):
    """`paid/realtime` → paid options, then those of `realtime.fallback_to` (another mode)."""
    head, _, rest = model.partition("/")
    mode = head if head in catalogs and rest else "free"
    options, seen = [], set()
    while mode and mode in catalogs and mode not in seen:
        seen.add(mode)
        chain = catalogs[mode].realtime
        catalog = catalogs[mode]
        allowed = (lambda p, c=catalog: c.is_private(p)) if private else (lambda p: True)
        if transcription:
            stt_only = RealtimeChain([], [s for s in chain.stt if allowed(s.provider)], chain.llm_alias, [])
            options.append(transcription_option(stt_only, mode_keys.get(mode, {}), pool, mode, ctx))
        else:
            options += realtime_options(chain, mode_keys.get(mode, {}), pool, llm_base_url, gateway_key, mode, ctx,
                                        allowed, private)
        mode = (chain.fallback_to or "").partition("/")[0] or None
    return options
