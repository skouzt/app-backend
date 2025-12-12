"""Base bot framework - SIMPLIFIED AND WORKING."""

from abc import ABC, abstractmethod
from typing import Optional, List
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.pipeline import Pipeline
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIProcessor
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.services.deepseek.llm import DeepSeekLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.transports.daily.transport import DailyTransport, DailyParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.sync.event_notifier import EventNotifier
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.transcriptions.language import Language
from pipecat.frames.frames import (
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TranscriptionFrame,
    StartInterruptionFrame,
    BotInterruptionFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
)
from openai.types.chat import ChatCompletionMessageParam
from loguru import logger
from services.kokoro_tts_enhanced import EnhancedKokoroTTSService
from .smart_endpointing import (
    CLASSIFIER_SYSTEM_INSTRUCTION,
    CompletenessCheck,
    OutputGate,
    StatementJudgeContextFilter,
)

class BaseBot(ABC):

    def __init__(self, config, system_messages: Optional[List[ChatCompletionMessageParam]] = None):
        """Initialize bot with core services and pipeline components."""
        self.config = config

        # Initialize STT service
        logger.info("Initializing Whisper STT service")
        self.stt = WhisperSTTService(
            model="base",
            device="cpu",
            compute_type="int8",
            language=Language.EN,
            no_speech_prob=0.6,
            vad_filter=False,  # Don't double-filter with SileroVAD
        )
        logger.success("Whisper STT initialized")

        # Initialize TTS service - FIXED VERSION
        logger.info(f"Initializing TTS service: {config.tts_provider}")
        match config.tts_provider:
            case "kokoro" | "kokoro_streaming":
                # Local ONNX Kokoro
                self.tts = EnhancedKokoroTTSService(
                    voice=config.kokoro_voice,
                    base_speed=config.kokoro_speed,
                    params={
                        "enable_ssml": True,
                        "enable_emotion": True,
                        "default_emotion": "empathetic"
                    }
                )
                logger.success(f"Kokoro ONNX TTS initialized: {config.kokoro_voice}")
                
            case "kokoro_fastapi":
                # Remote FastAPI Kokoro
                from services.kokoro_fastapi_tts import KokoroFastAPIService
                self.tts = KokoroFastAPIService(
                    voice=config.kokoro_voice,
                    speed=config.kokoro_speed,
                    base_url=config.kokoro_fastapi_url,
                    endpoint=config.kokoro_fastapi_endpoint,
                )
                logger.success(f"Kokoro FastAPI TTS initialized: {config.kokoro_fastapi_url}")
                
            case _:
                raise ValueError(f"Invalid TTS provider: {config.tts_provider}")

        # Initialize LLM services
        logger.info(f"Initializing LLM service: {config.llm_provider}")
        match config.llm_provider:
            case "deepseek":
                if not config.deepseek_api_key:
                    raise ValueError("DeepSeek API key is required")
                self.conversation_llm = DeepSeekLLMService(
                    api_key=config.deepseek_api_key,
                    model=config.deepseek_model,
                    base_url="https://api.deepseek.com/v1",
                    params=config.deepseek_params,
                )
                self.llm = self.conversation_llm
                if not config.google_api_key:
                    raise ValueError("Google API key required for smart endpointing")
                self.statement_llm = GoogleLLMService(
                    name="StatementJudger",
                    api_key=config.google_api_key,
                    model=config.classifier_model,
                    temperature=0.0,
                    system_instruction=CLASSIFIER_SYSTEM_INSTRUCTION,
                )
            case "google":
                if not config.google_api_key:
                    raise ValueError("Google API key is required")
                self.conversation_llm = GoogleLLMService(
                    api_key=config.google_api_key,
                    model=config.google_model,
                    params=config.google_params,
                    system_instruction="You are a helpful voice assistant.",
                )
                self.llm = self.conversation_llm
                self.statement_llm = GoogleLLMService(
                    name="StatementJudger",
                    api_key=config.google_api_key,
                    model=config.classifier_model,
                    temperature=0.0,
                    system_instruction=CLASSIFIER_SYSTEM_INSTRUCTION,
                )
            case "openai":
                if not config.openai_api_key:
                    raise ValueError("OpenAI API key is required")
                self.conversation_llm = OpenAILLMService(
                    api_key=config.openai_api_key,
                    model=config.openai_model,
                    params=config.openai_params,
                )
                self.llm = self.conversation_llm
                if not config.google_api_key:
                    raise ValueError("Google API key required for smart endpointing")
                self.statement_llm = GoogleLLMService(
                    name="StatementJudger",
                    api_key=config.google_api_key,
                    model=config.classifier_model,
                    temperature=0.0,
                    system_instruction=CLASSIFIER_SYSTEM_INSTRUCTION,
                )
            case _:
                raise ValueError(f"Invalid LLM provider: {config.llm_provider}")

        # Initialize context
        self.context = OpenAILLMContext(messages=system_messages)
        self.context_aggregator = self.conversation_llm.create_context_aggregator(self.context)

        # DISABLE MUTE FILTER COMPLETELY
        self.stt_mute_filter = None

        logger.debug(f"Initialized bot with config: {config}")

        # Initialize transport params
        self.transport_params = DailyParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            audio_in_channels=1,
            vad_enabled=True,
            transcription_enabled=False,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.7,
                    start_secs=0.3,
                    stop_secs=0.8,
                    min_volume=0.4
                )
            ),
        )

        # Initialize RTVI
        self.rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        # Initialize smart endpointing
        self.notifier = EventNotifier()
        self.statement_judge_context_filter = StatementJudgeContextFilter(notifier=self.notifier)
        self.completeness_check = CompletenessCheck(notifier=self.notifier)
        self.output_gate = OutputGate(notifier=self.notifier, start_open=True)

        async def user_idle_notifier(frame):
            await self.notifier.notify()

        self.user_idle = UserIdleProcessor(callback=user_idle_notifier, timeout=5.0)

        # These will be set up when needed
        self.transport: Optional[DailyTransport] = None
        self.task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None

    async def setup_transport(self, url: str, token: str):
        """Set up the transport with the given URL and token."""
        self.transport = DailyTransport(url, token, self.config.bot_name, self.transport_params)

        @self.transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            if self.task:
                await self.task.cancel()

        @self.transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_audio(participant["id"])
            await self._handle_first_participant()

        @self.transport.event_handler("on_app_message")
        async def on_app_message(transport, message, sender):
            if "message" not in message or not self.task:
                return

            await self.task.queue_frames(
                [
                    UserStartedSpeakingFrame(),
                    TranscriptionFrame(
                        user_id=sender,
                        timestamp=0, # type: ignore
                        text=message["message"]
                    ),
                    UserStoppedSpeakingFrame(),
                ]
            )

    def create_pipeline(self):
        """Create the processing pipeline - NO MUTE FILTER."""
        if not self.transport:
            raise RuntimeError("Transport must be set up before creating pipeline")

        async def block_user_stopped_speaking(frame):
            return not isinstance(frame, UserStoppedSpeakingFrame)

        async def pass_only_llm_trigger_frames(frame):
            return (
                isinstance(frame, OpenAILLMContextFrame)
                or isinstance(frame, StartInterruptionFrame)
                or isinstance(frame, BotInterruptionFrame)
                or isinstance(frame, FunctionCallInProgressFrame)
                or isinstance(frame, FunctionCallResultFrame)
            )

        async def discard_all(frame):
            return False

        # Build pipeline WITHOUT STT mute filter
        pipeline = Pipeline(
            [
                processor
                for processor in [
                    self.rtvi,
                    self.transport.input(),
                    self.stt,  # Audio goes DIRECTLY to Whisper
                    self.context_aggregator.user(),
                    ParallelPipeline(
                        [
                            FunctionFilter(filter=block_user_stopped_speaking),
                        ],
                        [
                            self.statement_judge_context_filter,
                            self.statement_llm,
                            self.completeness_check,
                            FunctionFilter(filter=discard_all),
                        ],
                        [
                            FunctionFilter(filter=pass_only_llm_trigger_frames),
                            self.conversation_llm,
                            self.output_gate,
                        ],
                    ),
                    self.tts,
                    self.user_idle,
                    self.transport.output(),
                    self.context_aggregator.assistant(),
                ]
                if processor is not None
            ]
        )

        self.task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        self.runner = PipelineRunner()
        
    async def start(self):
        """Start the bot's main task."""
        if not self.runner or not self.task:
            raise RuntimeError("Bot not properly initialized. Call create_pipeline first.")
        await self.runner.run(self.task)

    async def cleanup(self):
        """Clean up resources."""
        if self.runner:
            await self.runner.stop_when_done()
        if self.transport:
            try:
                await self.transport.stop() # type: ignore
            except Exception as e:
                logger.warning(f"Transport cleanup error: {e}")

    @abstractmethod
    async def _handle_first_participant(self):
        """Override in subclass to handle the first participant joining."""
        pass