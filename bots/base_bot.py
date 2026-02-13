"""Base bot framework - LIVEKIT VERSION (ENHANCED IDLE WITH PAUSE/RESUME)."""

import asyncio
import json
from abc import ABC, abstractmethod
import os
from typing import Dict, Optional, List, Callable, Any, Protocol, cast
from datetime import date, datetime
import aiohttp
from openai import AsyncOpenAI
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.pipeline import Pipeline
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIProcessor
from pipecat.services.deepseek.llm import DeepSeekLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams

# ── STT: Deepgram ──────────────────────────────────────────────────────────────
from pipecat.services.deepgram.stt import DeepgramSTTService
from deepgram import LiveOptions

# ── TTS: Google ────────────────────────────────────────────────────────────────
from pipecat.services.google.tts import GoogleTTSService

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import DataFrame
from pipecat.transcriptions.language import Language
from pipecat.frames.frames import (
    StartInterruptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TranscriptionFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TextFrame,
)
import httpx

from db.supabase import supabase
from .smart_endpointing import AssistantMemoryWriter
from pipecat.processors.user_idle_processor import UserIdleProcessor
from openai.types.chat import ChatCompletionMessageParam
from loguru import logger


class LLMServiceProtocol(Protocol):
    async def complete(self, context: OpenAILLMContext) -> str: ...


class BaseBot(ABC):

    def __init__(self, config, system_messages: Optional[List[ChatCompletionMessageParam]] = None):
        self.config = config
        self.user_id: str | None = None
        self.session_id = None
        self.api_base_url = getattr(config, 'api_base_url', 'http://localhost:8000')
        self._summary_flushed = False
        self.pause_trigger_source: Optional[str] = None

        # ── STT: Deepgram ──────────────────────────────────────────────────────
        deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        if not deepgram_api_key:
            raise ValueError("DEEPGRAM_API_KEY must be set in environment")

        self.stt = DeepgramSTTService(
            api_key=deepgram_api_key,
            live_options=LiveOptions(
                model="nova-3",
                language="en-US",
                smart_format=True,
                punctuate=True,
                filler_words=False,
                interim_results=True,
                utterance_end_ms="1000",
                vad_events=True,
            ),
        )

        # ── TTS: Google ────────────────────────────────────────────────────────
        google_credentials_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
        if not google_credentials_json:
            raise ValueError("GOOGLE_CREDENTIALS_JSON must be set in environment (service account JSON)")

        self.tts = GoogleTTSService(
            credentials=google_credentials_json,  # Full JSON string, not API key
            voice_id=getattr(config, "google_tts_voice", "en-US-Journey-F"),
            params=GoogleTTSService.InputParams(
                language="en-US",
                speaking_rate=getattr(config, "google_tts_speed", 1.0),
            ),
        )

        # ── LLM ───────────────────────────────────────────────────────────────
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

            case "google":
                if not config.google_api_key:
                    raise ValueError("Google API key is required")
                self.conversation_llm = GoogleLLMService(
                    api_key=config.google_api_key,
                    model=config.google_model,
                    params=config.google_params,
                    system_instruction="You are a helpful voice assistant.",
                )

            case "openai":
                if not config.openai_api_key:
                    raise ValueError("OpenAI API key is required")
                self.conversation_llm = OpenAILLMService(
                    api_key=config.openai_api_key,
                    model=config.openai_model,
                    params=config.openai_params,
                )

            case _:
                raise ValueError(f"Invalid LLM provider: {config.llm_provider}")

        self.context = OpenAILLMContext(messages=system_messages or [])
        self.context_aggregator = self.conversation_llm.create_context_aggregator(self.context)

        self.assistant_memory_writer = AssistantMemoryWriter(self.context_aggregator)

        self.transport_params = LiveKitParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.7,
                    start_secs=0.2,
                    stop_secs=0.7,
                    min_volume=0.4,
                )
            ),
        )

        self.rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        self.user_idle = UserIdleProcessor(
            callback=self._handle_user_idle,
            timeout=40.0,
        )

        self.transport: Optional[LiveKitTransport] = None
        self.task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None

        self.is_idle = False

    async def setup_transport(self, url: str, token: str, room_name: str):

        url = url.rstrip('/')
        if '/rtc' in url:
            logger.warning("URL contains '/rtc' - consider using base URL only (e.g., wss://your-server.com)")

        self.transport = LiveKitTransport(url, token, self.config.bot_name, self.transport_params)
        self.session_id = room_name

        @self.transport.event_handler("on_participant_disconnected")
        async def on_participant_disconnected(transport, participant, reason=None):
            participant_id = participant if participant else 'unknown'
            logger.info(f"Participant disconnected: {participant_id}")
            await self.flush_daily_summary(reason="participant_disconnected")
            if self.task:
                await self.task.cancel()

        @self.transport.event_handler("on_participant_connected")
        async def on_participant_connected(transport, participant):
            await transport.capture_participant_audio(participant.sid)
            await self._handle_first_participant()

        @self.transport.event_handler("on_data_received")
        async def on_data_received(transport, data, participant):
            try:
                message_str = data.decode("utf-8")
                payload = json.loads(message_str)

                if payload.get("type") == "session_control":
                    action = payload.get("action")

                    if action == "pause" and not self.is_idle:
                        await self.pause_session(trigger_source="ui")
                    elif action == "resume" and self.is_idle:
                        await self.resume_session(trigger_source="ui")
                    return

                text_message = payload.get("message", "")
                user_id = participant.identity if participant else "unknown"
                normalized_msg = text_message.strip().lower()

                if self.is_idle and self.pause_trigger_source == "idle":
                    if normalized_msg in ["hello", "hi", "hey", "ok", "continue"]:
                        logger.info("Voice resume detected", extra={"user_id": user_id})
                        await self.resume_session(trigger_source="voice")
                        return

            except Exception:
                logger.exception("Error handling data message")

    def create_pipeline(self):
        if not self.transport:
            raise RuntimeError("Transport must be set up before creating pipeline")

        pipeline = Pipeline([
            self.rtvi,
            self.transport.input(),
            self.stt,
            self.context_aggregator.user(),
            self.conversation_llm,
            self.assistant_memory_writer,
            self.tts,
            self.user_idle,
            self.transport.output(),
        ])

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
        if not self.runner or not self.task:
            raise RuntimeError("Bot not properly initialized. Call create_pipeline first.")
        await self.runner.run(self.task)

    async def cleanup(self):
        """Clean up resources and finalize session."""
        if getattr(self, "_summary_flushed", False):
            return

        try:
            await self.flush_daily_summary(reason="cleanup")
        except Exception as e:
            logger.error(f"Error during session finalization: {e}")

        if self.runner:
            await self.runner.stop_when_done()

        if self.transport:
            try:
                await self.transport.stop()  # type: ignore
            except Exception as e:
                logger.warning(f"Transport cleanup error: {e}")

    async def _handle_user_idle(self, processor: Optional[UserIdleProcessor] = None):

        if self.is_idle:
            return

        self.is_idle = True

        try:
            await self._notify_frontend_state_change("idle")

            pause_message = (
                "It seems like you may have stepped away. I've paused our session "
                "so you can continue right where you left off. Just say \"hello\" or "
                "click play when you're back."
            )

            if self.task:
                await self.task.queue_frames([
                    BotStartedSpeakingFrame(),
                    TextFrame(pause_message),
                    BotStoppedSpeakingFrame(),
                ])

            save_method: Optional[Callable] = getattr(self, 'save_session_data', None)
            if callable(save_method):
                save_method()

        except Exception:
            self.is_idle = False

    async def pause_session(self, trigger_source: str = "idle"):
        if self.is_idle:
            return

        self.is_idle = True
        self.pause_trigger_source = trigger_source

        try:
            if hasattr(self.user_idle, 'stop'):
                self.user_idle.stop()  # type: ignore

            if self.task:
                await self.task.queue_frames([StartInterruptionFrame()])

            await self._notify_frontend_state_change("idle")

            if trigger_source == "idle":
                pause_message = (
                    "It seems like you may have stepped away. I've paused our session "
                    "so you can continue right where you left off. Just say \"hello\" or "
                    "click play when you're back."
                )

                if self.task:
                    await self.task.queue_frames([
                        BotStartedSpeakingFrame(),
                        TextFrame(pause_message),
                        BotStoppedSpeakingFrame(),
                    ])

            save_method = getattr(self, 'save_session_data', None)
            if callable(save_method):
                save_method()

        except Exception as e:
            logger.error(f"Error during pause: {e}")
            self.is_idle = False
            self.pause_trigger_source = None

    async def resume_session(self, trigger_source: str = "voice"):

        if not self.is_idle:
            logger.debug("Session is not idle, ignoring resume request")
            return

        if self.pause_trigger_source == "ui" and trigger_source != "ui":
            logger.warning(
                "UI-paused session attempted voice resume",
                extra={"attempted_source": trigger_source},
            )
            return

        logger.info(
            "Session resumed",
            extra={"session_id": self.session_id, "trigger_source": trigger_source},
        )

        self.is_idle = False
        self.pause_trigger_source = None

        try:
            if hasattr(self.user_idle, 'start'):
                self.user_idle.start()  # type: ignore

            await self._notify_frontend_state_change("active")

            if trigger_source == "voice":
                welcome_back = "Welcome back! Let's continue."
                if self.task:
                    await self.task.queue_frames([
                        BotStartedSpeakingFrame(),
                        TextFrame(welcome_back),
                        BotStoppedSpeakingFrame(),
                    ])

            logger.info("Session resumed")

        except Exception as e:
            logger.error(f"Error during resume: {e}")
            self.is_idle = True

    async def _notify_frontend_state_change(self, state: str):
        try:
            room = self.transport._client._room  # type: ignore

            if not room or not hasattr(room, 'local_participant'):
                logger.warning("LiveKit room not accessible yet")
                return

            payload = json.dumps({
                "type": "session_state_change",
                "session_id": self.session_id,
                "state": state,
            }).encode("utf-8")

            await room.local_participant.publish_data(payload, reliable=True)  # type: ignore
            logger.info("Session state sent via LiveKit", extra={"state": state})

        except AttributeError as e:
            logger.error(f"LiveKit room attribute error: {e}")
        except Exception:
            logger.exception("Failed to send session state")

    def _extract_user_text(self) -> str:
        texts: list[str] = []

        for msg in self.context.messages:
            if msg.get("role") != "user":
                continue

            content = msg.get("content")

            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text")
                        if isinstance(text, str):
                            texts.append(text)

        return " ".join(texts).strip()

    async def _get_existing_daily_summary(self):

        result = (
            supabase
            .table("therapy_sessions")
            .select("title, summary, session_intensity")
            .eq("user_id", self.user_id)
            .eq("date", date.today().isoformat())
            .limit(1)
            .execute()
        )

        if not result.data:
            return None

        row = cast(dict[str, Any], result.data[0])

        title = row.get("title")
        summary = row.get("summary")
        intensity = row.get("session_intensity")

        if not title or not summary or intensity is None:
            logger.info(
                "Found incomplete daily summary row - will regenerate",
                extra={
                    "has_title": bool(title),
                    "has_summary": bool(summary),
                    "has_intensity": intensity is not None,
                },
            )
            return None

        return (title, summary, intensity)

    async def _generate_safe_summary(self):
        existing = await self._get_existing_daily_summary()
        user_text = self._extract_user_text()

        if len(user_text) < 10:
            return existing or (
                "Daily Emotional Reflection",
                "Today's check-in was brief. Emotional reflection will continue later.",
                3,
            )

        try:
            if existing:
                existing_title, existing_summary, existing_intensity = existing

                prompt = (
                    "You are refining a daily therapy summary.\n\n"
                    "Existing summary:\n"
                    f"Title: {existing_title}\n"
                    f"Summary: {existing_summary}\n"
                    f"Intensity: {existing_intensity}\n\n"
                    "Based on the NEW conversation content below, refine or extend the summary.\n"
                    "Do NOT contradict the existing summary.\n"
                    "If no meaningful new information exists, return the existing summary unchanged.\n"
                    "Write from the user's perspective, using 'you' to describe their experience.\n\n"
                    "Respond ONLY in JSON:\n"
                    '{ "title": "...", "summary": "...", "session_intensity": number }'
                )
            else:
                prompt = (
                    "Summarize today's therapy interaction from the user's perspective.\n"
                    "Use second-person ('you') to help them reflect on their own experience.\n"
                    "Examples: 'You explored feelings of...', 'You discussed challenges with...', 'You reflected on...'\n"
                    "Focus on what they shared, discovered, or worked through.\n\n"
                    "Then generate:\n"
                    "- a short session title (3–6 words, neutral, no advice)\n"
                    "- ONE emotional intensity score (1–10) based on this scale:\n"
                    "  1 = Too much (crisis level, overwhelming distress)\n"
                    "  2 = Anxious (high anxiety, very worried)\n"
                    "  3 = Overwhelmed (struggling to cope)\n"
                    "  4 = Strained (under significant pressure)\n"
                    "  5 = Heavy (burdened, weighed down)\n"
                    "  6 = Uneasy (uncomfortable, somewhat troubled)\n"
                    "  7 = Neutral (neither good nor bad, stable)\n"
                    "  8 = Light (mostly positive, manageable)\n"
                    "  9 = Okay (doing well, comfortable)\n"
                    "  10 = At ease (calm, peaceful, grounded)\n\n"
                    "Choose the intensity that best matches the overall emotional tone of the session.\n\n"
                    "Respond ONLY in JSON:\n"
                    '{ "title": "...", "summary": "...", "session_intensity": number }'
                )

            messages = cast(
                list[ChatCompletionMessageParam],
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_text},
                ],
            )

            client = AsyncOpenAI(
                api_key=self.config.deepseek_api_key,
                base_url="https://api.deepseek.com/v1",
            )

            response = await client.chat.completions.create(
                model=self.config.deepseek_model,
                messages=messages,
                temperature=0.3,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty LLM response")

            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()

            result = json.loads(cleaned)

            title = str(result.get("title", "")).strip() or "Daily Emotional Reflection"
            summary = str(result.get("summary", "")).strip()
            intensity = int(result.get("session_intensity", 3))
            intensity = max(1, min(10, intensity))

            if not summary:
                raise ValueError("Empty summary")

            return title, summary, intensity

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")

            if existing:
                return existing

            return (
                "Daily Emotional Reflection",
                "User checked in today. Emotional reflection will continue later.",
                3,
            )



    async def flush_daily_summary(self, reason: str):
        try:
            existing = await self._get_existing_daily_summary()

            if existing and getattr(self, "_summary_flushed", False):
                return

            title, summary, intensity = await self._generate_safe_summary()

            if not summary or not title or intensity is None:
                title = title or "Daily Emotional Reflection"
                summary = summary or (
                    "Today's check-in was brief. Emotional reflection will continue later."
                )
                intensity = intensity if intensity is not None else 3

            await self._send_session_end(
                title=title,
                summary=summary,
                intensity=intensity,
            )

            self._summary_flushed = True

        except Exception:
            logger.exception("Session end failed; summary NOT flushed — will retry later")

    async def _send_session_end(self, *, title: str, summary: str, intensity: int):

        if not getattr(self, "user_id", None):
            return False

        backend_url = os.getenv("BACKEND_API_URL")
        endpoint = f"{backend_url}/session/end"

        payload = {
            "user_id": self.user_id,
            "title": title,
            "summary": summary,
            "session_intensity": intensity,
        }

        logger.info(f"📤 Sending session end to: {endpoint}")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
            return True

        except httpx.HTTPStatusError:
            return False
        except httpx.ConnectError:
            return False
        except Exception:
            return False

    async def _handle_first_participant(self):
        # No TTS warmup needed — Google TTS is a managed API, no cold start
        if not self.task:
            logger.warning("⏳ Pipeline not ready yet - waiting 1 second...")
            await asyncio.sleep(1.0)
            if not self.task:
                logger.error("❌ Cannot send greeting: pipeline task is still not ready")
                return

        greeting = "Hello there! I'm here to listen whenever you're ready to share."

        try:
            await self.task.queue_frames([
                BotStartedSpeakingFrame(),
                TextFrame(greeting),
                BotStoppedSpeakingFrame(),
            ])
        except Exception as e:
            logger.error(f"❌ Failed to send greeting: {e}")