"""Base bot framework - LIVEKIT VERSION (ENHANCED IDLE WITH PAUSE/RESUME)."""

import asyncio
import json
from abc import ABC, abstractmethod
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
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams
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
    """Protocol defining the interface for LLM services that support direct completion."""
    async def complete(self, context: OpenAILLMContext) -> str: ...
class BaseBot(ABC):
 
    def __init__(self, config, system_messages: Optional[List[ChatCompletionMessageParam]] = None):
        """Initialize bot with core services and enhanced idle handling."""
        self.config = config
        self.user_id: str | None = None
        self.session_id = None  # Will be set when transport is initialized
        self.api_base_url = getattr(config, 'api_base_url', 'http://localhost:8000')
        self._summary_flushed = False
        # Initialize STT service
        # ✅ NEW: Track who paused the session
        self.pause_trigger_source: Optional[str] = None  # "idle" or "ui"
        logger.info("Initializing Whisper STT service")
        self.stt = WhisperSTTService(
            model="base",
            device="cpu",
            compute_type="int8",
            language=Language.EN,
            no_speech_prob=0.6,
            vad_filter=False,
        )
        logger.success("Whisper STT initialized")

        # Initialize TTS service
        logger.info(f"Initializing TTS service: {config.tts_provider}")
        match config.tts_provider:                
            case "kokoro_fastapi":
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
            
            

        # Initialize LLM service only (NO CLASSIFIER)
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
                logger.success("DeepSeek LLM initialized")
                
            case "google":
                if not config.google_api_key:
                    raise ValueError("Google API key is required")
                self.conversation_llm = GoogleLLMService(
                    api_key=config.google_api_key,
                    model=config.google_model,
                    params=config.google_params,
                    system_instruction="You are a helpful voice assistant.",
                )
                logger.success("Google LLM initialized")
                
            case "openai":
                if not config.openai_api_key:
                    raise ValueError("OpenAI API key is required")
                self.conversation_llm = OpenAILLMService(
                    api_key=config.openai_api_key,
                    model=config.openai_model,
                    params=config.openai_params,
                )
                logger.success("OpenAI LLM initialized")
                
            case _:
                raise ValueError(f"Invalid LLM provider: {config.llm_provider}")

        # Initialize context (NO CLASSIFIER)
        self.context = OpenAILLMContext(messages=system_messages or [])
        self.context_aggregator = self.conversation_llm.create_context_aggregator(self.context)

        logger.info("Initialized bot with simplified pipeline (NO classifier)")
        self.assistant_memory_writer = AssistantMemoryWriter(self.context_aggregator)

        self.transport_params = LiveKitParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.7,
                    start_secs=0.3,
                    stop_secs=0.7,
                    min_volume=0.4
                )
            ),
        )

        # Initialize RTVI
        self.rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        # Initialize idle processor with 55-second timeout
        self.user_idle = UserIdleProcessor(
            callback=self._handle_user_idle,  
            timeout=40.0
        )
        logger.info("User idle processor initialized (55s timeout)")

        # Initialize transport and task (will be set up later)
        self.transport: Optional[LiveKitTransport] = None
        self.task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None

        # ✅ PAUSE/RESUME STATE MANAGEMENT
        self.is_idle = False
        logger.info("Pause/resume state management initialized")

    async def setup_transport(self, url: str, token: str, room_name: str):
        """
        Set up the LiveKit transport.
        
        IMPORTANT: For LiveKit, `url` should be the base WebSocket URL of your LiveKit server,
        e.g., "wss://your-livekit-server.com" (NO path, NO /rtc suffix)
        
        Args:
            url: Base WebSocket URL (wss://your-server.com)
            token: LiveKit access token for the bot
            room_name: Name of the room to join
        """
        url = url.rstrip('/')
        if '/rtc' in url:
            logger.warning("URL contains '/rtc' - consider using base URL only (e.g., wss://your-server.com)")
        
        self.transport = LiveKitTransport(url, token, self.config.bot_name, self.transport_params)
        self.session_id = room_name  # Use room_name as session_id

        @self.transport.event_handler("on_participant_disconnected")
        async def on_participant_disconnected(transport, participant, reason=None):
            """Handle participant disconnection"""
            participant_id = participant if participant else 'unknown'
            logger.info(f"Participant disconnected: {participant_id}")
            await self.flush_daily_summary(reason="participant_disconnected")
            if self.task:
                await self.task.cancel()

        @self.transport.event_handler("on_participant_connected")
        async def on_participant_connected(transport, participant):
            """Handle participant connection"""
            participant_id = participant.identity if participant else 'unknown'
            logger.info(f"Participant connected: {participant_id}")
            await transport.capture_participant_audio(participant.sid)
            await self._handle_first_participant()

        @self.transport.event_handler("on_data_received")
        async def on_data_received(transport, data, participant):
            """Handle incoming data messages."""
            try:
                message_str = data.decode("utf-8")
                payload = json.loads(message_str)
                
                # ✅ SAFE: Log event type only, not content
                payload_type = payload.get("type", "unknown")
                if payload_type == "session_control":
                    action = payload.get("action", "unknown")
                    logger.info(
                        "UI control received",
                        extra={"type": payload_type, "action": action}
                    )
                else:
                    # For message types, just acknowledge receipt
                    logger.info(
                        "UI data received",
                        extra={"type": payload_type, "has_message": "message" in payload}
                    )

                # ✅ STRUCTURED SESSION CONTROL
                if payload.get("type") == "session_control":
                    action = payload.get("action")

                    if action == "pause" and not self.is_idle:
                        logger.info("UI requested PAUSE")
                        await self.pause_session(trigger_source="ui")
                        
                    elif action == "resume" and self.is_idle:
                        logger.info("UI requested RESUME")
                        await self.resume_session(trigger_source="ui")
                        
                    return  # Stop here for control messages

                # 🧓 Extract message and user info for voice processing
                text_message = payload.get("message", "")
                user_id = participant.identity if participant else "unknown"
                normalized_msg = text_message.strip().lower()

                # Voice resume only works if paused by "idle"
                if self.is_idle and self.pause_trigger_source == "idle":
                    if normalized_msg in ["hello", "hi", "hey", "ok", "continue"]:
                        logger.info("Voice resume detected", extra={"user_id": user_id})
                        await self.resume_session(trigger_source="voice")
                        return

            except Exception:
                logger.exception("Error handling data message")

    def create_pipeline(self):
        """Create the processing pipeline - SIMPLIFIED (NO CLASSIFIER)."""
        if not self.transport:
            raise RuntimeError("Transport must be set up before creating pipeline")

        logger.info("Creating simplified pipeline (NO classifier)")

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
        """Start the bot's main task."""
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

        # Stop pipeline
        if self.runner:
            await self.runner.stop_when_done()

        if self.transport:
            try:
                await self.transport.stop() # type: ignore
            except Exception as e:
                logger.warning(f"Transport cleanup error: {e}")

    

    # ✅ PAUSE/RESUME METHODS (CORRECTED)
    async def _handle_user_idle(self, processor: Optional[UserIdleProcessor] = None):
        """
        PAUSE SESSION: Called after 55 seconds of user silence.
        Sets session to idle state and notifies frontend.
        """
        # Prevent re-triggering if already idle
        if self.is_idle:
            logger.debug("Already in idle state, ignoring duplicate idle trigger")
            return
            
        logger.warning(
            "User idle timeout triggered",
            extra={"session_id": self.session_id, "timeout_seconds": 55.0}
        )
        
        self.is_idle = True
        
        try:
            # Notify frontend to show "paused" UI state
            await self._notify_frontend_state_change("idle")
            
            pause_message = (
                "It seems like you may have stepped away. I've paused our session "
                "so you can continue right where you left off. Just say \"hello\" or "
                "click play when you're back."
            )
            
            logger.info("Sending pause notification")
            
            if self.task:
                await self.task.queue_frames([
                    BotStartedSpeakingFrame(),
                    TextFrame(pause_message),
                    BotStoppedSpeakingFrame(),
                ])
            
            # Save session state while idle (optional but recommended)
            save_method: Optional[Callable] = getattr(self, 'save_session_data', None)
            if callable(save_method):
                logger.info("Saving session snapshot during idle")
                save_method()
            
            logger.info("Session paused")
            
        except Exception as e:
            logger.error(f"Error during idle transition: {e}")
            self.is_idle = False
    async def pause_session(self, trigger_source: str = "idle"):
        """
        Pause the session: stop Pipecat + disable audio input + notify UI.
        """
        if self.is_idle:
            logger.debug("Already idle, ignoring pause request")
            return
            
        logger.info(
            "Session paused",
            extra={"session_id": self.session_id, "trigger_source": trigger_source}
        )
        self.is_idle = True
        self.pause_trigger_source = trigger_source
        
        try:
            # 1. **STOP AUDIO INPUT** (prevent new speech detection)
            if hasattr(self.user_idle, 'stop'):
                self.user_idle.stop()  # type: ignore # Stop idle timer
            
            # 2. **INTERRUPT** current pipeline (stop speaking)
            if self.task:
                await self.task.queue_frames([StartInterruptionFrame()])
            
            # 3. Notify frontend UI
            await self._notify_frontend_state_change("idle")
            
            # 4. Send voice notification (only for idle timeout)
            if trigger_source == "idle":
                pause_message = (
                    "It seems like you may have stepped away. I've paused our session "
                    "so you can continue right where you left off. Just say \"hello\" or "
                    "click play when you're back."
                )
                
                logger.info("Sending pause notification")
                if self.task:
                    await self.task.queue_frames([
                        BotStartedSpeakingFrame(),
                        TextFrame(pause_message),
                        BotStoppedSpeakingFrame(),
                    ])
            
            # 5. Save snapshot
            save_method = getattr(self, 'save_session_data', None)
            if callable(save_method):
                logger.info("Saving session snapshot during pause")
                save_method()
                
            logger.info("Session paused")
            
        except Exception as e:
            logger.error(f"Error during pause: {e}")
            self.is_idle = False
            self.pause_trigger_source = None

    async def resume_session(self, trigger_source: str = "voice"):
        """
        Resume the session — with source enforcement + restart audio input.
        """
        if not self.is_idle:
            logger.debug("Session is not idle, ignoring resume request")
            return
        
        # ⭐ ENFORCEMENT: Button-only resume for UI-paused sessions
        if self.pause_trigger_source == "ui" and trigger_source != "ui":
            logger.warning(
                "UI-paused session attempted voice resume",
                extra={"attempted_source": trigger_source}
            )
            return
            
        logger.info(
            "Session resumed",
            extra={"session_id": self.session_id, "trigger_source": trigger_source}
        )
        
        self.is_idle = False
        self.pause_trigger_source = None
        
        try:
            # 1. **RESTART AUDIO INPUT** (re-enable speech detection)
            if hasattr(self.user_idle, 'start'):
                self.user_idle.start()  # type: ignore # Restart idle timer
            
            # 2. Notify frontend
            await self._notify_frontend_state_change("active")
            
            # 3. Voice acknowledgment (only for voice resume)
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
        """
        Send session state via LiveKit data channel — using the correct internal API.
        """
        try:
            # ✅ Access the ACTUAL LiveKit room (not the wrapper)
            # Pipecat v0.0.x stores it in _client._room or _client.room
            room = self.transport._client._room  # type: ignore
            
            if not room or not hasattr(room, 'local_participant'):
                logger.warning("LiveKit room not accessible yet")
                return

            payload = json.dumps({
                "type": "session_state_change",
                "session_id": self.session_id,
                "state": state,
            }).encode("utf-8")
            
            # ✅ Use the native LiveKit publish_data method
            await room.local_participant.publish_data(payload, reliable=True)  # type: ignore
            
            logger.info("Session state sent via LiveKit", extra={"state": state})

        except AttributeError as e:
            logger.error(f"LiveKit room attribute error: {e}")
            # Debug: log available attributes
            if hasattr(self.transport, '_client'):  # type: ignore
                client = self.transport._client  # type: ignore
                logger.debug(f"Transport client attributes: {dir(client)}")
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
                # Handle OpenAI content-part format
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text")
                        if isinstance(text, str):
                            texts.append(text)

        return " ".join(texts).strip()

    async def _get_existing_daily_summary(self):
        """
        Returns existing (title, summary, intensity) for today ONLY if complete.
        Returns None if row doesn't exist OR has incomplete data.
        """
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
        
        # ✅ CRITICAL: Only return if we have COMPLETE data
        # An incomplete row should be treated as non-existent
        if not title or not summary or intensity is None:
            logger.info(
                "Found incomplete daily summary row - will regenerate",
                extra={
                    "has_title": bool(title),
                    "has_summary": bool(summary),
                    "has_intensity": intensity is not None
                }
            )
            return None
        
        return (title, summary, intensity)


    async def _generate_safe_summary(self):
        """
        Disconnect-safe summary generation.
        DeepSeek-only. Pragmatic. Stable.
        Returns: (title, summary, session_intensity)
        """

        # 1️⃣ Check DB first (source of truth)
        existing = await self._get_existing_daily_summary()
        user_text = self._extract_user_text()

        # 🔒 Minimum signal gate - but ONLY if existing is complete
        if len(user_text) < 10 and existing:
            # existing is guaranteed to be complete (not None) due to updated _get_existing_daily_summary
            return existing

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
                "If no meaningful new information exists, return the existing summary unchanged.\n\n"
                "Respond ONLY in JSON:\n"
                '{ "title": "...", "summary": "...", "session_intensity": number }'
            )
            else:
                prompt = (
                    "Summarize today's therapy interaction in a neutral, empathetic way.\n"
                    "Write in third person without using 'user' - refer to the person naturally.\n"
                    "Then generate:\n"
                    "- a short session title (3–6 words, neutral, no advice)\n"
                    "- ONE emotional intensity score (1–10)\n\n"
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
            
            # 🔒 Absolute fallback - but ONLY if existing is complete
            if existing:
                return existing

            # ✅ CRITICAL: Always return complete data, never None values
            return (
                "Daily Emotional Reflection",
                "User checked in today. Emotional reflection will continue later.",
                3,
            )


    async def flush_daily_summary(self, reason: str):
        """
        Finalizes and sends daily summary.
        Idempotent. DB-safe. Disconnect-safe.
        """

        if getattr(self, "_summary_flushed", False):
            return

        try:
            # Check for COMPLETE summary only
            existing = await self._get_existing_daily_summary()
            
            if existing:
                logger.info("Complete daily summary already exists; skipping flush")
                self._summary_flushed = True
                return

            # Generate summary (will reuse incomplete row data if exists in DB)
            title, summary, intensity = await self._generate_safe_summary()

            # ✅ Validate before sending
            if not summary or not title or intensity is None:
                logger.error(
                    "Generated incomplete summary - this should never happen!",
                    extra={
                        "has_title": bool(title),
                        "has_summary": bool(summary),
                        "has_intensity": intensity is not None
                    }
                )
                # Force valid defaults
                title = title or "Daily Emotional Reflection"
                summary = summary or "Today's check-in was brief. Emotional reflection will continue later."
                intensity = intensity if intensity is not None else 3

            await self._send_session_end(
                summary=summary,
                intensity=intensity,
                title=title,
            )

            self._summary_flushed = True

            logger.warning(
                "Daily summary sent",
                extra={
                    "user_id": self.user_id,
                    "summary_length": len(summary) if summary else 0,
                    "has_title": bool(title),
                    "has_intensity": intensity is not None
                }
            )

        except Exception:
            logger.exception("flush_daily_summary failed (non-fatal)")


    async def _send_session_end(self,*,title: str,summary: str,intensity: int,):
        # 🔒 Hard guard — fail early, not at FastAPI
        if not getattr(self, "user_id", None):
            logger.error("Cannot send session end: user_id is missing")
            return

        payload = {
            "user_id": self.user_id,
            "title": title,
            "summary": summary,
            "session_intensity": intensity,
        }

        # ✅ SAFE: Log the event, not the payload
        logger.warning(
            "Session end notification sent",
            extra={
                "user_id": self.user_id,
                "endpoint": "/session/end",
                "status": "sent"
            }
        )

        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(
                f"{self.api_base_url}/session/end",
                json=payload,
            )

            if resp.status_code != 200:
                # ✅ SAFE: Log sanitized error info
                logger.error(
                    "Session end request failed",
                    extra={
                        "user_id": self.user_id,
                        "status_code": resp.status_code,
                        "response_length": len(resp.text) if resp.text else 0
                    }
                )
            else:
                logger.info("Session end stored successfully")


    @abstractmethod
    async def _handle_first_participant(self):
        """Override in subclass to handle the first participant joining."""
        pass