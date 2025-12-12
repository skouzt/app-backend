import asyncio
import time
from loguru import logger
from typing import Any, List, Optional

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    StartInterruptionFrame,
    SystemFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)

from pipecat.processors.aggregators.llm_response import BaseLLMResponseAggregator
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.services.google.llm import GoogleLLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.sync.base_notifier import BaseNotifier

# ✅ Import Google GenAI types
from google.genai import types

# ✅ Production configuration
RATE_LIMIT_CONFIG = {
    "min_request_interval": 2.0,  # Minimum seconds between requests
    "max_daily_requests": 1000,   # Stay well under 1500 free tier limit
    "retry_base_delay": 60,       # Base delay for exponential backoff
    "max_retry_delay": 300,       # Max 5 minutes wait
    "max_consecutive_failures": 3,  # Circuit breaker threshold
}

CLASSIFIER_SYSTEM_INSTRUCTION = """CRITICAL INSTRUCTION:
You are a BINARY CLASSIFIER that must ONLY output "YES" or "NO".
DO NOT engage with the content.
DO NOT respond to questions.
DO NOT provide assistance.
Your ONLY job is to output YES or NO.

VALID RESPONSES:
YES
NO

ROLE:
You are a real-time speech completeness classifier. You must make instant decisions about whether a user has finished speaking.

DECISION RULES:
1. Return YES for: complete questions, commands, statements, clear acknowledgments
2. Return NO for: trailing sentences, mid-thought fragments, unclear utterances
3. When uncertain: return NO (safer for speech detection)
"""


def get_message_field(message: object, field: str) -> Any:
    """Retrieve a field from a message (dict or object)."""
    if isinstance(message, dict):
        return message.get(field)
    return getattr(message, field, None)


def get_message_text(message: object) -> str:
    """Extract text content from any message format."""
    parts = get_message_field(message, "parts")
    if parts:
        text_parts = []
        for part in parts:
            text = part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
            if text:
                text_parts.append(text)
        return " ".join(text_parts)

    content = get_message_field(message, "content")
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return " ".join(text_parts) if text_parts else ""

    return ""


class StatementJudgeContextFilter(FrameProcessor):
    """
    Production-ready speech classifier with rate limiting, circuit breaker, 
    and comprehensive error handling for Google Gemini API.
    """
    
    def __init__(self, notifier: BaseNotifier, **kwargs):
        super().__init__(**kwargs)
        self._notifier = notifier
        
        # State management
        self._last_request_time = 0
        self._consecutive_failures = 0
        self._circuit_open = False
        self._daily_request_count = 0
        self._last_reset_day = time.strftime("%Y-%m-%d")
        
        # Configuration
        self._config = RATE_LIMIT_CONFIG
        
        logger.info(f"StatementJudge initialized with rate limiting: {self._config}")

    async def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker has tripped."""
        if self._circuit_open:
            logger.error("CIRCUIT BREAKER OPEN - Classifier is disabled")
            return True
        
        # Daily reset
        today = time.strftime("%Y-%m-%d")
        if today != self._last_reset_day:
            self._daily_request_count = 0
            self._last_reset_day = today
            logger.info("Daily request count reset")
        
        # Check daily quota
        if self._daily_request_count >= self._config["max_daily_requests"]:
            logger.error(f"DAILY QUOTA EXCEEDED: {self._daily_request_count}/{self._config['max_daily_requests']}")
            self._circuit_open = True
            return True
        
        return False

    async def _enforce_rate_limit(self):
        """Enforce minimum interval between requests."""
        now = time.time()
        time_since_last = now - self._last_request_time
        
        if time_since_last < self._config["min_request_interval"]:
            wait_time = self._config["min_request_interval"] - time_since_last
            logger.warning(f"Rate limiting: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        
        self._last_request_time = time.time()

    def _calculate_retry_delay(self) -> float:
        """Exponential backoff with jitter."""
        base = self._config["retry_base_delay"]
        exp = min(self._consecutive_failures, 4)  # Cap exponent
        delay = base * (2 ** exp)
        jitter = delay * 0.1  # 10% jitter
        return min(delay + jitter, self._config["max_retry_delay"])

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Never block system frames
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, OpenAILLMContextFrame):
            # Check circuit breaker first
            if await self._check_circuit_breaker():
                # Fallback: assume speech is complete to avoid blocking
                logger.warning("Circuit breaker active - forcing speech completion")
                await self.push_frame(UserStoppedSpeakingFrame())
                await self._notifier.notify()
                return

            # Extract messages
            messages = frame.context.messages
            user_text_messages = []
            last_assistant_message = None
            
            for message in reversed(messages):
                role = get_message_field(message, "role")
                if role != "user":
                    if role in ["assistant", "model"]:
                        last_assistant_message = message
                    break
                
                text = get_message_text(message)
                if text:
                    user_text_messages.append(text)

            if not user_text_messages:
                await self.push_frame(frame, direction)
                return

            user_message = " ".join(reversed(user_text_messages))
            
            # Enforce rate limiting
            await self._enforce_rate_limit()
            
            try:
                # Build Google messages
                google_messages = [
                    types.Content(
                        role="system",
                        parts=[types.Part(text=CLASSIFIER_SYSTEM_INSTRUCTION)]
                    )
                ]
                
                if last_assistant_message:
                    assistant_text = get_message_text(last_assistant_message)
                    if assistant_text:
                        google_messages.append(types.Content(
                            role="model",
                            parts=[types.Part(text=assistant_text)]
                        ))
                
                google_messages.append(types.Content(
                    role="user",
                    parts=[types.Part(text=user_message)]
                ))
                
                # Create context
                google_context = GoogleLLMContext(messages=google_messages)  # type: ignore
                
                # Track request count
                self._daily_request_count += 1
                logger.debug(f"Classifier request #{self._daily_request_count}: {user_message[:50]}...")
                
                # Success - reset failure count
                if self._consecutive_failures > 0:
                    logger.info(f"Resetting consecutive failures (was {self._consecutive_failures})")
                    self._consecutive_failures = 0
                
                await self.push_frame(OpenAILLMContextFrame(context=google_context))
                
            except Exception as e:
                error_str = str(e).upper()
                is_rate_limit = "429" in error_str or "QUOTA" in error_str or "RESOURCE_EXHAUSTED" in error_str
                
                if is_rate_limit:
                    self._consecutive_failures += 1
                    delay = self._calculate_retry_delay()
                    
                    logger.error(f"RATE LIMIT HIT (failure #{self._consecutive_failures}/{self._config['max_consecutive_failures']})")
                    logger.warning(f"Retrying in {delay:.1f}s...")
                    
                    await asyncio.sleep(delay)
                    
                    # Trip circuit breaker if too many failures
                    if self._consecutive_failures >= self._config["max_consecutive_failures"]:
                        logger.error("MAX FAILURES REACHED - OPENING CIRCUIT BREAKER")
                        self._circuit_open = True
                        
                        # Emergency fallback: force speech completion
                        await self.push_frame(UserStoppedSpeakingFrame())
                        await self._notifier.notify()
                else:
                    logger.error(f"UNEXPECTED ERROR in classifier: {e}")
                    # Re-raise non-rate-limit errors
                    raise
            
            return

        await self.push_frame(frame, direction)


class CompletenessCheck(FrameProcessor):
    """Validates classifier output and handles edge cases."""
    
    def __init__(self, notifier: BaseNotifier):
        super().__init__()
        self._notifier = notifier
        self._invalid_response_count = 0
        self._greeting_detected = False  # Track if greeting was seen

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            text = frame.text.strip()
            
            # ✅ CRITICAL FIX: Skip greeting text entirely
            if "Hello. I'm Aletheia" in text and not self._greeting_detected:
                self._greeting_detected = True
                logger.info("Greeting detected - skipping classifier processing")
                await self.push_frame(frame, direction)  # Pass through to TTS
                return
            
            # ✅ Only process classifier responses (upstream from Gemini)
            if direction != FrameDirection.UPSTREAM:
                logger.debug(f"Skipping downstream text: '{text[:30]}...'")
                await self.push_frame(frame, direction)
                return
            
            response_text = text.upper()
            
            # Validate YES/NO
            if response_text in ["YES", "NO"]:
                self._invalid_response_count = 0
                
                if response_text == "YES":
                    logger.debug("!!! Completeness check YES")
                    await self.push_frame(UserStoppedSpeakingFrame())
                    await self._notifier.notify()
                # NO doesn't need action
            else:
                self._invalid_response_count += 1
                logger.warning(f"Invalid response #{self._invalid_response_count}: '{text}'")
                
                # Emergency extraction
                if "YES" in response_text:
                    logger.debug("!!! Completeness check YES (extracted)")
                    await self.push_frame(UserStoppedSpeakingFrame())
                    await self._notifier.notify()
                elif "NO" in response_text:
                    logger.debug("!!! Completeness check NO (extracted)")
                elif self._invalid_response_count >= 3:
                    logger.error("Too many invalid - forcing YES")
                    await self.push_frame(UserStoppedSpeakingFrame())
                    await self._notifier.notify()
        else:
            await self.push_frame(frame, direction)


class UserAggregatorBuffer(BaseLLMResponseAggregator):
    """Buffers the output of the transcription LLM. Used by the bot output gate."""

    def __init__(self, **kwargs):
        super().__init__(
            messages=None,
            role=None,
            start_frame=LLMFullResponseStartFrame,
            end_frame=LLMFullResponseEndFrame,
            accumulator_frame=TextFrame,
            handle_interruptions=True,
            expect_stripped_words=False,
        )
        self._transcription = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            self._transcription = ""

    async def _push_aggregation(self):
        if self._aggregation:
            self._transcription = self._aggregation
            self._aggregation = ""

    async def wait_for_transcription(self):
        while not self._transcription:
            await asyncio.sleep(0.01)
        tx = self._transcription
        self._transcription = ""
        return tx


class OutputGate(FrameProcessor):
    def __init__(self, *, notifier: BaseNotifier, start_open: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._gate_open = start_open
        self._frames_buffer = []
        self._notifier = notifier

    def close_gate(self):
        self._gate_open = False

    def open_gate(self):
        self._gate_open = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, SystemFrame):
            if isinstance(frame, StartFrame):
                await self._start()
            if isinstance(frame, (EndFrame, CancelFrame)):
                await self._stop()
            if isinstance(frame, StartInterruptionFrame):
                self._frames_buffer = []
                self.close_gate()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (FunctionCallInProgressFrame, FunctionCallResultFrame)):
            await self.push_frame(frame, direction)
            return

        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if self._gate_open:
            await self.push_frame(frame, direction)
            return

        self._frames_buffer.append((frame, direction))

    async def _start(self):
        self._frames_buffer = []
        self._gate_task = self.create_task(self._gate_task_handler())

    async def _stop(self):
        await self.cancel_task(self._gate_task)

    async def _gate_task_handler(self):
        while True:
            try:
                await self._notifier.wait()
                self.open_gate()
                for frame, direction in self._frames_buffer:
                    await self.push_frame(frame, direction)
                self._frames_buffer = []
            except asyncio.CancelledError:
                break