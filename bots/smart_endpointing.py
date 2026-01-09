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
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    LLMMessagesFrame,
    TextFrame,
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
    Production-ready classifier with silent failover for rate limits.
    Logs are suppressed for expected errors to avoid spam.
    """
    
    def __init__(self, notifier: BaseNotifier, **kwargs):
        super().__init__(**kwargs)
        self._notifier = notifier
        
        # Rate limiting state
        self._last_request_time = 0
        self._min_request_interval = 3.0  # Increased to 3 seconds minimum
        
        # Circuit breaker state
        self._circuit_open = False
        self._consecutive_failures = 0
        self._max_failures = 3
        self._last_failure_time = 0
        self._cooldown_period = 120  # 2 minutes cooldown
        
        # Daily quota tracking
        self._daily_request_count = 0
        self._max_daily_requests = 800  # Stay well under 1500 limit
        
        logger.info("StatementJudge initialized with silent failover")

    async def _check_rate_limit(self) -> bool:
        """Check if we should make a request based on time limits."""
        now = time.time()
        time_since_last = now - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            logger.debug("Rate limit: waiting for minimum interval")
            return False
        
        return True

    async def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is closed (healthy)."""
        if not self._circuit_open:
            return True
        
        # Check cooldown
        time_since_failure = time.time() - self._last_failure_time
        if time_since_failure > self._cooldown_period:
            logger.info("Circuit breaker cooldown passed - retrying")
            self._circuit_open = False
            self._consecutive_failures = 0
            return True
        
        return False

    async def _trip_circuit_breaker(self, error_msg: str):
        """Trip breaker on repeated failures."""
        self._consecutive_failures += 1
        self._last_failure_time = time.time()
        
        if self._consecutive_failures >= self._max_failures:
            logger.warning("Circuit breaker opened due to repeated failures")
            logger.warning("Classifier will be disabled for 2 minutes")
            self._circuit_open = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, OpenAILLMContextFrame):
            # Check circuit breaker BEFORE processing
            if not await self._check_circuit_breaker():
                # Silent failover - no error logged
                logger.debug("Circuit breaker active - forcing completion")
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

            # ✅ SAFE: User message is extracted but NEVER logged
            user_message = " ".join(reversed(user_text_messages))
            
            # Check rate limit before API call
            if not await self._check_rate_limit():
                # Silent failover during rate limiting
                logger.debug("Rate limited - forcing completion")
                await self.push_frame(UserStoppedSpeakingFrame())
                await self._notifier.notify()
                return
            
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
                
                # ✅ SAFE: User message added to LLM request but not logged
                google_messages.append(types.Content(
                    role="user",
                    parts=[types.Part(text=user_message)]
                ))
                
                # Create context
                google_context = GoogleLLMContext(messages=google_messages)  # type: ignore
                self._daily_request_count += 1
                
                # Success - reset failures
                if self._consecutive_failures > 0:
                    logger.info(f"Failure count reset (was {self._consecutive_failures})")
                    self._consecutive_failures = 0
                
                await self.push_frame(OpenAILLMContextFrame(context=google_context))
                
            except Exception as e:
                error_str = str(e).upper()
                is_rate_limit = any(x in error_str for x in ["429", "QUOTA", "RESOURCE_EXHAUSTED"])
                
                if is_rate_limit:
                    # Silent failure for expected rate limits
                    logger.debug(f"Rate limit hit (failure #{self._consecutive_failures + 1})")
                    await self._trip_circuit_breaker(str(e))
                    
                    # Silent failover
                    logger.debug("Falling back to immediate completion")
                    await self.push_frame(UserStoppedSpeakingFrame())
                    await self._notifier.notify()
                else:
                    # ✅ SAFE: Log exception type only, not full message
                    logger.error(f"Non-rate-limit error in classifier: {type(e).__name__}")
                    logger.warning("Falling back to completion to avoid crash")
                    await self.push_frame(UserStoppedSpeakingFrame())
                    await self._notifier.notify()
            
            return

        await self.push_frame(frame, direction)

class CompletenessCheck(FrameProcessor):
    """Validates classifier output with noise suppression."""
    
    def __init__(self, notifier: BaseNotifier):
        super().__init__()
        self._notifier = notifier
        self._invalid_response_count = 0
        self._last_valid_response = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame) and direction == FrameDirection.UPSTREAM:
            try:
                text = frame.text.strip()
                
                # Skip empty responses (common during failures)
                if not text:
                    # ✅ REMOVED debug log in production
                    return
                
                response_text = text.upper()
                
                if response_text in ["YES", "NO"]:
                    self._invalid_response_count = 0
                    self._last_valid_response = response_text
                    
                    if response_text == "YES":
                        # ✅ REMOVED debug log in production
                        await self.push_frame(UserStoppedSpeakingFrame())
                        await self._notifier.notify()
                    # NO requires no action
                    
                else:
                    self._invalid_response_count += 1
                    
                    # ✅ SAFE: Only log count, not content
                    if self._invalid_response_count % 5 == 0:
                        logger.warning(f"Invalid classifier response count: {self._invalid_response_count}")
                    
                    # Emergency extraction
                    if "YES" in response_text:
                        # ✅ REMOVED debug log in production
                        await self.push_frame(UserStoppedSpeakingFrame())
                        await self._notifier.notify()
                    elif "NO" in response_text:
                        # ✅ REMOVED debug log in production
                        pass
                    elif self._invalid_response_count >= 10:
                        logger.error("Too many invalid classifier responses - forcing completion")
                        await self.push_frame(UserStoppedSpeakingFrame())
                        await self._notifier.notify()
                        
            except Exception as e:
                # ✅ SAFE: Log exception type only
                logger.error(f"Error in CompletenessCheck: {type(e).__name__}")
                # Never crash
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

from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.frames.frames import TextFrame

class AssistantMemoryWriter(FrameProcessor):
    def __init__(self, context_aggregator):
        super().__init__()
        self.context_aggregator = context_aggregator
        self._buffer = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        # Start of assistant response
        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = []

        # Collect streamed tokens
        elif isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            self._buffer.append(frame.text)

        # End of assistant response → write ONCE
        elif isinstance(frame, LLMFullResponseEndFrame):
            full_text = "".join(self._buffer).strip()
            self._buffer = []

            if full_text:
                self.context_aggregator.assistant().add_messages([
                    {
                        "role": "assistant",
                        "content": full_text,
                    }
                ])

        await self.push_frame(frame, direction)