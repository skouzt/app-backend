"""Therapy bot with session management and state-based conversation."""

import asyncio
import os  # ⭐ Added for environment variable access
import uuid
import hashlib  # ⭐ Added for safe user referencing
from datetime import datetime
from typing import List, Any, Optional, Dict  # ⭐ Added Optional, Dict for type hints

from loguru import logger
from pipecat.frames.frames import (
    TextFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
)
from openai.types.chat import ChatCompletionMessageParam

from bots.base_bot import BaseBot
from config.bot import BotConfig
from prompts.therapy_prompts_enhanced import get_enhanced_therapy_prompt
from services.session_store import SessionStore


def _safe_user_ref(user_id: str) -> str:
    """Create a safe, non-reversible reference for logging."""
    if not user_id or user_id == "anonymous":
        return "anonymous"
    # Create a deterministic but non-reversible hash
    return f"user_{hashlib.sha256(user_id.encode()).hexdigest()[:8]}"


class TherapyBotState:
    """Simple state machine for conversation flow."""
    
    GREETING = "greeting"
    ASSESSMENT = "assessment"
    EXPLORATION = "exploration"
    CLOSING = "closing"
    
    def __init__(self):
        self.current_state = self.GREETING
        self.session_data = {
            "mood_trend": "stable",
            "coping_strategies_count": 0,
            "topics_count": 0,
            "crisis_detected": False,
            "turn_count": 0,
        }
    
    def update(self, key: str, value: Any):
        """Update session metric."""
        if key in self.session_data:
            self.session_data[key] = value
            # ⭐ REMOVED: Debug log of sensitive session data
    
    def transition(self, new_state: str):
        """Move to new conversation state."""
        if new_state != self.current_state:
            # ✅ ACCEPTABLE: State transitions are system events
            logger.info(
                "State transition",
                extra={
                    "from_state": self.current_state,
                    "to_state": new_state
                }
            )
            self.current_state = new_state


class TherapyBot(BaseBot):
    """Therapy bot with state management and session persistence."""

    def __init__(self, config: BotConfig):
        """Initialize Aletheia with full therapeutic system prompt and session management."""
        
        # ⭐ FIXED: Get user_id from environment variable set by main.py
        user_id = os.environ.get("BOT_USER_ID")
        safe_user_ref = _safe_user_ref(user_id) if user_id else "anonymous"
        
        if not user_id:
            # Fallback: Use generic prompt if user_id is not available
            logger.warning("BOT_USER_ID not set in environment. Using generic therapy prompt.")
            system_messages: List[ChatCompletionMessageParam] = [{
                "role": "system", 
                "content": "You are Aletheia, a warm and supportive therapist."
            }]
        else:
            # ⭐ FIXED: Pass user_id to get personalized prompt
            logger.info(f"Fetching personalized prompt for {safe_user_ref}")
            prompt_config = get_enhanced_therapy_prompt(user_id)
            system_messages: List[ChatCompletionMessageParam] = prompt_config["task_messages"]
        
        logger.info("Initializing Aletheia with enhanced therapeutic prompt")
        super().__init__(config, system_messages)
        
        # Session management
        self.session_id = str(uuid.uuid4())
        self.user_id = user_id or "anonymous"  # ⭐ Use actual user_id if available
        self.safe_user_ref = safe_user_ref
        self.session_store = SessionStore()
        self.created_at = datetime.utcnow().isoformat()
        
        # Conversation state
        self.state = TherapyBotState()
        
        # Track if greeting was sent
        self._greeting_sent = False
        
        # ✅ SAFE: Log initialization without sensitive data
        logger.info(
            "Therapy bot initialized",
            extra={
                "session_id": self.session_id[:8],  # First 8 chars only
                "user_ref": safe_user_ref
            }
        )

    async def _handle_first_participant(self):
        """Handle first participant by starting session and sending greeting instruction."""
        
        # FIX: Wait for pipeline to be ready
        if self.task is None:
            logger.warning("Task not ready yet - waiting up to 5 seconds")
            for _ in range(50):  # Wait max 5 seconds
                if self.task is not None:
                    break
                await asyncio.sleep(0.1)
        
        if self.task is None:
            logger.error("Pipeline never became ready - cannot send greeting")
            return
        
        # Start session in database
        success = self.session_store.start_session(
            session_id=self.session_id,
            user_id=self.user_id, # type: ignore
            bot_name=self.config.bot_name
        )
        
        if not success:
            logger.warning(f"Failed to start session in database", extra={"session_id": self.session_id[:8]})
        
        # ✅ Send greeting instruction ONCE
        if not self._greeting_sent:
            await self.task.queue_frames([
                    BotStartedSpeakingFrame(),
                    TextFrame(
                        "Hello there... I'm Aletheia. I'm really glad you're here. "
                        "Take your time — there's no rush."
                    ),
                    BotStoppedSpeakingFrame(),
                ])
            self._greeting_sent = True
            logger.info("Greeting sent")
            
            # Transition from GREETING → ASSESSMENT
            self.state.transition(TherapyBotState.ASSESSMENT)
        
        logger.info("Therapy session started", extra={"session_id": self.session_id[:8]})

    def save_session_data(self):
        """Save session metadata when conversation ends."""
        try:
            if not hasattr(self, 'created_at'):
                return
                
            start_time = datetime.fromisoformat(self.created_at)
            duration_minutes = int((datetime.utcnow() - start_time).total_seconds() // 60)
            
            # Update state with final metrics
            self.state.update("duration_minutes", duration_minutes)
            
            metadata = {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "bot_name": self.config.bot_name,
                "created_at": self.created_at,
                **self.state.session_data
            }
            
            self.session_store.save_metadata(metadata)
            
            # ✅ SAFE: Log only non-sensitive metadata
            safe_metadata = {
                "session_id": self.session_id[:8],
                "user_ref": self.safe_user_ref,
                "duration_minutes": duration_minutes,
                "bot_name": self.config.bot_name,
                "has_metadata": True
            }
            logger.info("Session data saved", extra=safe_metadata)
            
        except Exception as e:
            logger.error(f"Failed to save session data: {type(e).__name__}")

    async def cleanup(self):
        """Override cleanup to save session data before shutdown."""
        logger.info("Cleaning up therapy bot", extra={"session_id": self.session_id[:8]})
        self.save_session_data()
        await super().cleanup()