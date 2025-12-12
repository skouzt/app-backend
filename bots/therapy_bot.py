"""Therapy bot with session management and state-based conversation."""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from loguru import logger
from pipecat.frames.frames import (
    TextFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from openai.types.chat import ChatCompletionMessageParam

from bots.base_bot import BaseBot
from config.bot import BotConfig
from prompts.therapy_prompts_enhanced import get_enhanced_therapy_prompt
from services.session_store import SessionStore


class TherapyBotState:
    """Simple state machine for conversation flow."""
    
    GREETING = "greeting"
    ASSESSMENT = "assessment"
    CLOSING = "closing"
    
    def __init__(self):
        self.current_state = self.GREETING
        self.session_data = {
            "mood_trend": "stable",
            "coping_strategies_count": 0,
            "topics_count": 0,
            "crisis_detected": False
        }
    
    def update(self, key: str, value: Any):
        """Update session metric."""
        if key in self.session_data:
            self.session_data[key] = value
    
    def transition(self, new_state: str):
        """Move to new conversation state."""
        logger.info(f"State transition: {self.current_state} → {new_state}")
        self.current_state = new_state


class TherapyBot(BaseBot):
    """Therapy bot with state management and session persistence."""

    def __init__(self, config: BotConfig):
        # Get enhanced therapy prompts
        prompt_config = get_enhanced_therapy_prompt()
        system_messages: List[ChatCompletionMessageParam] = prompt_config["task_messages"]
        
        logger.info("Initializing Aletheia with enhanced prompts")
        super().__init__(config, system_messages)
        
        # Session management
        self.session_id = str(uuid.uuid4())
        self.user_id = "anonymous"
        self.session_store = SessionStore()
        self.created_at = datetime.utcnow().isoformat()
        
        # Conversation state
        self.state = TherapyBotState()
        
        # Track first greeting completion
        self.greeting_complete = False
        
        logger.info(f"Therapy bot initialized with session ID: {self.session_id}")

    async def _handle_first_participant(self):
        """Handle first participant by starting session and sending greeting."""
        
        # Start session in database
        success = self.session_store.start_session(
            session_id=self.session_id,
            user_id=self.user_id,
            bot_name=self.config.bot_name
        )
        
        if not success:
            logger.warning(f"Failed to start session {self.session_id} in database")
        
        # Send greeting (this will unmute STT when complete)
        if self.task is not None:
            greeting = (
                "Hello. I'm Aletheia, here to listen and support you. "
                "How are you feeling today?"
            )
            
            # Queue frames with speech control to trigger unmute
            await self.task.queue_frames([
                BotStartedSpeakingFrame(),
                TextFrame(greeting),
                BotStoppedSpeakingFrame(),
            ])
            
            self.greeting_complete = True
            logger.info("✓ Greeting sent - STT should unmute after completion")
        else:
            logger.error("Task is None - cannot send greeting")
        
        logger.info(f"Therapy session {self.session_id} started")

    async def handle_transcription(self, transcription: str):
        """Process user transcription and generate response."""
        
        logger.info(f"User: {transcription}")
        
        # Update session metrics (simplified - in real implementation, 
        # you'd use LLM to extract this info)
        self.state.update("topics_count", self.state.session_data["topics_count"] + 1)
        
        # Create user message
        user_message: ChatCompletionMessageParam = {
            "role": "user",
            "content": transcription
        }
        
        # Add to context
        self.context.add_message(user_message)
        
        # Get bot response from LLM
        response = await self.llm.process_frame(
            OpenAILLMContextFrame(context=self.context),
            FrameDirection.DOWNSTREAM # type: ignore
        )
        
        # Queue response with speech control
        if self.task:
            await self.task.queue_frames([
                BotStartedSpeakingFrame(),
                TextFrame(response), # type: ignore
                BotStoppedSpeakingFrame(),
            ])
        
        # Save session data periodically
        self.save_session_data()

    def save_session_data(self):
        """Save session metadata when conversation ends."""
        try:
            if not hasattr(self, 'created_at'):
                return
                
            start_time = datetime.fromisoformat(self.created_at)
            duration_minutes = int((datetime.utcnow() - start_time).total_seconds() // 60)
            
            metadata = {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "duration_minutes": duration_minutes,
                **self.state.session_data
            }
            
            self.session_store.save_metadata(metadata)
            logger.info(f"Saved session data for {self.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")

    async def cleanup(self):
        """Override cleanup to save session data."""
        self.save_session_data()
        await super().cleanup()