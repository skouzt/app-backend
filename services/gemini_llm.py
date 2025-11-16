"""
Google Gemini 2.5 Flash LLM Service for Pipecat
Ultra-fast, cost-effective LLM
"""

import asyncio
import logging
from typing import AsyncGenerator, List, Dict
import google.generativeai as genai

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    LLMMessagesFrame,
    ErrorFrame,
)
from pipecat.services.ai_services import LLMService
from pipecat.processors.frame_processor import FrameDirection

logger = logging.getLogger(__name__)


class GeminiLLMService(LLMService):
    """
    Google Gemini 2.5 Flash LLM Service
    
    Features:
    - Ultra-fast responses (< 1s)
    - $0.02 per 1M input tokens
    - $0.04 per 1M output tokens
    - 1M token context window
    - Multimodal support
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-exp",
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self._api_key = api_key
        self._model_name = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._model = None
        self._chat = None
        self._conversation_history = []
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        logger.info(f"Initializing Gemini LLM: {model}")
    
    async def start(self, frame: Frame):
        """Initialize Gemini model"""
        await super().start(frame)
        
        try:
            # Initialize model
            self._model = genai.GenerativeModel(
                model_name=self._model_name,
                generation_config={
                    "temperature": self._temperature,
                    "max_output_tokens": self._max_output_tokens,
                }
            )
            
            logger.info(f"Gemini model initialized: {self._model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}", exc_info=True)
            raise
    
    async def set_context(self, messages: List[Dict]):
        """
        Set conversation context
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     roles: 'system', 'user', 'assistant'
        """
        try:
            self._conversation_history = []
            
            # Convert messages to Gemini format
            system_instruction = None
            chat_history = []
            
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                
                if role == "system":
                    # Gemini uses system_instruction separately
                    system_instruction = content
                elif role == "user":
                    chat_history.append({
                        "role": "user",
                        "parts": [content]
                    })
                elif role == "assistant":
                    chat_history.append({
                        "role": "model",  # Gemini uses 'model' instead of 'assistant'
                        "parts": [content]
                    })
            
            # Create chat session with history
            if system_instruction:
                self._model = genai.GenerativeModel(
                    model_name=self._model_name,
                    generation_config={
                        "temperature": self._temperature,
                        "max_output_tokens": self._max_output_tokens,
                    },
                    system_instruction=system_instruction
                )
            
            self._chat = self._model.start_chat(history=chat_history)
            self._conversation_history = chat_history
            
            logger.info(f"Context set with {len(messages)} messages")
            
        except Exception as e:
            logger.error(f"Error setting context: {e}", exc_info=True)
            raise
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames"""
        await super().process_frame(frame, direction)
        
        # Handle LLM messages frame
        if isinstance(frame, LLMMessagesFrame):
            await self.set_context(frame.messages)
        
        # Handle text frame (user input)
        elif isinstance(frame, TextFrame):
            # Generate response
            async for response_frame in self.generate_response(frame.text):
                await self.push_frame(response_frame, direction)
        else:
            await self.push_frame(frame, direction)
    
    async def generate_response(self, user_message: str) -> AsyncGenerator[Frame, None]:
        """
        Generate LLM response
        """
        if not self._chat:
            logger.error("Chat not initialized. Call set_context first.")
            yield ErrorFrame("Chat not initialized")
            return
        
        try:
            logger.debug(f"Generating response for: {user_message}")
            
            # Send message to Gemini (in thread to avoid blocking)
            response = await asyncio.to_thread(
                self._chat.send_message,
                user_message
            )
            
            # Get response text
            response_text = response.text
            
            logger.debug(f"Generated response: {response_text}")
            
            # Yield response as TextFrame
            yield TextFrame(text=response_text)
            
            # Update conversation history
            self._conversation_history.append({
                "role": "user",
                "parts": [user_message]
            })
            self._conversation_history.append({
                "role": "model",
                "parts": [response_text]
            })
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            yield ErrorFrame(f"LLM error: {str(e)}")
    
    def create_context_frame(self, messages: List[Dict]) -> LLMMessagesFrame:
        """
        Create a context frame for the pipeline
        
        Usage in pipeline:
            await task.queue_frames([llm.create_context_frame(messages)])
        """
        return LLMMessagesFrame(messages=messages)
    
    async def get_conversation_history(self) -> List[Dict]:
        """Get current conversation history"""
        return self._conversation_history.copy()


class GeminiLLMServiceStreaming(LLMService):
    """
    Streaming version of Gemini LLM
    Yields tokens as they're generated for lower latency
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-exp",
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self._api_key = api_key
        self._model_name = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._model = None
        self._chat = None
        self._conversation_history = []
        
        genai.configure(api_key=api_key)
        
        logger.info(f"Initializing Gemini LLM (Streaming): {model}")
    
    async def start(self, frame: Frame):
        """Initialize Gemini model"""
        await super().start(frame)
        
        try:
            self._model = genai.GenerativeModel(
                model_name=self._model_name,
                generation_config={
                    "temperature": self._temperature,
                    "max_output_tokens": self._max_output_tokens,
                }
            )
            
            logger.info(f"Gemini streaming model initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}", exc_info=True)
            raise
    
    async def set_context(self, messages: List[Dict]):
        """Set conversation context"""
        try:
            self._conversation_history = []
            
            system_instruction = None
            chat_history = []
            
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                
                if role == "system":
                    system_instruction = content
                elif role == "user":
                    chat_history.append({"role": "user", "parts": [content]})
                elif role == "assistant":
                    chat_history.append({"role": "model", "parts": [content]})
            
            if system_instruction:
                self._model = genai.GenerativeModel(
                    model_name=self._model_name,
                    generation_config={
                        "temperature": self._temperature,
                        "max_output_tokens": self._max_output_tokens,
                    },
                    system_instruction=system_instruction
                )
            
            self._chat = self._model.start_chat(history=chat_history)
            self._conversation_history = chat_history
            
        except Exception as e:
            logger.error(f"Error setting context: {e}", exc_info=True)
            raise
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames"""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMMessagesFrame):
            await self.set_context(frame.messages)
        elif isinstance(frame, TextFrame):
            async for response_frame in self.generate_response_streaming(frame.text):
                await self.push_frame(response_frame, direction)
        else:
            await self.push_frame(frame, direction)
    
    async def generate_response_streaming(self, user_message: str) -> AsyncGenerator[Frame, None]:
        """
        Generate streaming LLM response
        Yields text chunks as they arrive
        """
        if not self._chat:
            yield ErrorFrame("Chat not initialized")
            return
        
        try:
            logger.debug(f"Generating streaming response for: {user_message}")
            
            # Send message with streaming
            response = await asyncio.to_thread(
                self._chat.send_message,
                user_message,
                stream=True
            )
            
            full_response = ""
            
            # Stream chunks
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    # Yield each chunk as it arrives
                    yield TextFrame(text=chunk.text)
            
            # Update conversation history
            self._conversation_history.append({
                "role": "user",
                "parts": [user_message]
            })
            self._conversation_history.append({
                "role": "model",
                "parts": [full_response]
            })
            
        except Exception as e:
            logger.error(f"Error in streaming response: {e}", exc_info=True)
            yield ErrorFrame(f"LLM error: {str(e)}")
    
    def create_context_frame(self, messages: List[Dict]) -> LLMMessagesFrame:
        """Create a context frame"""
        return LLMMessagesFrame(messages=messages)
    
    async def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self._conversation_history.copy()