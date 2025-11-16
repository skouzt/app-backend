"""
Kokoro TTS Service for Pipecat
Kokoro v1.0 82M - Fast, high-quality open-source TTS
"""

import asyncio
import logging
from typing import AsyncGenerator
import numpy as np
import torch
from kokoro import KPipeline  # ✅ FIXED

from pipecat.frames.frames import (
    AudioRawFrame,
    TextFrame,
    Frame,
    ErrorFrame,
)
from pipecat.services.ai_services import TTSService

logger = logging.getLogger(__name__)


class KokoroTTSService(TTSService):
    """
    Kokoro TTS - 82M parameter model
    Fast, natural-sounding voice synthesis
    
    Voices:
    - af (female, American)
    - af_bella (female, warm)
    - af_sarah (female, professional)
    - am (male, American)
    - am_adam (male, deep)
    - am_michael (male, friendly)
    - bf (female, British)
    - bm (male, British)
    """
    
    def __init__(
        self,
        voice: str = "af_sarah",
        model_path: str = None,
        device: str = "cpu",
        speed: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self._voice = voice
        self._model_path = model_path
        self._device = device
        self._speed = speed
        self._model = None
        self._sample_rate = 24000
        
        logger.info(f"Initializing Kokoro TTS with voice: {voice}")
    
    async def start(self, frame: Frame):
        """Initialize Kokoro model"""
        await super().start(frame)
        
        try:
            # Load model in thread to avoid blocking
            self._model = await asyncio.to_thread(self._load_model)
            logger.info(f"Kokoro TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Kokoro TTS: {e}", exc_info=True)
            raise
    
    def _load_model(self):
        """Load Kokoro model"""
        try:
            from kokoro import KokoroModel
            
            # Load model
            if self._model_path:
                model = KokoroModel(self._model_path, device=self._device)
            else:
                # Download and cache model automatically
                model = KokoroModel.from_pretrained(
                    'hexgrad/Kokoro-82M',
                    device=self._device
                )
            
            return model
        except ImportError:
            logger.error("Kokoro not installed. Install with: pip install kokoro-tts")
            raise
    
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Generate speech from text
        """
        if not self._model:
            logger.error("Kokoro model not initialized")
            yield ErrorFrame("TTS model not initialized")
            return
        
        try:
            logger.debug(f"Generating speech for: {text}")
            
            # Generate audio in thread
            audio = await asyncio.to_thread(
                self._generate_audio,
                text
            )
            
            if audio is not None:
                # Convert to int16 PCM
                audio_int16 = (audio * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                
                # Yield audio frame
                yield AudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=self._sample_rate,
                    num_channels=1
                )
            else:
                logger.warning("No audio generated")
            
        except Exception as e:
            logger.error(f"Kokoro TTS error: {e}", exc_info=True)
            yield ErrorFrame(f"TTS error: {str(e)}")
    
    def _generate_audio(self, text: str) -> np.ndarray:
        """Generate audio using Kokoro"""
        try:
            # Generate audio
            audio = generate(
                text,
                voice=self._voice,
                speed=self._speed,
                model=self._model
            )
            
            return audio
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return None
    
    async def process_frame(self, frame: Frame, direction: str):
        """Process text frames"""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            # Generate speech for text
            async for audio_frame in self.run_tts(frame.text):
                await self.push_frame(audio_frame, direction)
        else:
            await self.push_frame(frame, direction)


class KokoroTTSServiceStreaming(TTSService):
    """
    Streaming version of Kokoro TTS
    Generates audio in chunks for lower latency
    """
    
    def __init__(
        self,
        voice: str = "af_sarah",
        model_path: str = None,
        device: str = "cpu",
        speed: float = 1.0,
        chunk_size: int = 50,  # Characters per chunk
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self._voice = voice
        self._model_path = model_path
        self._device = device
        self._speed = speed
        self._chunk_size = chunk_size
        self._model = None
        self._sample_rate = 24000
        
        logger.info(f"Initializing Kokoro TTS (Streaming) with voice: {voice}")
    
    async def start(self, frame: Frame):
        """Initialize Kokoro model"""
        await super().start(frame)
        
        try:
            self._model = await asyncio.to_thread(self._load_model)
            logger.info("Kokoro TTS model loaded (streaming mode)")
        except Exception as e:
            logger.error(f"Failed to load Kokoro TTS: {e}", exc_info=True)
            raise
    
    def _load_model(self):
        """Load Kokoro model"""
        try:
            from kokoro import KokoroModel
            
            if self._model_path:
                model = KokoroModel(self._model_path, device=self._device)
            else:
                model = KokoroModel.from_pretrained(
                    'hexgrad/Kokoro-82M',
                    device=self._device
                )
            
            return model
        except ImportError:
            logger.error("Kokoro not installed. Install with: pip install kokoro-tts")
            raise
    
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Generate speech in chunks for streaming
        """
        if not self._model:
            yield ErrorFrame("TTS model not initialized")
            return
        
        try:
            # Split text into chunks (by sentence or size)
            chunks = self._split_text(text)
            
            for chunk in chunks:
                if not chunk.strip():
                    continue
                
                logger.debug(f"Generating chunk: {chunk[:50]}...")
                
                # Generate audio for chunk
                audio = await asyncio.to_thread(
                    self._generate_audio,
                    chunk
                )
                
                if audio is not None:
                    audio_int16 = (audio * 32767).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()
                    
                    yield AudioRawFrame(
                        audio=audio_bytes,
                        sample_rate=self._sample_rate,
                        num_channels=1
                    )
        
        except Exception as e:
            logger.error(f"Kokoro TTS streaming error: {e}", exc_info=True)
            yield ErrorFrame(f"TTS error: {str(e)}")
    
    def _split_text(self, text: str) -> list:
        """Split text into chunks for streaming"""
        # Simple sentence splitting
        import re
        sentences = re.split(r'([.!?]+\s+)', text)
        
        chunks = []
        current_chunk = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            
            if len(current_chunk) + len(sentence) > self._chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence + punctuation
            else:
                current_chunk += sentence + punctuation
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text]
    
    def _generate_audio(self, text: str) -> np.ndarray:
        """Generate audio using Kokoro"""
        try:
            audio = generate(
                text,
                voice=self._voice,
                speed=self._speed,
                model=self._model
            )
            return audio
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return None
    
    async def process_frame(self, frame: Frame, direction: str):
        """Process text frames"""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            async for audio_frame in self.run_tts(frame.text):
                await self.push_frame(audio_frame, direction)
        else:
            await self.push_frame(frame, direction)
