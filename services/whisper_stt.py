"""
Complete Whisper STT Service for Pipecat
Open-source Speech-to-Text using OpenAI Whisper
"""

import asyncio
import logging
from typing import AsyncGenerator
import numpy as np
import whisper

from pipecat.frames.frames import (
    AudioRawFrame,
    TranscriptionFrame,
    Frame,
    ErrorFrame,
)
from pipecat.services.ai_services import STTService
from pipecat.processors.frame_processor import FrameDirection

logger = logging.getLogger(__name__)


class WhisperSTTService(STTService):
    """
    Whisper-based Speech-to-Text service for Pipecat
    
    Models:
    - tiny: 39M params, fastest, lowest quality
    - base: 74M params, good balance (recommended)
    - small: 244M params, better quality
    - medium: 769M params, high quality
    - large: 1550M params, best quality
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        language: str = "en",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self._model_size = model_size
        self._device = device
        self._language = language
        self._model = None
        self._audio_buffer = []
        self._sample_rate = 16000
        self._buffer_duration = 1.0  # Process every 1 second
        
        logger.info(f"Initializing Whisper STT: model={model_size}, device={device}")
    
    async def start(self, frame: Frame):
        """Initialize Whisper model"""
        await super().start(frame)
        
        try:
            logger.info(f"Loading Whisper {self._model_size} model...")
            
            # Load model in thread to avoid blocking
            self._model = await asyncio.to_thread(
                whisper.load_model,
                self._model_size,
                device=self._device
            )
            
            logger.info(f"Whisper model {self._model_size} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            raise
    
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        Process audio and generate transcription
        """
        if not self._model:
            logger.error("Whisper model not initialized")
            yield ErrorFrame("Whisper model not initialized")
            return
        
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Pad or trim to 30 seconds (Whisper's expected input length)
            if len(audio_np) < self._sample_rate * 30:
                audio_np = np.pad(audio_np, (0, self._sample_rate * 30 - len(audio_np)))
            else:
                audio_np = audio_np[:self._sample_rate * 30]
            
            logger.debug(f"Processing audio chunk: {len(audio_np)} samples")
            
            # Transcribe in thread to avoid blocking
            result = await asyncio.to_thread(
                self._model.transcribe,
                audio_np,
                language=self._language,
                fp16=(self._device == "cuda"),
                task="transcribe",
                verbose=False
            )
            
            text = result.get("text", "").strip()
            
            if text:
                logger.debug(f"Transcription: {text}")
                
                # Create transcription frame
                yield TranscriptionFrame(
                    text=text,
                    user_id="user",
                    timestamp=0
                )
                # Set is_final attribute
                frame = TranscriptionFrame(text=text, user_id="user", timestamp=0)
                frame.is_final = True
                yield frame
            else:
                logger.debug("No speech detected in audio chunk")
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}", exc_info=True)
            yield ErrorFrame(f"Transcription error: {str(e)}")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process audio frames"""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, AudioRawFrame):
            # Add audio to buffer
            self._audio_buffer.append(frame.audio)
            
            # Calculate buffer duration
            total_bytes = sum(len(chunk) for chunk in self._audio_buffer)
            buffer_duration = total_bytes / (self._sample_rate * 2)  # 2 bytes per sample (int16)
            
            # Process when buffer reaches threshold
            if buffer_duration >= self._buffer_duration:
                # Concatenate buffered audio
                audio_data = b''.join(self._audio_buffer)
                self._audio_buffer.clear()
                
                # Transcribe
                async for transcription_frame in self.run_stt(audio_data):
                    await self.push_frame(transcription_frame, direction)
        else:
            # Pass through other frames
            await self.push_frame(frame, direction)


class FasterWhisperSTTService(STTService):
    """
    Faster Whisper implementation using faster-whisper library
    ~4x faster than standard Whisper with same accuracy
    Recommended for production
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",  # int8, float16, float32
        language: str = "en",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model = None
        self._audio_buffer = []
        self._sample_rate = 16000
        self._buffer_duration = 1.0
        
        logger.info(f"Initializing Faster Whisper STT: {model_size}, device: {device}")
    
    async def start(self, frame: Frame):
        """Initialize Faster Whisper model"""
        await super().start(frame)
        
        try:
            from faster_whisper import WhisperModel
            
            logger.info(f"Loading Faster Whisper {self._model_size} model...")
            
            # Load model
            self._model = await asyncio.to_thread(
                WhisperModel,
                self._model_size,
                device=self._device,
                compute_type=self._compute_type
            )
            
            logger.info(f"Faster Whisper model {self._model_size} loaded successfully")
            
        except ImportError:
            logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to load Faster Whisper: {e}", exc_info=True)
            raise
    
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio with Faster Whisper"""
        if not self._model:
            yield ErrorFrame("Faster Whisper model not initialized")
            return
        
        try:
            # Convert to numpy array
            audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            
            logger.debug(f"Processing audio: {len(audio_np)} samples")
            
            # Transcribe with VAD (Voice Activity Detection)
            segments, info = await asyncio.to_thread(
                self._model.transcribe,
                audio_np,
                language=self._language,
                beam_size=5,
                vad_filter=True,  # Enable VAD
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    threshold=0.5
                )
            )
            
            # Collect all segments
            full_text = ""
            for segment in segments:
                full_text += segment.text + " "
            
            text = full_text.strip()
            
            if text:
                logger.debug(f"Faster Whisper transcription: {text}")
                
                frame = TranscriptionFrame(
                    text=text,
                    user_id="user",
                    timestamp=0
                )
                frame.is_final = True
                yield frame
            else:
                logger.debug("No speech detected")
        
        except Exception as e:
            logger.error(f"Faster Whisper error: {e}", exc_info=True)
            yield ErrorFrame(f"Transcription error: {str(e)}")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process audio frames"""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, AudioRawFrame):
            self._audio_buffer.append(frame.audio)
            
            total_bytes = sum(len(chunk) for chunk in self._audio_buffer)
            buffer_duration = total_bytes / (self._sample_rate * 2)
            
            if buffer_duration >= self._buffer_duration:
                audio_data = b''.join(self._audio_buffer)
                self._audio_buffer.clear()
                
                async for transcription_frame in self.run_stt(audio_data):
                    await self.push_frame(transcription_frame, direction)
        else:
            await self.push_frame(frame, direction)


# Helper function to choose the best STT service
def create_whisper_service(
    model_size: str = "base",
    device: str = "cpu",
    use_faster: bool = True,
    **kwargs
) -> STTService:
    """
    Factory function to create the best available Whisper service
    
    Args:
        model_size: Whisper model size (tiny, base, small, medium, large)
        device: cpu or cuda
        use_faster: Try to use faster-whisper if available
        **kwargs: Additional arguments
    
    Returns:
        STTService instance
    """
    if use_faster:
        try:
            import faster_whisper
            logger.info("Using Faster Whisper (4x faster)")
            return FasterWhisperSTTService(
                model_size=model_size,
                device=device,
                **kwargs
            )
        except ImportError:
            logger.warning("faster-whisper not available, falling back to standard Whisper")
    
    logger.info("Using standard Whisper")
    return WhisperSTTService(
        model_size=model_size,
        device=device,
        **kwargs
    )