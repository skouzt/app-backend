"""
Enhanced Kokoro TTS Service with emotion and pause support.
Compatible with kokoro_onnx implementation.
"""

import re
import asyncio
import numpy as np
from pathlib import Path
from typing import AsyncGenerator, Optional, Union, Dict, Any
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    ErrorFrame,
)
from pipecat.services.tts_service import TTSService
from kokoro_onnx import Kokoro

# Base dir = .../server/services
BASE_DIR = Path(__file__).resolve().parent

# Model files live in ../models relative to this file
DEFAULT_MODEL_PATH = BASE_DIR.parent / "models" / "kokoro-v1.0.onnx"
DEFAULT_VOICES_PATH = BASE_DIR.parent / "models" / "voices-v1.0.bin"

# Emotion to voice parameter mapping
EMOTION_MAP = {
    "calm": {"speed": 0.85, "pause_multiplier": 1.25},
    "empathetic": {"speed": 0.90, "pause_multiplier": 1.20},
    "thoughtful": {"speed": 0.88, "pause_multiplier": 1.30},
    "encouraging": {"speed": 0.95, "pause_multiplier": 1.10},
    "concerned": {"speed": 0.87, "pause_multiplier": 1.15},
    "warm": {"speed": 0.95, "pause_multiplier": 1.15},
    "neutral": {"speed": 1.0, "pause_multiplier": 1.0},
}


class EnhancedKokoroTTSService(TTSService):
    """Kokoro TTS with emotion markers and natural pauses."""

    def __init__(
        self,
        *,
        voice: str = "af",
        base_speed: float = 1.0,
        device: str = "cpu",
        sample_rate: int = 24000,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        voices_path: str | Path = DEFAULT_VOICES_PATH,
        params: Optional[Union[Dict[str, Any], object]] = None,  # ✅ Accept dict or object
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        self._voice = voice
        self._base_speed = base_speed
        self._device = device
        self._sample_rate = sample_rate
        
        # ✅ FIXED: Handle both dict and object params
        if params is None:
            # Default values
            self._enable_ssml = True
            self._enable_emotion = True
            self._default_emotion = "empathetic"
        elif isinstance(params, dict):
            # Dict format
            self._enable_ssml = params.get("enable_ssml", True)
            self._enable_emotion = params.get("enable_emotion", True)
            self._default_emotion = params.get("default_emotion", "empathetic")
        else:
            # Object format (has attributes)
            self._enable_ssml = getattr(params, "enable_ssml", True)
            self._enable_emotion = getattr(params, "enable_emotion", True)
            self._default_emotion = getattr(params, "default_emotion", "empathetic")
        
        self._current_emotion = self._default_emotion

        model_path = Path(model_path)
        voices_path = Path(voices_path)

        logger.info(
            f"Loading Kokoro model "
            f"(model_path={model_path}, voices_path={voices_path}) "
            f"[device hint: {device}]"
        )

        # Initialize Kokoro ONNX
        self._kokoro = Kokoro(str(model_path), str(voices_path))

        logger.success("Kokoro model loaded")
        logger.info(f"Enhanced Kokoro TTS: voice={voice}, base_speed={base_speed}")

    def _extract_emotion_blocks(self, text: str) -> list:
        """Parse text into emotion blocks with metadata."""
        blocks = []
        emotion_pattern = r'<emotion name="(\w+)">(.*?)</emotion>'
        last_end = 0

        for match in re.finditer(emotion_pattern, text, re.DOTALL):
            start, end = match.span()
            
            # Add text before emotion tag (if any)
            if start > last_end:
                prefix = text[last_end:start].strip()
                if prefix:
                    blocks.append({
                        "text": prefix,
                        "emotion": self._current_emotion,
                    })
            
            # Add emotion block
            emotion_name = match.group(1)
            emotion_text = match.group(2).strip()
            
            if emotion_text:
                blocks.append({
                    "text": emotion_text,
                    "emotion": emotion_name,
                })
                self._current_emotion = emotion_name
            
            last_end = end

        # Add remaining text
        if last_end < len(text):
            suffix = text[last_end:].strip()
            if suffix:
                blocks.append({
                    "text": suffix,
                    "emotion": self._current_emotion,
                })

        return blocks

    def _process_pauses(self, text: str) -> tuple[str, list]:
        """Extract pause markers and return cleaned text with pause positions."""
        pauses = []
        pause_pattern = r'<pause time="([\d.]+)s"\s*/>'
        
        # Find all pauses
        offset = 0
        for match in re.finditer(pause_pattern, text):
            pause_duration = float(match.group(1))
            position = match.start() - offset
            pauses.append({
                "position": position,
                "duration": pause_duration,
            })
            offset += len(match.group(0))
        
        # Remove pause markers
        cleaned_text = re.sub(pause_pattern, "", text)
        
        return cleaned_text, pauses

    def _calculate_voice_params(self, emotion: str) -> dict:
        """Calculate voice speed and pause multiplier for emotion."""
        emotion_params = EMOTION_MAP.get(emotion, EMOTION_MAP["neutral"])
        
        return {
            "speed": self._base_speed * emotion_params["speed"],
            "pause_multiplier": emotion_params["pause_multiplier"],
        }

    async def _generate_audio(self, text: str, emotion: str) -> np.ndarray:
        """Generate audio using Kokoro ONNX with emotion parameters."""
        params = self._calculate_voice_params(emotion)
        
        try:
            # Run Kokoro generation in thread pool (it's CPU-bound)
            loop = asyncio.get_event_loop()
            audio_array = await loop.run_in_executor(
                None,
                lambda: self._kokoro.create(
                    text=text,
                    voice=self._voice,
                    speed=params["speed"],
                )
            )
            
            # Some versions return (samples, sample_rate)
            if isinstance(audio_array, tuple):
                audio_array, sr = audio_array
                self._sample_rate = int(sr)
            
            if audio_array is not None and len(audio_array) > 0:
                duration = len(audio_array) / self._sample_rate
                logger.debug(f"Generated audio: {duration:.2f}s")
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Kokoro generation failed: {e}")
            return np.array([], dtype=np.float32)

    def _audio_to_bytes(self, audio_array: np.ndarray) -> bytes:
        """Convert float32 audio to int16 PCM bytes."""
        if len(audio_array) == 0:
            return b""
        
        # Clip to [-1, 1] range and convert to int16
        audio_clipped = np.clip(audio_array, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _generate_silence(self, duration: float) -> bytes:
        """Generate silence for pause markers."""
        num_samples = int(duration * self._sample_rate)
        silence = np.zeros(num_samples, dtype=np.int16)
        return silence.tobytes()

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS audio with emotion and pause support."""
        
        logger.debug(f"Generating TTS: [{text[:50]}{'...' if len(text) > 50 else ''}]")

        try:
            yield TTSStartedFrame()

            if not self._enable_emotion:
                # Simple generation without emotion parsing
                audio = await self._generate_audio(text, self._current_emotion)
                if len(audio) > 0:
                    yield TTSAudioRawFrame(
                        audio=self._audio_to_bytes(audio),
                        sample_rate=self._sample_rate,
                        num_channels=1,
                    )
            else:
                # Process emotion blocks
                blocks = self._extract_emotion_blocks(text)
                
                for block in blocks:
                    block_text = block["text"]
                    emotion = block["emotion"]
                    
                    # Process pauses
                    cleaned_text, pauses = self._process_pauses(block_text)
                    
                    if cleaned_text.strip():
                        # Generate audio for this block
                        audio = await self._generate_audio(cleaned_text, emotion)
                        
                        if len(audio) > 0:
                            yield TTSAudioRawFrame(
                                audio=self._audio_to_bytes(audio),
                                sample_rate=self._sample_rate,
                                num_channels=1,
                            )
                    
                    # Handle explicit pauses
                    for pause in pauses:
                        pause_duration = pause["duration"]
                        if pause_duration > 0:
                            silence = self._generate_silence(pause_duration)
                            yield TTSAudioRawFrame(
                                audio=silence,
                                sample_rate=self._sample_rate,
                                num_channels=1,
                            )

            yield TTSStoppedFrame()
            
        except Exception as e:
            logger.error(f"Enhanced Kokoro TTS failed: {e}")
            yield ErrorFrame(error=f"Enhanced Kokoro TTS error: {e}")