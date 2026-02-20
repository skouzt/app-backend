"""Alibaba Qwen-TTS Service for Pipecat."""

import os
import base64
import asyncio
import aiohttp
from typing import AsyncGenerator
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    ErrorFrame,
)
from pipecat.services.tts_service import TTSService


class QwenTTSService(TTSService):
    """
    Alibaba Cloud Qwen-TTS Service for Pipecat.
    Uses Qwen3-TTS-Flash model with Cherry voice.
    """

    def __init__(
        self,
        *,
        api_key: str = None,
        voice: str = "Cherry",  # Sunny, friendly young female
        model: str = "qwen3-tts-flash",
        sample_rate: int = 24000,
        speed: float = 1.0,
        language_type: str = "English",  # Match text language
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError("DASHSCOPE_API_KEY not provided")
        
        self._voice = voice
        self._model = model
        self._speed = speed
        self._sample_rate = sample_rate
        self._language_type = language_type
        
        # DashScope API endpoint (Singapore region for international)
        self._base_url = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        
        logger.info(f"QwenTTSService: voice={voice}, model={model}")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech using Qwen-TTS."""
        
        if not text or not text.strip():
            return
        
        try:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": self._model,
                "input": {
                    "text": text,
                    "voice": self._voice,
                    "language_type": self._language_type,
                },
                "parameters": {
                    "speed": self._speed,
                    "sample_rate": self._sample_rate,
                    "format": "pcm",
                }
            }
            
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self._base_url,
                    headers=headers,
                    json=payload
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Qwen-TTS error: {response.status} - {error_text}")
                        yield ErrorFrame(error=f"Qwen-TTS API error: {response.status}")
                        return
                    
                    result = await response.json()
                    
                    # Extract base64 audio
                    if "output" not in result or "audio" not in result["output"]:
                        logger.error(f"Invalid response: {result}")
                        yield ErrorFrame(error="Invalid Qwen-TTS response")
                        return
                    
                    audio_b64 = result["output"]["audio"]
                    audio_bytes = base64.b64decode(audio_b64)
                    
                    if not audio_bytes:
                        yield ErrorFrame(error="Empty audio")
                        return
                    
                    logger.debug(f"Generated {len(audio_bytes)} bytes PCM")
                    
                    yield TTSStartedFrame()
                    yield TTSAudioRawFrame(
                        audio=audio_bytes,
                        sample_rate=self._sample_rate,
                        num_channels=1,
                    )
                    yield TTSStoppedFrame()

        except asyncio.TimeoutError:
            logger.error("Qwen-TTS timeout")
            yield ErrorFrame(error="Qwen-TTS timeout")
        except Exception as e:
            logger.error(f"Qwen-TTS error: {e}")
            yield ErrorFrame(error=f"Qwen-TTS failed: {str(e)}")

    async def cleanup(self):
        pass