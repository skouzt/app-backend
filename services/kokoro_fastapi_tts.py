"""Kokoro TTS Service with performance optimizations for high-latency networks."""

import aiohttp
import asyncio
import io
import socket
import time
from typing import AsyncGenerator, Optional
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    ErrorFrame,
)
from pipecat.services.tts_service import TTSService


class KokoroFastAPIService(TTSService):
    """Kokoro TTS Service optimized for cross-region deployment."""

    def __init__(
        self,
        *,
        voice: str = "af_bella",
        speed: float = 1.0,
        base_url: str = "http://0.0.0.0:8880",
        endpoint: str = "/v1/audio/speech",
        model: str = "kokoro",
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        self._voice = voice
        self._speed = speed
        self._base_url = base_url.rstrip("/")
        self._endpoint = endpoint
        self._model = model
        
        # ✅ OPTIMIZED: Persistent session with aggressive timeouts and IPv4
        timeout = aiohttp.ClientTimeout(
            total=15,      # Max total time per request
            connect=3,     # Max time to establish connection
            sock_read=12   # Max time between bytes received
        )
        
        # ✅ Connection pooling to reuse connections
        connector = aiohttp.TCPConnector(
            limit=10,              # Max concurrent connections
            ttl_dns_cache=300,     # Cache DNS for 5 minutes
            use_dns_cache=True,
            family=socket.AF_INET  # Force IPv4 (faster in some regions)
        )
        
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
        
        logger.info(f"Kokoro FastAPI service initialized: {base_url}{endpoint}")
        logger.info(f"Timeout config: total={timeout.total}s, connect={timeout.connect}s, read={timeout.sock_read}s")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech by calling the FastAPI service with performance tracking."""
        
        start_time = time.time()
        logger.debug(f"⏱️ TTS START: text_len={len(text)} chars, target_url={self._base_url}{self._endpoint}")
        
        try:
            # ✅ OpenAI-compatible format
            payload = {
                "input": text,
                "voice": self._voice,
                "speed": self._speed,
                "model": self._model,
            }
            
            # ✅ Timeout wrapper to fail fast
            async with asyncio.timeout(12):
                async with self._session.post(
                    f"{self._base_url}{self._endpoint}",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"❌ TTS HTTP {response.status} after {time.time() - start_time:.2f}s")
                        logger.debug(f"Error details: {error_text[:100]}...")
                        yield ErrorFrame(error=f"TTS API error: {response.status}")
                        return
                    
                    # ✅ Get audio data
                    audio_bytes = await response.read()
                    elapsed = time.time() - start_time
                    
                    logger.debug(f"✅ TTS COMPLETE: {elapsed:.2f}s, {len(audio_bytes)} bytes, status={response.status}")
                    
                    # Parse audio (your existing logic)
                    content_type = response.headers.get('Content-Type', '')
                    if 'audio/mpeg' in content_type or audio_bytes.startswith(b'ID3'):
                        audio_data = self._decode_mp3(audio_bytes)
                    elif 'audio/wav' in content_type or audio_bytes.startswith(b'RIFF'):
                        audio_data = self._parse_wav(audio_bytes)
                    else:
                        audio_data = self._parse_raw_audio(audio_bytes)
                    
                    if not audio_data:
                        logger.error(f"❌ No audio data parsed after {elapsed:.2f}s")
                        yield ErrorFrame(error="Empty audio response")
                        return
                    
                    yield TTSStartedFrame()
                    yield TTSAudioRawFrame(
                        audio=audio_data,
                        sample_rate=self._sample_rate,
                        num_channels=1,
                    )
                    yield TTSStoppedFrame()

        except asyncio.TimeoutError:
            logger.error(f"❌ TTS TIMEOUT after {time.time() - start_time:.2f}s")
            yield ErrorFrame(error=f"TTS timeout - service too slow (>12s)")
            
        except Exception as e:
            logger.error(f"❌ TTS EXCEPTION after {time.time() - start_time:.2f}s: {type(e).__name__}")
            yield ErrorFrame(error=f"TTS generation error: {type(e).__name__}")

    # Your existing audio parsing methods remain unchanged...
    def _decode_mp3(self, mp3_bytes: bytes) -> bytes:
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
            if audio.channels > 1:
                audio = audio.set_channels(1)
            if audio.frame_rate != self._sample_rate:
                logger.debug(f"Resampling from {audio.frame_rate}Hz to {self._sample_rate}Hz")
                audio = audio.set_frame_rate(self._sample_rate)
            audio = audio.set_sample_width(2)
            return audio.raw_data
        except ImportError:
            logger.error("pydub not installed")
            return b""
        except Exception as e:
            logger.error(f"MP3 decode failed: {type(e).__name__}")
            return b""

    def _parse_raw_audio(self, audio_bytes: bytes) -> bytes:
        try:
            import numpy as np
            num_samples = len(audio_bytes) // 4
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32, count=num_samples)
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_int16 = (audio_array * 32767).astype(np.int16)
            return audio_int16.tobytes()
        except Exception as e:
            logger.error(f"Raw audio parse failed: {type(e).__name__}")
            return b""

    def _parse_wav(self, audio_bytes: bytes) -> bytes:
        try:
            import wave
            wav_buffer = io.BytesIO(audio_bytes)
            with wave.open(wav_buffer, 'rb') as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                logger.debug(f"WAV: {channels}ch, {sample_width*8}bit, {sample_rate}Hz")
                frames = wav_file.readframes(wav_file.getnframes())
                if sample_width == 4:
                    import numpy as np
                    audio_array = np.frombuffer(frames, dtype=np.float32)
                    audio_array = np.clip(audio_array, -1.0, 1.0)
                    audio_int16 = (audio_array * 32767).astype(np.int16)
                    return audio_int16.tobytes()
                elif sample_width == 2:
                    return frames
                else:
                    logger.error(f"Unsupported WAV: {sample_width*8}bit")
                    return b""
        except Exception as e:
            logger.error(f"WAV parse failed: {type(e).__name__}")
            return b""

    async def cleanup(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("Kokoro FastAPI session closed")