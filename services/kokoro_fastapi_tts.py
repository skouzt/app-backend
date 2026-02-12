"""Kokoro TTS Service with RunPod serverless support."""

import os
import aiohttp
import asyncio
import io
import socket
from typing import AsyncGenerator, Optional

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    ErrorFrame,
)
from pipecat.services.tts_service import TTSService


class KokoroFastAPIService(TTSService):
    """Kokoro TTS Service for RunPod serverless deployment."""

    def __init__(
        self,
        *,
        voice: str = "af_bella",
        speed: float = 1.0,
        base_url: str = None,
        endpoint: str = "/v1/audio/speech",
        model: str = "kokoro",
        sample_rate: int = 24000,
        api_key: str = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        # Get from parameters or environment
        self._api_key = api_key or os.getenv("RUNPOD_API_KEY")
        self._base_url = (base_url or os.getenv("KOKORO_FASTAPI_URL", "https://ykbxj7gx1zifmn.api.runpod.ai")).rstrip("/")
        self._endpoint = endpoint
        self._voice = voice
        self._speed = speed
        self._model = model
        
        if not self._api_key:
            raise ValueError("RUNPOD_API_KEY not provided")
        
        timeout = aiohttp.ClientTimeout(
            total=15,
            connect=3,
            sock_read=12
        )
        
        connector = aiohttp.TCPConnector(
            limit=10,
            ttl_dns_cache=300,
            use_dns_cache=True,
            family=socket.AF_INET
        )
        
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech using RunPod Kokoro TTS."""
        
        try:
            payload = {
                "input": text,
                "voice": self._voice,
                "speed": self._speed,
                "model": self._model,
            }
            
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json"
            }
            
            async with asyncio.timeout(12):
                async with self._session.post(
                    f"{self._base_url}{self._endpoint}",
                    json=payload,
                    headers=headers
                ) as response:
                    
                    if response.status != 200:
                        yield ErrorFrame(error=f"TTS API error: {response.status}")
                        return
                    
                    audio_bytes = await response.read()
                    
                    if not audio_bytes:
                        yield ErrorFrame(error="Empty audio response")
                        return
                    
                    audio_data = self._decode_audio(audio_bytes, response.headers.get('Content-Type', ''))
                    
                    if not audio_data:
                        yield ErrorFrame(error="Audio decode failed")
                        return
                    
                    yield TTSStartedFrame()
                    yield TTSAudioRawFrame(
                        audio=audio_data,
                        sample_rate=self._sample_rate,
                        num_channels=1,
                    )
                    yield TTSStoppedFrame()

        except asyncio.TimeoutError:
            yield ErrorFrame(error="TTS timeout")
        except Exception:
            yield ErrorFrame(error="TTS generation failed")

    def _decode_audio(self, audio_bytes: bytes, content_type: str) -> bytes:
        """Decode audio based on content type."""
        if 'audio/mpeg' in content_type or audio_bytes.startswith(b'ID3'):
            return self._decode_mp3(audio_bytes)
        elif 'audio/wav' in content_type or audio_bytes.startswith(b'RIFF'):
            return self._parse_wav(audio_bytes)
        else:
            return self._parse_raw_audio(audio_bytes)

    def _decode_mp3(self, mp3_bytes: bytes) -> bytes:
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
            if audio.channels > 1:
                audio = audio.set_channels(1)
            if audio.frame_rate != self._sample_rate:
                audio = audio.set_frame_rate(self._sample_rate)
            audio = audio.set_sample_width(2)
            return audio.raw_data
        except Exception:
            return b""

    def _parse_raw_audio(self, audio_bytes: bytes) -> bytes:
        try:
            import numpy as np
            num_samples = len(audio_bytes) // 4
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32, count=num_samples)
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_int16 = (audio_array * 32767).astype(np.int16)
            return audio_int16.tobytes()
        except Exception:
            return b""

    def _parse_wav(self, audio_bytes: bytes) -> bytes:
        try:
            import wave
            wav_buffer = io.BytesIO(audio_bytes)
            with wave.open(wav_buffer, 'rb') as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
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
                    return b""
        except Exception:
            return b""


    async def ping(self) -> bool:
        """Send minimal TTS job to wake RunPod serverless worker"""
        if not self._api_key:
            return False

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }

        # Minimal payload — just enough to spin the worker up
        payload = {
            "input": {
                "text": ".",
                "voice": self._voice,
                "speed": self._speed,
            }
        }

        try:
            async with self._session.post(
                f"{self._base_url}/runsync",  # ✅ not /ping
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)  # cold start can take 30-50s
            ) as response:
                data = await response.json()
                # RunPod returns status "COMPLETED" on success
                success = response.status == 200 and data.get("status") == "COMPLETED"
                if not success:
                    logger.warning(f"TTS warm-up unexpected response: {data}")
                return success
        except Exception as e:
            logger.warning(f"Ping failed: {e}")
            return False
    
    async def cleanup(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()