"""Kokoro TTS Service using FastAPI endpoint (handles MP3 audio response)."""

import aiohttp
import io
import tempfile
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
    """Kokoro TTS Service that calls a remote FastAPI instance."""

    def __init__(
        self,
        *,
        voice: str = "af_bella",
        speed: float = 1.0,
        base_url: str = "http://localhost:8880",
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
        self._session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"Kokoro FastAPI service initialized: {base_url}{endpoint}")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech by calling the FastAPI service."""
        
        logger.debug(f"Generating TTS via FastAPI: [{text[:50]}...]")
        
        try:
            if not self._session or self._session.closed:
                self._session = aiohttp.ClientSession()
            
            # OpenAI-compatible format
            payload = {
                "input": text,
                "voice": self._voice,
                "speed": self._speed,
                "model": self._model,
            }
            
            async with self._session.post(
                f"{self._base_url}{self._endpoint}",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Kokoro FastAPI error: {response.status} - {error_text}")
                    yield ErrorFrame(error=f"TTS API error: {response.status}")
                    return
                
                # Get binary audio data
                audio_bytes = await response.read()
                logger.debug(f"Received {len(audio_bytes)} bytes of audio")
                
                # Check content type
                content_type = response.headers.get('Content-Type', '')
                logger.debug(f"Content-Type: {content_type}")
                
                # Parse audio based on format
                if 'audio/mpeg' in content_type or 'audio/mp3' in content_type or audio_bytes.startswith(b'ID3'):
                    # It's an MP3 file
                    audio_data = self._decode_mp3(audio_bytes)
                elif 'audio/wav' in content_type or audio_bytes.startswith(b'RIFF'):
                    # It's a WAV file
                    audio_data = self._parse_wav(audio_bytes)
                else:
                    # Try raw audio data
                    audio_data = self._parse_raw_audio(audio_bytes)
                
                if not audio_data:
                    logger.error("No audio data generated")
                    yield ErrorFrame(error="Empty audio response")
                    return
                
                yield TTSStartedFrame()
                yield TTSAudioRawFrame(
                    audio=audio_data,
                    sample_rate=self._sample_rate,
                    num_channels=1,
                )
                yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"Kokoro FastAPI generation failed: {e}")
            yield ErrorFrame(error=f"TTS generation error: {e}")

    def _decode_mp3(self, mp3_bytes: bytes) -> bytes:
        """Decode MP3 to raw PCM using pydub."""
        try:
            from pydub import AudioSegment
            import io
            
            # Load MP3 from bytes
            audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
            
            # Convert to mono if needed
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Resample if needed
            if audio.frame_rate != self._sample_rate:
                logger.debug(f"Resampling from {audio.frame_rate}Hz to {self._sample_rate}Hz")
                audio = audio.set_frame_rate(self._sample_rate)
            
            # Convert to 16-bit PCM
            audio = audio.set_sample_width(2)  # 2 bytes = 16 bits
            
            # Get raw PCM data
            pcm_data = audio.raw_data
            
            logger.debug(f"Decoded MP3: {len(pcm_data)} bytes PCM, {audio.frame_rate}Hz, {audio.channels}ch")
            
            return pcm_data
            
        except ImportError:
            logger.error("pydub not installed. Install with: pip install pydub")
            logger.error("Also requires ffmpeg: sudo apt install ffmpeg")
            return b""
        except Exception as e:
            logger.error(f"Failed to decode MP3: {e}")
            return b""

    def _parse_raw_audio(self, audio_bytes: bytes) -> bytes:
        """Parse raw audio data (assumed to be float32 PCM)."""
        try:
            import numpy as np
            
            # Assume float32 PCM data
            num_samples = len(audio_bytes) // 4  # 4 bytes per float32
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32, count=num_samples)
            
            logger.debug(f"Parsed raw audio: {len(audio_array)} samples (float32)")
            
            # Convert float32 [-1.0, 1.0] to int16 PCM
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_int16 = (audio_array * 32767).astype(np.int16)
            
            return audio_int16.tobytes()
            
        except Exception as e:
            logger.error(f"Failed to parse raw audio: {e}")
            return b""

    def _parse_wav(self, audio_bytes: bytes) -> bytes:
        """Parse WAV file and extract raw PCM data."""
        try:
            import wave
            
            # Create BytesIO object
            wav_buffer = io.BytesIO(audio_bytes)
            
            # Open as WAV file
            with wave.open(wav_buffer, 'rb') as wav_file:
                # Verify format
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                
                logger.debug(f"WAV format: {channels}ch, {sample_width*8}bit, {sample_rate}Hz")
                
                # Update sample rate if different
                if sample_rate != self._sample_rate:
                    logger.warning(f"Sample rate mismatch: expected {self._sample_rate}, got {sample_rate}")
                    self._sample_rate = sample_rate
                
                # Read raw frames
                frames = wav_file.readframes(wav_file.getnframes())
                
                # If it's float32, convert to int16
                if sample_width == 4:
                    import numpy as np
                    audio_array = np.frombuffer(frames, dtype=np.float32)
                    audio_array = np.clip(audio_array, -1.0, 1.0)
                    audio_int16 = (audio_array * 32767).astype(np.int16)
                    return audio_int16.tobytes()
                elif sample_width == 2:
                    # Already int16
                    return frames
                else:
                    logger.error(f"Unsupported WAV format: {sample_width*8}bit")
                    return b""
                    
        except Exception as e:
            logger.error(f"Failed to parse WAV: {e}")
            return b""

    async def cleanup(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("Kokoro FastAPI session closed")