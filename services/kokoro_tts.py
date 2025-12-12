"""Kokoro TTS Service for Pipecat."""

from pathlib import Path
from typing import AsyncGenerator

import numpy as np
from loguru import logger
from kokoro_onnx import Kokoro

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    ErrorFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService

# Base dir = .../server/services
BASE_DIR = Path(__file__).resolve().parent

# Model files live in ../models relative to this file
DEFAULT_MODEL_PATH = BASE_DIR.parent / "models" / "kokoro-v1.0.onnx"
DEFAULT_VOICES_PATH = BASE_DIR.parent / "models" / "voices-v1.0.bin"


class KokoroTTSService(TTSService):
    """Kokoro 82M TTS Service (non-streaming, single chunk)."""

    def __init__(
        self,
        *,
        voice: str = "af_sarah",
        speed: float = 1.0,
        device: str = "cpu",   # Just logged; kokoro-onnx doesn't take this
        sample_rate: int = 24000,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        voices_path: str | Path = DEFAULT_VOICES_PATH,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._voice = voice
        self._speed = speed
        self._device = device
        self._sample_rate = sample_rate

        model_path = Path(model_path)
        voices_path = Path(voices_path)

        logger.info(
            f"Loading Kokoro model "
            f"(model_path={model_path}, voices_path={voices_path}) "
            f"[device hint: {device}]"
        )

        # Kokoro(model_path, voices_path, espeak_config=None, vocab_config=None)
        self._kokoro = Kokoro(str(model_path), str(voices_path))

        logger.success("Kokoro model loaded")
        logger.info(f"Kokoro TTS initialized with voice={voice}, speed={speed}")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Generate speech from text.

        Pipecat expects an async generator yielding Frames:
        - TTSStartedFrame
        - one or more TTSAudioRawFrame
        - TTSStoppedFrame
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            # Kokoro API: create(text=..., voice=..., speed=...)
            audio_array = self._kokoro.create(
                text=text,
                voice=self._voice,
                speed=self._speed,
            )

            # Some versions return (samples, sample_rate)
            if isinstance(audio_array, tuple):
                audio_array, sr = audio_array
                self._sample_rate = int(sr)

            # float32 [-1, 1] -> int16 PCM
            audio_int16 = (audio_array * 32767).astype(np.int16)

            yield TTSStartedFrame()
            yield TTSAudioRawFrame(
                audio=audio_int16.tobytes(),
                sample_rate=self._sample_rate,
                num_channels=1,
            )
            yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"{self} Kokoro TTS failed: {e}")
            yield ErrorFrame(error=f"Kokoro TTS error: {e}")


class KokoroStreamingTTSService(KokoroTTSService):
    """
    Placeholder streaming-compatible class.

    For now this just behaves like the base KokoroTTSService (single chunk),
    but keeps a separate type in case you later implement true streaming.
    """

    # No override needed yet; inherits run_tts from KokoroTTSService
    pass
