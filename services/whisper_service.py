# external_whisper_stt.py (CORRECTED - Stateless Wrapper)
import asyncio
from typing import AsyncGenerator, Optional
import aiohttp
from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame, ErrorFrame, CancelFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

class ExternalWhisperSegmentedSTTService(SegmentedSTTService):
    def __init__(self, *, stt_url: str, api_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._stt_url = f"{stt_url.rstrip('/')}/transcribe_direct"  # Use direct endpoint
        self._api_key = api_key
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self):
        if not self._session:
            headers = {"X-API-Key": self._api_key} if self._api_key else {}
            self._session = aiohttp.ClientSession(headers=headers)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio to API and yield transcription immediately"""
        await self._ensure_session()
        assert self._session is not None

        try:
            form = aiohttp.FormData()
            form.add_field("file", audio, filename="audio.wav", content_type="audio/wav")

            async with self._session.post(self._stt_url, data=form) as resp:
                if resp.status != 200:
                    logger.error(f"STT API error {resp.status}")
                    yield ErrorFrame(error=f"HTTP {resp.status}")
                    return

                data = await resp.json()
                text = data.get("text", "").strip()

                if text:
                    yield TranscriptionFrame(
                        text=text,
                        user_id=self._user_id,
                        timestamp=time_now_iso8601(),
                        result=data
                    )
        except Exception as e:
            logger.error(f"STT error: {e}")
            yield ErrorFrame(error=str(e))

    async def cleanup(self):
        if self._session:
            await self._session.close()
            self._session = None
