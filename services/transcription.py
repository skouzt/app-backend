"""Cloud speech-to-text — the fallback when on-device recognition comes up empty.

The app transcribes on the phone first: free, private, no upload. This is only
reached when the platform recogniser is unavailable or returns nothing, so the
per-minute meter runs rarely rather than on every voice note. That distinction is
what keeps the ₹999 plan profitable — Deepgram bills ~$0.0077/min, and at ten voice
notes a day on every take it would cost more than the plan earns.

Audio is forwarded straight to Deepgram and never written to disk or storage.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx
from loguru import logger

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

# nova-3, punctuated and sentence-cased, so the draft reads like something a person
# would send rather than a flat run of words.
PARAMS = {
    "model": "nova-3",
    "language": "en",
    "smart_format": "true",
    "punctuate": "true",
}

# A take is capped at 60s client-side; this is the safety net if that is ever bypassed.
MAX_BYTES = 10 * 1024 * 1024
REQUEST_TIMEOUT = 30.0


class TranscriptionUnavailable(RuntimeError):
    """No cloud transcription configured or reachable."""


def is_configured() -> bool:
    return bool(os.getenv("DEEPGRAM_API_KEY"))


async def transcribe(audio: bytes, content_type: Optional[str] = None) -> str:
    key = os.getenv("DEEPGRAM_API_KEY")
    if not key:
        raise TranscriptionUnavailable("DEEPGRAM_API_KEY not set")

    if not audio:
        return ""
    if len(audio) > MAX_BYTES:
        raise ValueError("Audio too large")

    headers = {
        "Authorization": f"Token {key}",
        "Content-Type": content_type or "audio/m4a",
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(DEEPGRAM_URL, params=PARAMS, headers=headers, content=audio)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.warning(f"Deepgram rejected the audio: {e.response.status_code}")
        raise TranscriptionUnavailable("Transcription service error") from e
    except Exception as e:
        logger.warning(f"Deepgram unreachable: {type(e).__name__}")
        raise TranscriptionUnavailable("Transcription service unreachable") from e

    try:
        alt = data["results"]["channels"][0]["alternatives"][0]
        return str(alt.get("transcript") or "").strip()
    except (KeyError, IndexError, TypeError):
        logger.warning("Deepgram returned an unexpected shape")
        return ""
