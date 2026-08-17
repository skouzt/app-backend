"""Personalization preferences — how the user wants Lily to sound.

Read and written by the Personalization screen. What is stored here shapes the
system prompt (see prompts/lily_chat.py), so saving invalidates that user's
cached prompt — otherwise a change would appear to do nothing for up to a minute
and the user would tap it again wondering if it took.
"""

from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from core.rate_limit import READ, rate_limited
from prompts.lily_chat import invalidate_prompt_cache
from services.chat_service import in_thread
from services.personalization_service import (
    NOTE_MAX_CHARS,
    fetch_personalization,
    save_personalization,
)

router = APIRouter()


class PersonalizationPayload(BaseModel):
    tone: str
    # Validated against the known keys and levels in the service layer; an
    # unrecognised entry is dropped rather than rejected.
    traits: Dict[str, str] = Field(default_factory=dict)
    check_ins: bool = True
    note: str = Field(default="", max_length=NOTE_MAX_CHARS)


class PersonalizationResponse(PersonalizationPayload):
    pass


@router.get("/personalization", response_model=PersonalizationResponse)
async def read_personalization(user_id: str = Depends(rate_limited("prefs_read", READ))):
    try:
        return await in_thread(fetch_personalization, user_id)
    except Exception as e:
        logger.error(f"personalization read failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Could not load your preferences")


@router.put("/personalization", response_model=PersonalizationResponse)
async def write_personalization(
    payload: PersonalizationPayload,
    user_id: str = Depends(rate_limited("prefs_write", READ)),
):
    try:
        saved = await in_thread(save_personalization, user_id, payload.model_dump())
    except Exception as e:
        logger.error(f"personalization save failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Could not save your preferences")

    invalidate_prompt_cache(user_id)
    return saved
