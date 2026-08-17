"""Chat endpoints.

Two routes, both scoped to the caller's own user_id from the Clerk token. Sessions
are never exposed — the client sends a message and reads its thread; the server does
the grouping and summarising quietly underneath.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from loguru import logger
from pydantic import BaseModel, Field

from core.rate_limit import CHAT, CLOSE, READ, TRANSCRIBE, rate_limited
from services.chat_service import (
    close_active_session,
    in_thread,
    fetch_messages,
    get_active_session_id,
    send_message,
)
from services.session_reaper import finalise_session
from services.subscription_service import get_subscription_state
from services.transcription import TranscriptionUnavailable, is_configured, transcribe

router = APIRouter()

MAX_MESSAGE_CHARS = 4000


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: str


class SendMessageResponse(BaseModel):
    session_id: str
    user_message: MessageOut
    reply: MessageOut


class MessagesPage(BaseModel):
    messages: List[MessageOut]
    has_more: bool
    next_cursor: Optional[str] = None


def _out(row: Dict[str, Any]) -> MessageOut:
    return MessageOut(
        id=str(row["id"]),
        role=str(row["role"]),
        content=str(row["content"]),
        created_at=str(row["created_at"]),
    )


@router.post("/chat/message", response_model=SendMessageResponse)
async def post_message(
    payload: SendMessageRequest,
    user_id: str = Depends(rate_limited("chat", CHAT)),
):
    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="Message is empty")

    access = await in_thread(get_subscription_state, user_id)
    if not access["allowed"]:
        raise HTTPException(
            status_code=402,
            detail={"reason": access["reason"], "plan": access.get("plan")},
        )

    try:
        session_id, user_msg, reply = await send_message(user_id, content)
    except Exception as e:
        logger.error(f"send_message failed for user: {type(e).__name__}: {e}")
        raise HTTPException(status_code=503, detail="Lily could not reply just now")

    return SendMessageResponse(
        session_id=session_id,
        user_message=_out(user_msg),
        reply=_out(reply),
    )


class TranscriptionResponse(BaseModel):
    text: str


@router.post("/chat/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    user_id: str = Depends(rate_limited("transcribe", TRANSCRIBE)),
):
    """Fallback dictation, used only when on-device recognition produced nothing.

    Gated behind the subscription because it costs per minute, unlike the on-device
    path. Audio is forwarded to the provider and never stored.
    """
    if not is_configured():
        raise HTTPException(status_code=503, detail="Dictation fallback not available")

    if not (await in_thread(get_subscription_state, user_id))["allowed"]:
        raise HTTPException(status_code=402, detail="Subscription required")

    try:
        data = await audio.read()
        text = await transcribe(data, audio.content_type)
    except ValueError:
        # Only raised by the size guard in transcribe(); the message is ours, but
        # stating it here keeps a future ValueError from leaking something else.
        raise HTTPException(status_code=413, detail="Audio too large")
    except TranscriptionUnavailable:
        raise HTTPException(status_code=503, detail="Could not transcribe just now")
    except Exception as e:
        logger.error(f"transcribe failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Could not transcribe")

    return TranscriptionResponse(text=text)


@router.post("/chat/session/close")
async def close_session(
    background: BackgroundTasks,
    # Each close queues a summarisation call, so this is a paid path too. It is
    # already bounded by the chat limit (you need a session before you can close
    # one), but alternating send/close would otherwise turn 300 messages into
    # 300 summaries.
    user_id: str = Depends(rate_limited("chat_close", CLOSE)),
):
    """End the conversation in progress — what "New chat" does.

    Summarising runs in the background so the response is immediate; without it the
    user would wait on an LLM call just to clear the screen.
    """
    try:
        session_id = await in_thread(close_active_session, user_id)
    except Exception as e:
        logger.error(f"close_active_session failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Could not end the conversation")

    if session_id:
        background.add_task(finalise_session, session_id)

    return {"closed": bool(session_id), "session_id": session_id}


@router.get("/chat/messages", response_model=MessagesPage)
async def get_messages(
    limit: int = Query(50, ge=1, le=100),
    before: Optional[str] = Query(None, description="ISO timestamp cursor"),
    all_sessions: bool = Query(
        False, description="Span every session instead of just the live one"
    ),
    user_id: str = Depends(rate_limited("chat_read", READ)),
):
    try:
        # The chat screen shows the conversation in progress. Once a session closes —
        # by idle or by "New chat" — its thread belongs to Summaries, not here.
        session_id = None if all_sessions else await in_thread(get_active_session_id, user_id)
        if not all_sessions and session_id is None:
            return MessagesPage(messages=[], has_more=False, next_cursor=None)

        page = await in_thread(
            fetch_messages, user_id, limit, before, session_id
        )
    except Exception as e:
        logger.error(f"fetch_messages failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load conversation")

    return MessagesPage(
        messages=[_out(m) for m in page["messages"]],
        has_more=page["has_more"],
        next_cursor=page["next_cursor"],
    )
