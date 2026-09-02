"""Chat: sessions, messages, and Lily's replies.

Sessions are invisible to the client. A conversation starts on the first message and
covers one local day, at which point the reaper summarises it
so the *next* conversation opens with context. Nothing here is exposed as an endpoint —
the app only ever sends and reads messages.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from db.supabase import supabase
from services.user_info_service import fetch_user_info, local_today
from prompts.lily_chat import build_chat_messages
from services.llm import complete

# How long a conversation stays open with no messages before it closes and gets
# summarised. 5 minutes fragmented the record — a two-message "Hii / Nothing much"
# became its own permanent entry — and made the thread vanish while someone was still
# thinking. A whole day was the other extreme: nothing gets summarised until the day
# ends, so anything past HISTORY_TURNS is in neither the context window nor memory.

# How much of the thread to replay to the model. Older turns are represented by the
# continuity block instead, so this stays bounded regardless of conversation length.
HISTORY_TURNS = 40

TABLE_SESSIONS = "therapy_sessions"
TABLE_MESSAGES = "messages"


# Transient connection faults, not application errors. supabase-py shares one httpx
# client with HTTP/2 multiplexing; when the server recycles that connection (GOAWAY)
# every in-flight stream fails at once. Serialised code never saw this because there
# was only ever one request in flight — concurrency exposed it. Retrying gets a fresh
# connection, which is exactly what the HTTP/2 spec expects a client to do.
_TRANSIENT = ("RemoteProtocolError", "ConnectError", "ReadError", "ConnectTimeout", "ReadTimeout")
_DB_RETRIES = 2


def _is_transient(exc: BaseException) -> bool:
    return type(exc).__name__ in _TRANSIENT


async def in_thread(fn, *args, **kwargs):
    """Run a blocking DB call off the event loop, retrying transient faults.

    supabase-py is synchronous. Called directly from an `async def`, every query
    stalls the whole event loop — one user's 8 round-trips freeze every other
    request in the process, including ones that only needed to read a token.
    Handing them to the threadpool lets requests actually overlap.

    Retries are safe here because inserts carry a client-generated UUID primary key:
    a genuinely duplicated write fails on the key rather than creating a second row.
    """
    last: BaseException | None = None
    for attempt in range(_DB_RETRIES + 1):
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except Exception as e:
            if not _is_transient(e) or attempt == _DB_RETRIES:
                raise
            last = e
            logger.warning(
                f"DB call hit {type(e).__name__}, retrying "
                f"({attempt + 1}/{_DB_RETRIES})"
            )
            await asyncio.sleep(0.15 * (attempt + 1))
    raise last  # unreachable, but keeps the type checker honest


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ── Sessions ─────────────────────────────────────────────────────────────────


def _fetch_active(user_id: str) -> Optional[Dict[str, Any]]:
    res = (
        supabase.table(TABLE_SESSIONS)
        .select("id, started_at:start_time, last_message_at, message_count, date")
        .eq("user_id", user_id)
        .eq("status", "active")
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def _mark_closing(session_id: str) -> None:
    """Hand a stale session to the reaper. Never summarises inline — sends stay fast."""
    supabase.table(TABLE_SESSIONS).update(
        {"status": "closing", "updated_at": _iso(_now())}
    ).eq("id", session_id).eq("status", "active").execute()


def _create_session(user_id: str, on_date: Optional[str] = None) -> Dict[str, Any]:
    now = _now()
    row = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        # The user's local day, not the server's. Sessions are day-scoped, so
        # this value is the identity of the session, not a label on it.
        "date": on_date or now.date().isoformat(),
        "status": "active",
        "start_time": _iso(now),
        "last_message_at": _iso(now),
        "message_count": 0,
        "created_at": _iso(now),
        "updated_at": _iso(now),
    }
    supabase.table(TABLE_SESSIONS).insert(row).execute()
    return row


def resolve_session(user_id: str) -> Dict[str, Any]:
    """Return the session this message belongs to.

    One session per local day, so a day yields exactly one summary. A gap in the
    conversation no longer starts a new session: someone who writes at breakfast
    and again at midnight is having one day, and reading that back as two
    disconnected entries is not what a journal is for.

    A session is only retired when the user's local date has moved on, which is
    why the reaper's cutoff is a date rather than an idle window.
    """
    info = fetch_user_info(user_id) or {}
    today = local_today(info.get("timezone")).isoformat()

    active = _fetch_active(user_id)

    if active:
        if str(active.get("date")) == today:
            return active
        # A new day began: close yesterday's so it can be summarised, and open
        # today's.
        _mark_closing(str(active["id"]))

    try:
        return _create_session(user_id, on_date=today)
    except Exception as e:
        # The partial unique index means a concurrent send may have just created one.
        logger.warning(f"Session create raced, re-reading: {type(e).__name__}")
        existing = _fetch_active(user_id)
        if existing:
            return existing
        raise


# ── Messages ─────────────────────────────────────────────────────────────────


def _insert_message(session_id: str, user_id: str, role: str, content: str) -> Dict[str, Any]:
    row = {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "user_id": user_id,
        "role": role,
        "content": content,
        "created_at": _iso(_now()),
    }
    supabase.table(TABLE_MESSAGES).insert(row).execute()
    return row


def _touch_session(session_id: str, added: int, known_count: int = 0) -> None:
    """One round-trip. The prior count comes from the session row we already read in
    resolve_session, so re-selecting it was a wasted query on every single message."""
    now = _now()
    supabase.table(TABLE_SESSIONS).update(
        {
            "last_message_at": _iso(now),
            "message_count": known_count + added,
            "updated_at": _iso(now),
        }
    ).eq("id", session_id).execute()


def load_thread(session_id: str, limit: int = HISTORY_TURNS) -> List[Dict[str, Any]]:
    res = (
        supabase.table(TABLE_MESSAGES)
        .select("role, content, created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return res.data or []


def get_active_session_id(user_id: str) -> Optional[str]:
    """The conversation currently in progress, if any."""
    active = _fetch_active(user_id)
    return str(active["id"]) if active else None


def close_active_session(user_id: str) -> Optional[str]:
    """End the current conversation now, rather than waiting for the idle timer.

    Only flips the flag — the reaper writes the summary — so the request stays fast
    and a failed summary can be retried instead of losing the session.
    """
    session_id = get_active_session_id(user_id)
    if session_id:
        _mark_closing(session_id)
    return session_id


def fetch_messages(
    user_id: str,
    limit: int = 50,
    before: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Newest-first page of the user's messages.

    Scoped to one session when `session_id` is given — the chat screen shows the
    conversation in progress, and past ones live in Summaries. Without it, the query
    spans every session, which is what History would want.
    """
    q = (
        supabase.table(TABLE_MESSAGES)
        .select("id, session_id, role, content, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit + 1)
    )
    if session_id:
        q = q.eq("session_id", session_id)
    if before:
        q = q.lt("created_at", before)

    rows = q.execute().data or []
    has_more = len(rows) > limit
    page = rows[:limit]

    return {
        "messages": list(reversed(page)),  # oldest-first for rendering
        "has_more": has_more,
        "next_cursor": page[-1]["created_at"] if page and has_more else None,
    }


# ── The send path ────────────────────────────────────────────────────────────


async def send_message(user_id: str, content: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Persist the user's turn, get Lily's reply, persist that too."""
    session = await in_thread(resolve_session, user_id)
    session_id = str(session["id"])
    prior_count = int(session.get("message_count") or 0)

    user_msg = await in_thread(_insert_message, session_id, user_id, "user", content)

    history, system_messages = await asyncio.gather(
        in_thread(load_thread, session_id),
        in_thread(build_chat_messages, user_id),
    )
    messages = system_messages + [
        {"role": "assistant" if m["role"] == "lily" else "user", "content": m["content"]}
        for m in history
    ]

    try:
        reply_text = await complete(messages)
    except Exception as e:
        logger.error(f"Chat completion failed: {type(e).__name__}: {e}")
        # The user's message is already saved, so the turn isn't lost — but we do not
        # persist a fake reply, or the thread would fill with apologies on an outage.
        await in_thread(_touch_session, session_id, 1, prior_count)
        raise

    reply_msg = await in_thread(_insert_message, session_id, user_id, "lily", reply_text)
    await in_thread(_touch_session, session_id, 2, prior_count)

    return session_id, user_msg, reply_msg
