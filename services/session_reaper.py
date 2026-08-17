"""Closes idle conversations and writes their summary.

Runs as a background task. Two phases on purpose:

  active  → closing   (cheap, just a flag)
  closing → ended     (LLM call, then the summary lands)

Splitting them means the send path never waits on a summary, a crash mid-summary
leaves the session recoverable rather than stuck in 'active' forever, and a second
pass can't double-charge an LLM call for the same session.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from db.supabase import supabase
from services.chat_service import IDLE_MINUTES, TABLE_SESSIONS, load_thread
from services.llm import complete_json
from prompts.lily_chat import invalidate_prompt_cache

SWEEP_SECONDS = 60
BATCH = 20

# Matches the scale the app renders: 1 = crisis, 10 = at ease.
INTENSITY_GUIDE = """\
  1  Too much — crisis level, overwhelming distress
  2  Anxious
  3  Overwhelmed
  4  Strained
  5  Heavy
  6  Uneasy
  7  Neutral
  8  Light
  9  Okay
 10  At ease — calm, grounded"""

FALLBACK = ("A brief check-in", "You stopped by but didn't get far into it.", 7)


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


def _retire_idle() -> int:
    """active → closing for anything past the idle cutoff."""
    cutoff = _iso(_now() - timedelta(minutes=IDLE_MINUTES))
    res = (
        supabase.table(TABLE_SESSIONS)
        .select("id")
        .eq("status", "active")
        .lt("last_message_at", cutoff)
        .limit(BATCH)
        .execute()
    )
    ids = [r["id"] for r in (res.data or [])]
    for sid in ids:
        supabase.table(TABLE_SESSIONS).update(
            {"status": "closing", "updated_at": _iso(_now())}
        ).eq("id", sid).eq("status", "active").execute()
    return len(ids)


async def _summarise(thread: List[Dict[str, Any]]) -> Tuple[str, str, int]:
    """Ask for a title, a summary written to the user, and an intensity score."""
    transcript = "\n".join(
        f"{'Lily' if m['role'] == 'lily' else 'Them'}: {m['content']}" for m in thread
    )

    system = f"""\
Summarise this conversation for the person who had it. They will read this later as a
record of their own day, so write it to them, as "you".

Return JSON only:
{{"title": "...", "summary": "...", "session_intensity": <1-10>}}

title    3-6 words, plain and neutral. No advice, no diagnosis.
summary  2-3 sentences. What they talked about and how it seemed to sit with them.
         Warm, specific, never clinical. Do not invent anything not in the text.
session_intensity  the overall emotional tone, on this scale:
{INTENSITY_GUIDE}"""

    result = await complete_json(
        [{"role": "system", "content": system}, {"role": "user", "content": transcript}]
    )

    title = str(result.get("title") or "").strip() or FALLBACK[0]
    summary = str(result.get("summary") or "").strip()
    if not summary:
        raise ValueError("summary missing")

    try:
        intensity = int(result.get("session_intensity", 7))
    except (TypeError, ValueError):
        intensity = 7

    return title, summary, max(1, min(10, intensity))


async def _finalise_one(session: Dict[str, Any]) -> None:
    sid = str(session["id"])
    thread = load_thread(sid, limit=200)

    user_turns = [m for m in thread if m["role"] == "user"]
    if not user_turns or sum(len(m["content"]) for m in user_turns) < 15:
        title, summary, intensity = FALLBACK
    else:
        try:
            title, summary, intensity = await _summarise(thread)
        except Exception as e:
            # Still close it — a session stuck in 'closing' would block nothing but
            # would never appear in their history, which is worse than a plain summary.
            logger.error(f"Summary failed for {sid}: {type(e).__name__}: {e}")
            title, summary, intensity = FALLBACK

    now = _now()
    started = _parse(session.get("start_time"))
    last = _parse(session.get("last_message_at")) or now
    minutes = max(1, int((last - started).total_seconds() / 60)) if started else 1

    supabase.table(TABLE_SESSIONS).update(
        {
            "status": "ended",
            "title": title,
            "summary": summary,
            "session_intensity": intensity,
            "end_time": _iso(last),
            "duration_minutes": minutes,
            "updated_at": _iso(now),
        }
    ).eq("id", sid).eq("status", "closing").execute()

    # This session's summary is now part of what Lily knows, so the cached prompt
    # for that user is stale.
    if session.get("user_id"):
        invalidate_prompt_cache(str(session["user_id"]))

    logger.info(f"Session {sid} closed: {title!r}")


async def finalise_session(session_id: str) -> None:
    """Summarise one session by id, now. Used when the user ends a chat by hand.

    Safe to race with the reaper: the update is guarded on status='closing', so
    whichever finishes second is a no-op rather than a second LLM bill.
    """
    res = (
        supabase.table(TABLE_SESSIONS)
        .select("id, user_id, start_time, last_message_at")
        .eq("id", session_id)
        .eq("status", "closing")
        .limit(1)
        .execute()
    )
    if not res.data:
        return
    try:
        await _finalise_one(res.data[0])
    except Exception as e:
        # The periodic sweep will retry it.
        logger.error(f"Immediate finalise failed for {session_id}: {type(e).__name__}: {e}")


async def _finalise_pending() -> int:
    res = (
        supabase.table(TABLE_SESSIONS)
        .select("id, user_id, start_time, last_message_at")
        .eq("status", "closing")
        .limit(BATCH)
        .execute()
    )
    sessions = res.data or []
    for s in sessions:
        try:
            await _finalise_one(s)
        except Exception as e:
            logger.error(f"Finalise failed for {s.get('id')}: {type(e).__name__}: {e}")
    return len(sessions)


async def sweep() -> None:
    retired = _retire_idle()
    finalised = await _finalise_pending()
    if retired or finalised:
        logger.info(f"Reaper: {retired} retired, {finalised} summarised")


async def run_reaper() -> None:
    logger.info(f"Session reaper started (idle={IDLE_MINUTES}m, sweep={SWEEP_SECONDS}s)")
    while True:
        try:
            await sweep()
        except Exception as e:
            logger.error(f"Reaper sweep failed: {type(e).__name__}: {e}")
        await asyncio.sleep(SWEEP_SECONDS)
