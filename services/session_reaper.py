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
from services.chat_service import TABLE_SESSIONS, load_thread
from services.user_info_service import local_today
from services.llm import complete_json
from prompts.lily_chat import invalidate_prompt_cache
from services.memory_service import update_from_conversation

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


# How long the conversation must be quiet before its summary is written or
# rewritten. Short enough that a user who chats for five minutes sees today's
# entry soon after; long enough not to summarise mid-thought.
IDLE_MINUTES = 10


def _refresh_stale_summaries_query() -> List[Dict[str, Any]]:
    """Open sessions that have gone quiet and contain something new.

    The message-count watermark is what makes this cheap to run every sweep: a
    session whose summary already covers every message it holds is skipped
    without touching the model.
    """
    cutoff = _iso(_now() - timedelta(minutes=IDLE_MINUTES))
    res = (
        supabase.table(TABLE_SESSIONS)
        .select("id, user_id, start_time, last_message_at, message_count, summarised_message_count")
        .eq("status", "active")
        .lt("last_message_at", cutoff)
        .limit(BATCH)
        .execute()
    )
    return [
        r
        for r in (res.data or [])
        if int(r.get("message_count") or 0) > int(r.get("summarised_message_count") or 0)
    ]


async def _refresh_one(session: Dict[str, Any]) -> None:
    """Rewrite the summary of a session that is still open.

    The whole day's thread is summarised, not just the new messages: the row
    holds one summary for one day, and appending would produce a list of
    fragments rather than an account of the day.
    """
    sid = str(session["id"])
    thread = load_thread(sid, limit=200)

    user_turns = [m for m in thread if m["role"] == "user"]
    if not user_turns or sum(len(m["content"]) for m in user_turns) < 15:
        return  # Nothing said yet worth describing.

    try:
        title, summary, intensity = await _summarise(thread)
    except Exception as e:
        # Leave the watermark alone so the next sweep tries again.
        logger.error(f"Summary refresh failed for {sid}: {type(e).__name__}: {e}")
        return

    now = _now()
    started = _parse(session.get("start_time"))
    last = _parse(session.get("last_message_at")) or now
    minutes = max(1, int((last - started).total_seconds() / 60)) if started else 1

    supabase.table(TABLE_SESSIONS).update(
        {
            "title": title,
            "summary": summary,
            "session_intensity": intensity,
            "duration_minutes": minutes,
            # Deliberately not touching status: the day is not over, and the user
            # may well pick the conversation back up in an hour.
            "summarised_message_count": int(session.get("message_count") or 0),
            "updated_at": _iso(now),
        }
    ).eq("id", sid).eq("status", "active").execute()

    if session.get("user_id"):
        invalidate_prompt_cache(str(session["user_id"]))

    logger.info(f"Session {sid} summary refreshed: {title!r}")


async def _refresh_open_summaries() -> int:
    sessions = await asyncio.to_thread(_refresh_stale_summaries_query)
    for s in sessions:
        try:
            await _refresh_one(s)
        except Exception as e:
            logger.error(f"Refresh failed for {s.get('id')}: {type(e).__name__}: {e}")
    return len(sessions)


def _retire_idle() -> int:
    """active → closing for any session whose day has ended.

    The cutoff is a date, not an idle window: one session covers one local day,
    so a gap in the conversation is not the end of anything. A session closes
    when the user's own date moves on — which is why each row is compared
    against its owner's timezone rather than a single server-wide cutoff. UTC
    midnight is 5:30am in India, and closing there would cut the day in half for
    most of these users.
    """
    res = (
        supabase.table(TABLE_SESSIONS)
        .select("id, user_id, date")
        .eq("status", "active")
        .limit(BATCH)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return 0

    # One lookup for the batch rather than one per session.
    user_ids = list({r["user_id"] for r in rows})
    info = (
        supabase.table("user_info")
        .select("user_id, timezone")
        .in_("user_id", user_ids)
        .execute()
    )
    zones = {r["user_id"]: r.get("timezone") for r in (info.data or [])}

    now = _now()
    stale = [
        r["id"]
        for r in rows
        if str(r.get("date") or "") < local_today(zones.get(r["user_id"]), now).isoformat()
    ]

    for sid in stale:
        supabase.table(TABLE_SESSIONS).update(
            {"status": "closing", "updated_at": _iso(_now())}
        ).eq("id", sid).eq("status", "active").execute()
    return len(stale)


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

    # The day's summary was very likely written already by the refresh pass. If
    # nothing has been said since, closing the session is a status change — there
    # is no reason to pay for the same summary twice.
    if (
        session.get("summary")
        and int(session.get("message_count") or 0) <= int(session.get("summarised_message_count") or 0)
    ):
        now = _now()
        started = _parse(session.get("start_time"))
        last = _parse(session.get("last_message_at")) or now
        minutes = max(1, int((last - started).total_seconds() / 60)) if started else 1
        supabase.table(TABLE_SESSIONS).update(
            {
                "status": "ended",
                "end_time": _iso(last),
                "duration_minutes": minutes,
                "updated_at": _iso(now),
            }
        ).eq("id", sid).eq("status", "closing").execute()
        logger.info(f"Session {sid} closed with its existing summary")
        return

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
            "summarised_message_count": int(session.get("message_count") or 0),
            "updated_at": _iso(now),
        }
    ).eq("id", sid).eq("status", "closing").execute()

    # The day is over, so fold it into what Lily remembers. Done here rather
    # than on every summary refresh: memory is for future conversations, so one
    # extraction per person per day keeps the cost flat however much they wrote.
    user_id = session.get("user_id")
    if user_id:
        try:
            await update_from_conversation(str(user_id), thread, session_id=sid)
        except Exception as e:
            # A failed extraction leaves memory as it was; the session is still
            # closed and summarised, which is the part the user sees.
            logger.error(f"Memory update failed for {sid}: {type(e).__name__}: {e}")

        # Summary and memory are both part of what Lily knows, so the cached
        # prompt for that user is stale.
        invalidate_prompt_cache(str(user_id))

    logger.info(f"Session {sid} closed: {title!r}")


async def finalise_session(session_id: str) -> None:
    """Summarise one session by id, now. Used when the user ends a chat by hand.

    Safe to race with the reaper: the update is guarded on status='closing', so
    whichever finishes second is a no-op rather than a second LLM bill.
    """
    res = (
        supabase.table(TABLE_SESSIONS)
        .select(
            "id, user_id, start_time, last_message_at, message_count, "
            "summarised_message_count, summary"
        )
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
        .select(
            "id, user_id, start_time, last_message_at, message_count, "
            "summarised_message_count, summary"
        )
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
    # Refresh first: a session that is about to be retired for the day is worth
    # summarising as an open session, so the closing pass usually finds its work
    # already done.
    refreshed = await _refresh_open_summaries()
    retired = _retire_idle()
    finalised = await _finalise_pending()
    if refreshed or retired or finalised:
        logger.info(
            f"Reaper: {refreshed} refreshed, {retired} retired, {finalised} closed"
        )


async def run_reaper() -> None:
    logger.info(
        f"Session reaper started (one session per local day, "
        f"summary refresh after {IDLE_MINUTES}m idle, sweep={SWEEP_SECONDS}s)"
    )
    while True:
        try:
            await sweep()
        except Exception as e:
            logger.error(f"Reaper sweep failed: {type(e).__name__}: {e}")
        await asyncio.sleep(SWEEP_SECONDS)
