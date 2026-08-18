"""Patterns Lily notices across conversations.

Two kinds:

  · Deterministic — counted from session rows. Free, instant, and always true.
  · Thematic — one cheap LLM pass over recent *summaries* (never transcripts), to
    surface recurring subjects and how they've shifted.

Everything here is gated behind a minimum amount of history. An "insight" drawn from
two conversations is noise, and in a product like this one wrong observation costs
more trust than several right ones earn.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from db.supabase import supabase
from services.chat_service import in_thread
from services.llm import complete_json

# Nothing is shown before this much history exists.
MIN_SESSIONS = 5
# A theme must recur this often to count as a pattern rather than a coincidence.
MIN_THEME_COUNT = 3

THEME_SESSIONS = 20
THEME_CACHE_TTL = 6 * 3600

# user_id -> (computed_at, insights). Per-process; a restart just recomputes.
_theme_cache: Dict[str, tuple[float, List[Dict[str, Any]]]] = {}


def _parse(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _card(kind: str, icon: str, title: str, body: str) -> Dict[str, Any]:
    return {"kind": kind, "icon": icon, "title": title, "body": body}


def _fetch_sessions(user_id: str, limit: int = 60) -> List[Dict[str, Any]]:
    res = (
        supabase.table("therapy_sessions")
        .select("id, created_at, start_time, session_intensity, title, summary, status")
        .eq("user_id", user_id)
        .eq("status", "ended")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


# ── Deterministic ────────────────────────────────────────────────────────────


def _late_night(sessions: List[Dict[str, Any]], tz_offset_min: int) -> Optional[Dict[str, Any]]:
    """Timestamps are UTC; the app passes its offset so "late" means late *for them*."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    late = 0
    for s in sessions:
        started = _parse(s.get("start_time") or s.get("created_at"))
        if not started or started < cutoff:
            continue
        local_hour = (started + timedelta(minutes=tz_offset_min)).hour
        if local_hour >= 23 or local_hour < 5:
            late += 1

    if late < MIN_THEME_COUNT:
        return None
    return _card(
        "rhythm",
        "🌙",
        "Late nights",
        f"You've reached out late at night {late} times this month. "
        "Not a problem in itself — just something you might not have noticed.",
    )


def _cadence(sessions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    this_week = prev_week = 0
    for s in sessions:
        d = _parse(s.get("created_at"))
        if not d:
            continue
        age = (now - d).days
        if age < 7:
            this_week += 1
        elif age < 14:
            prev_week += 1

    if this_week + prev_week < MIN_THEME_COUNT:
        return None
    if this_week >= prev_week + 2:
        return _card(
            "rhythm", "🌱", "Showing up more",
            f"You've checked in {this_week} times this week, up from {prev_week}.",
        )
    if prev_week >= this_week + 2:
        return _card(
            "rhythm", "🍂", "A quieter week",
            f"{this_week} check-in{'s' if this_week != 1 else ''} this week, "
            f"after {prev_week} last week. However you want it is fine.",
        )
    return None


def _intensity_shift(sessions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    recent, earlier = [], []
    for s in sessions:
        d = _parse(s.get("created_at"))
        i = s.get("session_intensity")
        if not d or i is None:
            continue
        age = (now - d).days
        if age < 7:
            recent.append(int(i))
        elif age < 28:
            earlier.append(int(i))

    if len(recent) < 2 or len(earlier) < 3:
        return None

    r, e = sum(recent) / len(recent), sum(earlier) / len(earlier)
    if r - e >= 1:
        return _card(
            "change", "☀️", "Lighter lately",
            "Your recent conversations have landed easier than they did a few weeks ago.",
        )
    if e - r >= 1:
        return _card(
            "change", "🌊", "Heavier lately",
            "The last week has sat heavier than the few before it. Worth naming, if you want to.",
        )
    return None


# ── Thematic (one LLM pass over summaries) ───────────────────────────────────


async def _themes(user_id: str, sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cached = _theme_cache.get(user_id)
    if cached and time.time() - cached[0] < THEME_CACHE_TTL:
        return cached[1]

    usable = [s for s in sessions[:THEME_SESSIONS] if (s.get("summary") or "").strip()]
    if len(usable) < MIN_SESSIONS:
        return []

    digest = "\n".join(
        f"- {(s.get('title') or '').strip()}: {(s.get('summary') or '').strip()}" for s in usable
    )

    system = f"""\
Below are summaries of someone's recent conversations, newest first.

Find at most two things worth telling them. Prefer:
  · a subject that genuinely recurs (say how many conversations it appears in)
  · a change over time in how they relate to something

Rules:
  · Only report a recurring subject if it appears in at least {MIN_THEME_COUNT} conversations.
  · Observations, never diagnoses. No advice. Never claim certainty about their inner life.
  · Address them as "you". Warm and plain, the way a friend who'd been paying attention
    would say it. No clinical language.
  · If nothing genuinely recurs, return an empty list. Saying nothing beats inventing
    a pattern.

Return JSON only:
{{"insights": [{{"title": "3-5 words", "body": "1-2 sentences"}}]}}"""

    try:
        result = await complete_json(
            [{"role": "system", "content": system}, {"role": "user", "content": digest}],
            # Generous because this model reasons before answering, and reasoning is
            # billed and budgeted as completion tokens: a ~400-token answer needed
            # ~2.5x that in total. Called at most once per user per 6 hours, so the
            # headroom costs nothing.
            max_tokens=2000,
        )
        raw = result.get("insights") or []
        cards = [
            _card("theme", "💡", str(i.get("title") or "").strip(), str(i.get("body") or "").strip())
            for i in raw[:2]
            if str(i.get("body") or "").strip()
        ]
    except Exception as e:
        logger.warning(f"Theme insights failed for user: {type(e).__name__}: {e}")
        return []

    _theme_cache[user_id] = (time.time(), cards)
    return cards


# ── Entry point ──────────────────────────────────────────────────────────────


async def get_insights(user_id: str, tz_offset_min: int = 0) -> Dict[str, Any]:
    sessions = await in_thread(_fetch_sessions, user_id)

    if len(sessions) < MIN_SESSIONS:
        return {
            "ready": False,
            "sessions_needed": MIN_SESSIONS - len(sessions),
            "insights": [],
        }

    cards: List[Dict[str, Any]] = []
    cards.extend(await _themes(user_id, sessions))
    for fn in (
        lambda: _intensity_shift(sessions),
        lambda: _cadence(sessions),
        lambda: _late_night(sessions, tz_offset_min),
    ):
        card = fn()
        if card:
            cards.append(card)

    return {"ready": True, "sessions_needed": 0, "insights": cards[:4]}
