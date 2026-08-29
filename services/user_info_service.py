from typing import Optional, Dict, Any, cast
from db.supabase import supabase


def fetch_user_info(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch complete user background info from user_info table
    used for LLM context.
    """

    result = (
        supabase
        .table("user_info")
        .select("*")  # Fetch all columns
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

    if not result.data or not isinstance(result.data[0], dict):
        return None

    # Cast to satisfy type checker - we know it's a dict
    return cast(Dict[str, Any], result.data[0])

# ── Timezone ─────────────────────────────────────────────────────────────────
#
# Sessions are keyed on the user's local calendar day, so this is what decides
# where one day ends. Stored per user rather than derived per request because
# the reaper closes yesterday's sessions without anyone being online to ask.

from datetime import date, datetime, timezone as _tz  # noqa: E402
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # noqa: E402

from loguru import logger  # noqa: E402


def set_timezone(user_id: str, tz_name: str) -> None:
    """Record the device's IANA zone, e.g. "Asia/Kolkata"."""
    supabase.table("user_info").update({"timezone": tz_name}).eq("user_id", user_id).execute()


def zone_for(tz_name: Optional[str]) -> _tz:
    """Resolve a stored zone name, falling back to UTC.

    An unknown or missing name must not raise: a user who has not sent a
    timezone yet still needs their messages to land in some day, and UTC is the
    only answer available without guessing.
    """
    if not tz_name:
        return _tz.utc
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning("unknown timezone {!r}, falling back to UTC", tz_name)
        return _tz.utc


def local_today(tz_name: Optional[str], now: Optional[datetime] = None) -> date:
    """The calendar date it currently is for this user."""
    moment = now or datetime.now(_tz.utc)
    return moment.astimezone(zone_for(tz_name)).date()
