from datetime import datetime
from typing import TypedDict, cast
from db.supabase import supabase


class TherapySession(TypedDict):
    id: str
    user_id: str


def create_therapy_session(
    *,
    user_id: str,
    summary: str,
    dominant_emotion: str | None,
    session_intensity: int,
    started_at: datetime,
    ended_at: datetime,
) -> TherapySession:
    response = supabase.table("therapy_sessions").insert({
        "user_id": user_id,
        "summary": summary,
        "dominant_emotion": dominant_emotion,
        "session_intensity": session_intensity,
        "session_started_at": started_at.isoformat(),
        "session_ended_at": ended_at.isoformat(),
    }).execute()

    if not response.data:
        raise RuntimeError("Failed to create therapy session")

    return cast(TherapySession, response.data[0])
