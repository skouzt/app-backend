from typing import List, Dict, Any, cast
from supabase import Client
from datetime import datetime

from db.supabase import supabase

def fetch_recent_sessions(user_id: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Fetch the most recent therapy sessions for a user.

    Args:
        user_id: The UUID of the user
        limit: Number of recent sessions to retrieve (default: 3)

    Returns:
        List of session dictionaries, ordered by date descending
    """
    try:
        response = (
            supabase
            .table("therapy_sessions")
            .select("id, date, summary, title, session_intensity")
            .eq("user_id", user_id)
            .order("date", desc=True)
            .limit(limit)
            .execute()
        )

        if not response.data:
            return []

        sessions: List[Dict[str, Any]] = []

        for raw in response.data:
            row = cast(dict[str, Any], raw)

            # ---- Date formatting (safe) ----
            date_str = ""
            raw_date = row.get("date")
            if isinstance(raw_date, str):
                try:
                    date_obj = datetime.fromisoformat(raw_date)
                    date_str = date_obj.strftime("%B %d")
                except ValueError:
                    date_str = ""

            sessions.append({
                "id": row.get("id"),
                "date": date_str,
                "title": str(row.get("title") or "").strip(),
                "summary": str(row.get("summary") or "").strip(),
                "session_intensity": (
                    int(row["session_intensity"])
                    if isinstance(row.get("session_intensity"), (int, float))
                    else None
                ),
            })

        return sessions

    except Exception as e:
        return []
