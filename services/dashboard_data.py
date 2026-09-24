"""Data and rendering for the admin dashboard.

Shared deliberately: `scripts/gen_dashboard.py` writes a snapshot to disk and
`api/v1/admin.py` renders the same thing live behind the admin login. One
implementation so the two can never drift.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from db.supabase import supabase

TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "dashboard.html"

SAFETY = {
    "Yes, recently": "critical",
    "A few times before": "warning",
    "No": "none",
}


def split(value: str | None) -> list[str]:
    """Multi-select answers are stored as one comma-joined string."""
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def collect() -> dict:
    info = supabase.table("user_info").select("*").execute().data
    sessions = supabase.table("therapy_sessions").select("*").execute().data
    messages = supabase.table("messages").select("user_id,created_at,role").execute().data
    subs = supabase.table("dodo_subscriptions").select("*").execute().data

    # Counted here rather than in SQL so this stays four plain reads.
    n_sessions: dict[str, int] = {}
    n_messages: dict[str, int] = {}
    last_seen: dict[str, str] = {}
    for row in sessions:
        uid = row["user_id"]
        n_sessions[uid] = n_sessions.get(uid, 0) + 1
    for row in messages:
        uid = row["user_id"]
        n_messages[uid] = n_messages.get(uid, 0) + 1
        if row["created_at"] > last_seen.get(uid, ""):
            last_seen[uid] = row["created_at"]

    email_of = {r["user_id"]: r.get("email") for r in info}
    name_of = {r["user_id"]: (r.get("name") or "").strip() for r in info}

    users = [
        {
            "id": r["user_id"],
            "name": (r.get("name") or "").strip(),
            "email": r.get("email"),
            "joined": r["created_at"],
            "age": r.get("age"),
            "gender": r.get("gender"),
            "support_style": r.get("support_style"),
            "duration": r.get("Duration"),
            "coping": r.get("Coping_Style"),
            "network": r.get("Support_Network"),
            "timezone": r.get("timezone"),
            "difficulty": split(r.get("Current_Difficulty")),
            "impact": split(r.get("Daily_Impact")),
            "safety": SAFETY.get(r.get("Safety_Check"), "unknown"),
            "sessions": n_sessions.get(r["user_id"], 0),
            "messages": n_messages.get(r["user_id"], 0),
            "last_active": last_seen.get(r["user_id"]),
        }
        for r in info
    ]
    users.sort(key=lambda u: u["joined"], reverse=True)

    sess = [
        {
            "id": r["id"],
            "user": email_of.get(r["user_id"]) or name_of.get(r["user_id"]) or r["user_id"],
            "user_id": r["user_id"],
            "status": r.get("status"),
            "messages": r.get("message_count") or 0,
            "created": r["created_at"],
            "last_message": r.get("last_message_at"),
        }
        for r in sessions
    ]
    sess.sort(key=lambda s: s["created"], reverse=True)

    pays = [
        {
            "user": email_of.get(r["user_id"]) or r["user_id"],
            "user_id": r["user_id"],
            "ref": r.get("dodo_subscription_id"),
            "plan": r.get("plan_key"),
            "status": r.get("status"),
            "amount": float(r["amount"]) if r.get("amount") is not None else None,
            "currency": r.get("currency"),
            "region": r.get("region"),
            "created": r["created_at"],
            "next_billing": r.get("next_billing_date"),
        }
        for r in subs
    ]
    pays.sort(key=lambda p: p["created"], reverse=True)

    # Daily counts drive the activity chart. Built from real timestamps only —
    # days with no traffic are emitted as zero so the axis stays continuous.
    def day(ts: str) -> str:
        return ts[:10]

    buckets: dict[str, dict[str, int]] = {}
    for r in sessions:
        buckets.setdefault(day(r["created_at"]), {"sessions": 0, "messages": 0})["sessions"] += 1
    for r in messages:
        buckets.setdefault(day(r["created_at"]), {"sessions": 0, "messages": 0})["messages"] += 1
    for u in info:
        buckets.setdefault(day(u["created_at"]), {"sessions": 0, "messages": 0})

    daily = [{"date": d, **buckets[d]} for d in sorted(buckets)]

    return {"users": users, "sessions": sess, "payments": pays, "daily": daily}


def render(data: dict | None = None) -> str:
    """Inline the data into the template and return the finished page."""
    if data is None:
        data = collect()
    data.setdefault("built", date.today().isoformat())

    html = TEMPLATE.read_text()
    return html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
