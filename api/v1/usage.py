from fastapi import APIRouter, Depends, HTTPException
from datetime import date, datetime
from calendar import monthrange
import traceback

from core.security import get_current_user_id
from db.supabase import supabase


router = APIRouter()

PLAN_CONFIG = {
    "clarity": {
        "sessions": 10,
        "minutes_per_session": 40,
        "price": 14,
    },
    "insight": {
        "sessions": 15,
        "minutes_per_session": 40,
        "price": 19,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_current_month_range():
    today = date.today()
    start = today.replace(day=1)
    last_day = monthrange(today.year, today.month)[1]
    end = today.replace(day=last_day)
    return start.isoformat(), end.isoformat()


# ──────────────────────────────────────────────────────────────────────────────
# CHECK USAGE
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/usage/check")
async def check_session_allowed(
    user_id: str = Depends(get_current_user_id),
):
    try:
        sub = (
            supabase.table("dodo_subscriptions")
            .select("plan_key, status, expires_at, trial_end")
            .eq("user_id", user_id)
            .in_("status", ["active", "trialing"])   # both pass the paywall
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        if not sub.data:
            return {
                "allowed": False,
                "reason": "no_active_plan",
                "sessions_used": 0,
                "sessions_limit": 0,
                "remaining_sessions": 0,
                "plan": None,
            }

        sub_row = sub.data[0]
        plan_key = sub_row.get("plan_key")
        status = sub_row.get("status")

        if plan_key not in PLAN_CONFIG:
            return {
                "allowed": False,
                "reason": "invalid_plan_config",
                "sessions_used": 0,
                "sessions_limit": 0,
                "remaining_sessions": 0,
                "plan": plan_key,
            }

        # ── Expiry check ──────────────────────────────────────────────────────
        # For trialing users: check trial_end.
        # For active users:   check expires_at.
        # Both fields map to expires_at in the DB (trial_end is also set there),
        # but being explicit here makes the logic readable and safe.
        check_date = (
            sub_row.get("trial_end")
            if status == "trialing"
            else sub_row.get("expires_at")
        )

        if check_date:
            try:
                expiry_dt = datetime.fromisoformat(check_date)
                if expiry_dt < datetime.utcnow():
                    return {
                        "allowed": False,
                        "reason": "trial_expired" if status == "trialing" else "expired",
                        "sessions_used": 0,
                        "sessions_limit": 0,
                        "remaining_sessions": 0,
                        "plan": plan_key,
                    }
            except Exception:
                pass

        # ── Session count ─────────────────────────────────────────────────────
        limit = PLAN_CONFIG[plan_key]["sessions"]
        month_start, month_end = get_current_month_range()

        usage = (
            supabase.table("daily_usage")
            .select("id")
            .eq("user_id", user_id)
            .gte("usage_date", month_start)
            .lte("usage_date", month_end)
            .execute()
        )

        sessions_used = len(usage.data or [])
        remaining = max(0, limit - sessions_used)

        return {
            "allowed": sessions_used < limit,
            "sessions_used": sessions_used,
            "sessions_limit": limit,
            "remaining_sessions": remaining,
            "plan": plan_key,
            "is_trialing": status == "trialing",
            "reason": None if sessions_used < limit else "limit_reached",
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to check usage: {str(e)}")


# ──────────────────────────────────────────────────────────────────────────────
# RECORD USAGE
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/usage/record")
async def record_session(
    user_id: str = Depends(get_current_user_id),
):
    try:
        today = date.today().isoformat()

        try:
            supabase.table("daily_usage").insert({
                "user_id": user_id,
                "usage_date": today,
                "sessions_count": 1,
                "minutes_used": 0,
                "created_at": datetime.utcnow().isoformat(),
            }).execute()

            return {"recorded": True}

        except Exception:
            return {"recorded": False, "message": "Already recorded for today"}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to record session: {str(e)}")