from fastapi import APIRouter, Depends, HTTPException
from datetime import date, datetime
from calendar import monthrange
import traceback

from core.security import get_current_user_id
from db.supabase import supabase


router = APIRouter()

# ✅ PLAN CONFIG (matches your pricing)
PLAN_CONFIG = {
    "clarity": {
        "sessions": 10,
        "minutes_per_session": 40,
        "price":14,
    },
    "insight": {
        "sessions": 15,
        "minutes_per_session": 40,
        "price":19
    },
}


# -------------------------------
# Helpers
# -------------------------------

def get_current_month_range():
    today = date.today()
    start = today.replace(day=1)
    last_day = monthrange(today.year, today.month)[1]
    end = today.replace(day=last_day)
    return start.isoformat(), end.isoformat()


# -------------------------------
# CHECK USAGE
# -------------------------------

@router.get("/usage/check")
async def check_session_allowed(
    user_id: str = Depends(get_current_user_id),
):
    try:
        # 🔍 Fetch latest active subscription
        sub = (
            supabase.table("dodo_subscriptions")
            .select("plan_key, status, expires_at")
            .eq("user_id", user_id)
            .eq("status", "active")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        # ❌ No subscription
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

        # ❌ Invalid plan
        if plan_key not in PLAN_CONFIG:
            return {
                "allowed": False,
                "reason": "invalid_plan_config",
                "sessions_used": 0,
                "sessions_limit": 0,
                "remaining_sessions": 0,
                "plan": plan_key,
            }

        # ❌ Expiry check
        expires_at = sub_row.get("expires_at")
        if expires_at:
            try:
                expiry_dt = datetime.fromisoformat(expires_at)
                if expiry_dt < datetime.utcnow():
                    return {
                        "allowed": False,
                        "reason": "expired",
                        "sessions_used": 0,
                        "sessions_limit": 0,
                        "remaining_sessions": 0,
                        "plan": plan_key,
                    }
            except Exception:
                pass

        # ✅ Now safe to define limit
        limit = PLAN_CONFIG[plan_key]["sessions"]

        # 📊 Get monthly usage
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
            "reason": None if sessions_used < limit else "limit_reached",
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check usage: {str(e)}"
        )
        
# -------------------------------
# RECORD USAGE
# -------------------------------

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
            # already exists → ignore
            return {"recorded": False, "message": "Already recorded for today"}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record session: {str(e)}"
        )