"""Subscription gating.

Billing is a plain gate: you either have an active (or trialing) subscription or you
don't. There is no metering — the plans sell unlimited conversations, so counting
sessions or minutes would contradict what the paywall promises.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from db.supabase import supabase

ACTIVE_STATUSES = ("active", "trialing")

# A cancelled subscription is still *paid for* until the period it covers runs out.
# Revoking on the cancellation webhook would take away access someone already bought —
# and contradict the app, which promises "access until <date>".
GRANTING_STATUSES = ("active", "trialing", "cancelled")


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def get_subscription_state(user_id: str) -> Dict[str, Any]:
    """Resolve whether this user may talk to Lily right now."""
    res = (
        supabase.table("dodo_subscriptions")
        .select("plan_key, status, expires_at, trial_end")
        .eq("user_id", user_id)
        .in_("status", list(GRANTING_STATUSES))
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if not res.data:
        return {
            "allowed": False,
            "reason": "no_active_plan",
            "plan": None,
            "is_trialing": False,
            "is_cancelled": False,
        }

    row = res.data[0]
    status = row.get("status")
    plan = row.get("plan_key")
    is_trialing = status == "trialing"
    is_cancelled = status == "cancelled"

    # Trial rows carry their end in trial_end; paid rows in expires_at.
    expiry = _parse_ts(row.get("trial_end") if is_trialing else row.get("expires_at"))
    expired = bool(expiry and expiry < datetime.now(timezone.utc))

    if expired:
        reason = "trial_expired" if is_trialing else "expired"
    elif is_cancelled and not expiry:
        # Cancelled with no known end date — nothing to honour, so treat it as over.
        reason = "expired"
    else:
        reason = None

    return {
        "allowed": reason is None,
        "reason": reason,
        "plan": plan,
        "status": status,
        "is_trialing": is_trialing,
        # Cancelled but still inside the paid period: access continues, renewal won't.
        "is_cancelled": is_cancelled,
        "expires_at": expiry.isoformat() if expiry else None,
    }
