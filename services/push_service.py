"""Push notification device registry.

Only the token side lives here: which devices belong to whom, and whether they
want to hear anything. Sending is a separate concern and deliberately not in
this module yet.

Expo tokens look like `ExponentPushToken[xxxxxxxxxxxxxxxxxxxxxx]`. They are
issued per app install, so they change on reinstall, and they are portable
between users when a device is handed over — which is why the token, not the
user, is the identity of a row.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from loguru import logger

from db.supabase import supabase

TABLE = "push_tokens"

# Expo issues both its own token format and, for bare workflows, raw FCM/APNs
# device tokens. Only the Expo form is accepted here because that is what the
# send path will address; anything else is a client bug worth failing loudly on
# rather than storing an address nothing can deliver to.
_EXPO_TOKEN = re.compile(r"^Expo(nent)?PushToken\[[A-Za-z0-9_-]+\]$")


def is_valid_token(token: str) -> bool:
    return bool(_EXPO_TOKEN.match(token or ""))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def register_token(user_id: str, token: str, platform: str) -> None:
    """Attach a device to this user, or move it if it belonged to another.

    Upsert on the token: reinstalling, signing out and back in, or handing the
    phone to someone else all converge on one row rather than leaving a stale
    one that would keep receiving another person's notifications.
    """
    supabase.table(TABLE).upsert(
        {
            "token": token,
            "user_id": user_id,
            "platform": platform,
            # Registering is an explicit act — it follows a granted permission
            # prompt — so it also clears any previous opt-out on this device.
            "enabled": True,
            "updated_at": _now(),
        },
        on_conflict="token",
    ).execute()

    logger.info("push token registered platform={} user={}", platform, user_id)


def unregister_token(user_id: str, token: str) -> None:
    """Forget a device. Used on sign-out.

    Scoped by user_id as well as token so a stale client cannot delete a
    registration that has since moved to somebody else's account.
    """
    supabase.table(TABLE).delete().eq("token", token).eq("user_id", user_id).execute()
    logger.info("push token removed user={}", user_id)


def set_enabled(user_id: str, enabled: bool) -> None:
    """Apply the Settings toggle to every device this user has registered.

    The preference is the person's, not the phone's: turning notifications off
    on a tablet should not leave them arriving on a handset.
    """
    supabase.table(TABLE).update({"enabled": enabled, "updated_at": _now()}).eq(
        "user_id", user_id
    ).execute()

    logger.info("push notifications {} user={}", "enabled" if enabled else "disabled", user_id)


def get_state(user_id: str) -> Dict[str, Any]:
    """What Settings needs to render: are we on, and is any device registered."""
    result = (
        supabase.table(TABLE).select("token, enabled").eq("user_id", user_id).execute()
    )
    rows: List[Dict[str, Any]] = result.data or []

    return {
        "devices": len(rows),
        # Off when nothing is registered, so a fresh install shows the toggle
        # off rather than promising notifications it cannot deliver.
        "enabled": any(row.get("enabled") for row in rows),
    }


def active_tokens(user_id: str) -> List[str]:
    """Addresses to send to. Used by the sending path, once it exists."""
    result = (
        supabase.table(TABLE)
        .select("token")
        .eq("user_id", user_id)
        .eq("enabled", True)
        .execute()
    )
    return [row["token"] for row in (result.data or [])]
