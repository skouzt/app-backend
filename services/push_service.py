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

import asyncio
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence

import httpx
from loguru import logger

from db.supabase import supabase

TABLE = "push_tokens"

EXPO_SEND_URL = "https://exp.host/--/api/v2/push/send"
EXPO_RECEIPTS_URL = "https://exp.host/--/api/v2/push/getReceipts"

# Expo rejects a request carrying more than 100 messages.
BATCH_SIZE = 100

# PostgREST caps a response at 1000 rows and says nothing about it. Reading a
# broadcast audience without paging would silently address the first thousand
# people and drop the rest — with no error anywhere to notice.
PAGE_SIZE = 1000

REQUEST_TIMEOUT = 30.0

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


# --- Sending ------------------------------------------------------------------


def all_active_tokens() -> List[str]:
    """Every reachable device, paged.

    PostgREST returns at most 1000 rows per request. Paging is not an
    optimisation here: without it a broadcast quietly reaches the first page and
    everyone beyond it is simply missed.
    """
    tokens: List[str] = []
    offset = 0

    while True:
        result = (
            supabase.table(TABLE)
            .select("token")
            .eq("enabled", True)
            .order("token")
            .range(offset, offset + PAGE_SIZE - 1)
            .execute()
        )
        page = [row["token"] for row in (result.data or [])]
        tokens.extend(page)

        if len(page) < PAGE_SIZE:
            return tokens
        offset += PAGE_SIZE


def _chunks(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def forget_tokens(tokens: Iterable[str]) -> None:
    """Drop devices the provider says no longer exist."""
    dead = list(tokens)
    if not dead:
        return
    supabase.table(TABLE).delete().in_("token", dead).execute()
    logger.info("removed {} unreachable push token(s)", len(dead))


async def _post(client: httpx.AsyncClient, url: str, payload: Any) -> Dict[str, Any]:
    response = await client.post(
        url,
        json=payload,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    response.raise_for_status()
    return response.json()


async def _collect_receipts(
    client: httpx.AsyncClient, ticket_to_token: Dict[str, str]
) -> List[str]:
    """Return the tokens the provider reports as gone.

    A send answering `ok` only means Expo accepted the message; delivery is
    reported later, per ticket. Skipping this step is why dead tokens accumulate
    — the send path never sees a failure, so nothing is ever cleaned up.
    """
    unreachable: List[str] = []
    ticket_ids = list(ticket_to_token)

    for batch in _chunks(ticket_ids, BATCH_SIZE):
        try:
            body = await _post(client, EXPO_RECEIPTS_URL, {"ids": list(batch)})
        except Exception as exc:
            # A receipt we cannot read is not evidence the device is gone.
            logger.warning("push receipts unavailable: {}", type(exc).__name__)
            continue

        for ticket_id, receipt in (body.get("data") or {}).items():
            if receipt.get("status") == "error":
                error = (receipt.get("details") or {}).get("error")
                if error == "DeviceNotRegistered":
                    unreachable.append(ticket_to_token[ticket_id])
                else:
                    logger.warning("push receipt error: {}", error)

    return unreachable


async def send_to_tokens(
    tokens: Sequence[str], *, title: str, body: str, data: Dict[str, Any] | None = None
) -> Dict[str, int]:
    """Deliver one message to a list of devices.

    Returns counts rather than raising: a broadcast that fails for some devices
    has still succeeded for the rest, and the caller wants to know both.
    """
    if not tokens:
        return {"sent": 0, "failed": 0, "removed": 0}

    messages_sent = 0
    failed = 0
    ticket_to_token: Dict[str, str] = {}
    unreachable: List[str] = []

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for batch in _chunks(tokens, BATCH_SIZE):
            payload = [
                {
                    "to": token,
                    "title": title,
                    "body": body,
                    # Android files notifications by channel; without this one
                    # they land in the default channel with no heads-up banner.
                    "channelId": "lily",
                    **({"data": data} if data else {}),
                }
                for token in batch
            ]

            try:
                response = await _post(client, EXPO_SEND_URL, payload)
            except Exception as exc:
                failed += len(batch)
                logger.warning("push batch failed: {}", type(exc).__name__)
                continue

            # Tickets come back in request order, which is the only way to know
            # which token a given result belongs to.
            for token, ticket in zip(batch, response.get("data") or []):
                if ticket.get("status") == "ok" and ticket.get("id"):
                    ticket_to_token[ticket["id"]] = token
                    messages_sent += 1
                    continue

                failed += 1
                error = (ticket.get("details") or {}).get("error")
                if error == "DeviceNotRegistered":
                    unreachable.append(token)
                else:
                    logger.warning("push rejected: {}", error or ticket.get("message"))

            # Expo throttles around 600 messages a second. One pause per batch
            # keeps a large broadcast under that without pacing every message.
            await asyncio.sleep(0.2)

        if ticket_to_token:
            # Receipts are not ready the instant a send returns.
            await asyncio.sleep(5)
            unreachable.extend(await _collect_receipts(client, ticket_to_token))

    if unreachable:
        await asyncio.to_thread(forget_tokens, set(unreachable))

    return {"sent": messages_sent, "failed": failed, "removed": len(set(unreachable))}


async def send_to_user(
    user_id: str, *, title: str, body: str, data: Dict[str, Any] | None = None
) -> Dict[str, int]:
    tokens = await asyncio.to_thread(active_tokens, user_id)
    return await send_to_tokens(tokens, title=title, body=body, data=data)


async def send_to_all(
    *, title: str, body: str, data: Dict[str, Any] | None = None
) -> Dict[str, int]:
    """Broadcast to every device that has notifications enabled."""
    tokens = await asyncio.to_thread(all_active_tokens)
    logger.info("broadcasting to {} device(s)", len(tokens))
    return await send_to_tokens(tokens, title=title, body=body, data=data)
