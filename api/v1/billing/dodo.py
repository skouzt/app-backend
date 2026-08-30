from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional, TypedDict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel
import structlog

from core.config import settings
from core.security import verify_clerk_token
from core.billing.dodo_client import DodoClient
from core.billing.plans import TRIAL_DAYS, describe, get_plan, is_valid_interval
from core.billing.region import (
    REGION_INTL,
    is_trusted,
    region_for_country,
    resolve_billing_country,
)
from db.supabase import supabase
from services.push_service import send_to_user
from services.subscription_service import get_subscription_state

router = APIRouter()
dodo   = DodoClient()
logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# PLAN CONFIG
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# TYPES
# ──────────────────────────────────────────────────────────────────────────────

class DodoSubscriptionRow(TypedDict, total=False):
    id: str
    user_id: str
    dodo_subscription_id: str
    plan_key: str
    status: str
    expires_at: str
    next_billing_date: str
    trial_end: str


# ──────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE
# ──────────────────────────────────────────────────────────────────────────────

class CreateCheckoutRequest(BaseModel):
    plan_key: str                      # "monthly" | "yearly"
    region: Optional[str] = None       # display hint only; the server re-resolves
    customer_name: Optional[str] = None
    return_url: Optional[str] = settings.DODO_DEFAULT_RETURN_URL


class SubscriptionStatusResponse(BaseModel):
    status: str
    plan: str
    expires_at: Optional[str] = None
    next_billing_date: Optional[str] = None
    trial_end: Optional[str] = None
    is_trialing: bool = False
    region: Optional[str] = None
    currency: Optional[str] = None
    amount: Optional[float] = None
    period: Optional[str] = None
    unlimited: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

async def _get_user_email(user: dict, user_id: str) -> Optional[str]:
    email = user.get("email")
    if email:
        return str(email)

    res = supabase.table("user_info").select("email").eq("user_id", user_id).execute()
    if res.data and isinstance(res.data[0], dict):
        return str(res.data[0].get("email", ""))
    return None


def _upsert_subscription(user_id: str, payload: dict) -> None:
    existing = (
        supabase.table("dodo_subscriptions")
        .select("id")
        .eq("user_id", user_id)
        .execute()
    )

    if existing.data:
        supabase.table("dodo_subscriptions").update(payload).eq("user_id", user_id).execute()
    else:
        payload["id"] = str(uuid.uuid4())
        payload["created_at"] = datetime.utcnow().isoformat()
        supabase.table("dodo_subscriptions").insert(payload).execute()


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────

# ── 1. CREATE CHECKOUT ────────────────────────────────────────────────────────

@router.post("/billing/create-checkout")
async def create_dodo_checkout(
    body: CreateCheckoutRequest,
    request: Request,
    user: dict = Depends(verify_clerk_token),
):
    user_id = user.get("user_id") or user.get("sub")
    if not user_id:
        raise HTTPException(400, "User ID not found")

    if not is_valid_interval(body.plan_key):
        raise HTTPException(400, "Invalid plan")

    # The client's `region` is derived from device locale and is trivially spoofable,
    # so it only wins if the edge gave us nothing better.
    # Country-level now that every market has its own price; the old IN/INTL split
    # would have shown a Japanese customer the base USD price while Dodo charged ¥.
    region, source = resolve_billing_country(request, body.region)
    if not is_trusted(source) and body.region and body.region.upper() != region:
        logger.warning("region_hint_overridden", claimed=body.region, resolved=region)

    plan = get_plan(body.plan_key, region)
    if not plan:
        raise HTTPException(400, "No pricing for this plan and region")

    email = await _get_user_email(user, user_id)
    if not email:
        raise HTTPException(400, "Email not found")

    try:
        checkout_url, subscription_id = dodo.get_checkout_url(
            plan_key=body.plan_key,
            email=email,
            customer_name=body.customer_name or email.split("@")[0],
            return_url=body.return_url,
            user_id=user_id,
            region=region,
            region_trusted=is_trusted(source),
            trial_period_days=TRIAL_DAYS,
        )
    except ValueError as e:
        # Missing product mapping is a configuration problem, not a payment failure.
        logger.error("checkout_misconfigured", error=str(e))
        raise HTTPException(503, str(e))
    except Exception as e:
        logger.error("checkout_failed", exc_info=True)
        raise HTTPException(502, str(e))

    supabase.table("pending_verifications").insert({
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "plan_key": body.plan_key,
        "region": region,
        "dodo_subscription_id": subscription_id,
        "created_at": datetime.utcnow().isoformat(),
    }).execute()

    return {
        "url": checkout_url,
        "subscription_id": subscription_id,
        **describe(body.plan_key, region),
        "region_source": source,
    }


# ── 2. WEBHOOK ────────────────────────────────────────────────────────────────

def _is_duplicate(exc: Exception) -> bool:
    """Whether a failed insert was a primary-key collision.

    23505 is Postgres' unique_violation. It is matched on the message as well
    as the attribute because postgrest-py has moved the code between an
    attribute and a dict across versions, and mistaking a collision for a real
    outage would 500 on an event that was already handled correctly.
    """
    if getattr(exc, "code", None) == "23505":
        return True
    text = str(exc).lower()
    return "23505" in text or "duplicate key" in text


@router.post("/billing/dodo/webhook")
async def dodo_webhook(request: Request, background: BackgroundTasks):
    raw_body = await request.body()

    h = {k.lower(): v for k, v in request.headers.items()}

    valid = DodoClient.verify_webhook_signature(
        raw_body=raw_body,
        webhook_id=h.get("webhook-id", ""),
        webhook_timestamp=h.get("webhook-timestamp", ""),
        webhook_signature=h.get("webhook-signature", ""),
        secret=settings.DODO_WEBHOOK_SECRET,
    )

    if not valid:
        raise HTTPException(401, "Invalid signature")

    payload = json.loads(raw_body)

    # ── Idempotency ───────────────────────────────────────────────────────────
    # The insert *is* the claim. id is the primary key, so a redelivery arriving
    # while the first is still in flight loses the race inside the database —
    # a select-then-insert could interleave and let both through.
    event_id = payload.get("id")
    claimed = False

    if event_id:
        try:
            supabase.table("webhook_events").insert({"id": event_id}).execute()
            claimed = True
        except Exception as e:
            if not _is_duplicate(e):
                raise
            return {"received": True}

    event_type = payload.get("type")
    data = payload.get("data", {})

    try:
        if event_type in ("subscription.active", "subscription.created"):
            await _on_subscription_activated(data, background)

        elif event_type == "subscription.renewed":
            # Deliberately silent. A renewal is the absence of news, and a
            # monthly "we charged you" push is the kind of thing people disable
            # the whole notification channel over.
            await _on_subscription_renewed(data)

        elif event_type == "subscription.cancelled":
            await _on_subscription_cancelled(data, background)

        elif event_type == "subscription.expired":
            await _on_subscription_expired(data)

        elif event_type in ("subscription.failed", "subscription.past_due"):
            await _on_subscription_failed(data)

        elif event_type == "subscription.on_hold":
            await _on_subscription_failed(data)  # treat on_hold same as past_due

    except Exception:
        # Release the claim so Dodo's retry is allowed to run.
        #
        # Previously the claim was written before dispatch and never withdrawn,
        # so any handler failure was permanent: Dodo retried, the guard saw the
        # event id and returned early, and the event was dropped. On
        # subscription.active that is a paid subscription the database never
        # hears about.
        if claimed:
            try:
                supabase.table("webhook_events").delete().eq("id", event_id).execute()
            except Exception:
                logger.error("webhook_claim_release_failed", event_id=event_id)
        logger.exception("webhook_handler_failed", event_id=event_id, event_type=event_type)
        raise

    return {"received": True}


# ── 3. CHECK & ACTIVATE ───────────────────────────────────────────────────────

@router.post("/billing/check-and-activate")
async def check_and_activate_subscription(user: dict = Depends(verify_clerk_token)):
    user_id = user.get("user_id") or user.get("sub")

    # STEP 1: an existing subscription that actually grants access.
    #
    # This used to short-circuit on the mere existence of a row, which made
    # resubscribing impossible: a returning customer whose plan had lapsed hit this
    # branch, got "already_activated" describing the dead row, and never reached
    # step 2 — so their new checkout was never written. They could pay repeatedly
    # and stay locked out. Whether the row grants access is the question, not
    # whether it exists.
    existing = (
        supabase.table("dodo_subscriptions")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )

    if existing.data and get_subscription_state(user_id)["allowed"]:
        sub = existing.data[0]
        plan = sub.get("plan_key")
        status = sub.get("status")

        return {
            "found": True,
            "already_activated": True,
            "status": status,
            "is_trialing": status == "trialing",
            "plan": plan,
            "trial_end": sub.get("trial_end"),
            **describe(plan, sub.get("region") or REGION_INTL),
        }

    if existing.data:
        logger.info(
            "resubscribe_over_lapsed_row",
            user_id=user_id,
            old_status=existing.data[0].get("status"),
        )

    # STEP 2: webhook hasn't fired yet — check pending
    pv = (
        supabase.table("pending_verifications")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if not pv.data:
        return {"found": False}

    pending = pv.data[0]
    plan_key = pending.get("plan_key")
    sub_id = pending.get("dodo_subscription_id")
    region = pending.get("region") or REGION_INTL
    plan_cfg = get_plan(plan_key, region) or {}
    trial_days = TRIAL_DAYS if plan_cfg else 0

    if trial_days > 0:
        # Optimistically write trialing so user gets access immediately.
        # subscription.active webhook will upsert over this with verified status.
        trial_end = (datetime.utcnow() + timedelta(days=trial_days)).isoformat()

        _upsert_subscription(user_id, {
            "user_id": user_id,
            "dodo_subscription_id": sub_id,
            "plan_key": plan_key,
            "region": region,
            "currency": plan_cfg.get("currency"),
            "amount": plan_cfg.get("amount"),
            "status": "trialing",
            "expires_at": trial_end,
            "next_billing_date": trial_end,
            "trial_end": trial_end,
            "updated_at": datetime.utcnow().isoformat(),
        })

        return {
            "found": True,
            "activated": True,
            "status": "trialing",
            "is_trialing": True,
            "plan": plan_key,
            "trial_end": trial_end,
            **describe(plan_key, region),
        }

    # No trial — wait for webhook
    return {"found": True, "activated": False, "pending": True}


# ── 4. SUBSCRIPTION STATUS ────────────────────────────────────────────────────

@router.get("/billing/me/subscription", response_model=SubscriptionStatusResponse)
async def get_my_subscription(user: dict = Depends(verify_clerk_token)):
    user_id = user.get("user_id") or user.get("sub")

    res = supabase.table("dodo_subscriptions").select("*").eq("user_id", user_id).execute()

    if not res.data:
        return SubscriptionStatusResponse(status="none", plan="none")

    row = res.data[0]
    plan = row.get("plan_key")
    status = row.get("status")
    region = row.get("region") or REGION_INTL
    cfg = get_plan(plan, region) or {}

    return SubscriptionStatusResponse(
        status=status,
        plan=plan,
        expires_at=row.get("expires_at"),
        next_billing_date=row.get("next_billing_date"),
        trial_end=row.get("trial_end"),
        is_trialing=status == "trialing",
        region=region,
        currency=cfg.get("currency"),
        amount=cfg.get("amount"),
        period=cfg.get("period"),
    )


# ── 5. CANCEL ─────────────────────────────────────────────────────────────────

@router.post("/billing/cancel-subscription")
async def cancel_subscription(user: dict = Depends(verify_clerk_token)):
    user_id = user.get("user_id") or user.get("sub")

    res = supabase.table("dodo_subscriptions").select("*").eq("user_id", user_id).execute()

    if not res.data:
        raise HTTPException(404, "No subscription")

    sub_id = res.data[0].get("dodo_subscription_id")

    try:
        dodo._client.subscriptions.update(sub_id, status="cancelled")
    except Exception as e:
        raise HTTPException(502, str(e))

    return {"message": "Cancellation scheduled"}


# ── 6. PAYMENT HISTORY ────────────────────────────────────────────────────────

_CURRENCY_SYMBOL = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}


def _money(minor: Any, currency: str) -> str:
    """Dodo stores minor units; render them the way the app displays prices."""
    try:
        amount = (int(minor) or 0) / 100
    except (TypeError, ValueError):
        amount = 0.0
    sym = _CURRENCY_SYMBOL.get((currency or "").upper(), "")
    return f"{sym}{amount:,.2f}"


@router.get("/billing/payments")
async def list_payments(user: dict = Depends(verify_clerk_token)):
    """Payment history for the Subscription screen.

    Read live from Dodo rather than mirrored into Supabase — receipts are Dodo's
    record, and duplicating them means a second thing to keep in sync and reconcile.
    """
    user_id = user.get("user_id") or user.get("sub")

    res = (
        supabase.table("dodo_subscriptions")
        .select("dodo_subscription_id, plan_key, region")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not res.data:
        return {"payments": []}

    row = res.data[0]
    sub_id = row.get("dodo_subscription_id")
    if not sub_id:
        return {"payments": []}

    try:
        resp = dodo._client.payments.list(subscription_id=sub_id)
        items = list(getattr(resp, "items", None) or [])
    except Exception:
        # History is informational — never break the screen over it.
        logger.warning("payment_history_unavailable", sub_id=sub_id, exc_info=True)
        return {"payments": [], "unavailable": True}

    plan_label = f"Lily Unlimited · {str(row.get('plan_key') or '').title()}"

    payments = []
    for p in items:
        amount = getattr(p, "total_amount", None)
        currency = str(getattr(p, "currency", "") or "")
        status = str(getattr(p, "status", "") or "")
        created = getattr(p, "created_at", None)
        is_trial = (amount or 0) == 0

        payments.append(
            {
                "id": str(getattr(p, "payment_id", "") or ""),
                "date": str(created) if created else None,
                "description": "3-day free trial" if is_trial else plan_label,
                "amount": _money(amount, currency),
                "amount_minor": amount,
                "currency": currency,
                "status": "Free" if is_trial else (status.title() or "Paid"),
            }
        )

    return {"payments": payments}


# ──────────────────────────────────────────────────────────────────────────────
# BILLING NOTIFICATIONS
# ──────────────────────────────────────────────────────────────────────────────
#
# These run as background tasks, after the webhook has already answered Dodo.
# Sending inline would add the Expo round-trip plus the five-second wait for
# delivery receipts to every billing webhook, which is long enough for Dodo to
# time out and mark the endpoint unhealthy.
#
# Devices that have notifications switched off are filtered out by
# active_tokens(), so the Settings toggle governs these the same as any other.


def _subscription_row(sub_id: str) -> dict:
    """The stored row for a Dodo subscription.

    Cancellation and expiry webhooks carry only the subscription id — the
    user_id lives in checkout metadata, which is on the activation event alone.
    """
    res = (
        supabase.table("dodo_subscriptions")
        .select("user_id, status, expires_at")
        .eq("dodo_subscription_id", sub_id)
        .limit(1)
        .execute()
    )
    return (res.data or [{}])[0]


def _format_date(raw: Any) -> Optional[str]:
    """An ISO timestamp as "12 September", or None if it is unparseable."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    # Not %-d: that directive is platform-specific and would raise on Windows.
    return f"{dt.day} {dt:%B}"


async def _notify(user_id: Optional[str], title: str, body: str) -> None:
    """Push a billing update, swallowing every failure.

    A notification is the least important thing happening in this webhook. If
    Expo is down, the subscription state has still been recorded correctly and
    nothing here should surface as a failed webhook.
    """
    if not user_id:
        return
    try:
        await send_to_user(user_id, title=title, body=body, data={"type": "billing"})
    except Exception as e:
        logger.warning("billing_push_failed", user_id=user_id, error=type(e).__name__)


# ──────────────────────────────────────────────────────────────────────────────
# WEBHOOK HANDLERS
# ──────────────────────────────────────────────────────────────────────────────

async def _on_subscription_activated(data: dict, background: Optional[BackgroundTasks] = None):
    meta = data.get("metadata", {}) or {}
    user_id = meta.get("user_id")
    plan = meta.get("plan_key")
    sold_region = meta.get("region") or REGION_INTL
    sub_id = data.get("subscription_id")
    next_billing = data.get("next_billing_date")

    # The country the payment actually came from is the only authoritative signal —
    # it arrives too late to pick the product, but it catches a client that lied about
    # its region to reach the cheaper tier.
    paid_country = DodoClient.billing_country(data)
    paid_region = region_for_country(paid_country) if paid_country else None
    if paid_region and paid_region != sold_region:
        logger.warning(
            "billing_region_mismatch",
            sub_id=sub_id,
            sold_region=sold_region,
            paid_country=paid_country,
            paid_region=paid_region,
        )

    region = paid_region or sold_region
    plan_cfg = get_plan(str(plan), region) or {}

    # ── Detect trial via Dodo's workaround ───────────────────────────────────
    # Trial = exactly 1 payment exists AND its amount is 0
    is_trialing = False
    trial_end = None

    try:
        payments = dodo._client.payments.list(subscription_id=sub_id)
        payment_list = payments.items if hasattr(payments, "items") else []

        if len(payment_list) == 1 and getattr(payment_list[0], "total_amount", None) == 0:
            is_trialing = True
            trial_end = next_billing  # first real charge fires at next_billing_date
    except Exception:
        logger.warning("trial_payment_check_failed", sub_id=sub_id)

    # Dodo sends both subscription.created and subscription.active for the same
    # purchase, and they carry different event ids — so the idempotency guard on
    # the webhook does not collapse them. Notify only on the transition into a
    # paid state, or the user is congratulated twice for buying once.
    was_live = _subscription_row(sub_id).get("status") in ("active", "trialing")

    _upsert_subscription(user_id, {
        "user_id": user_id,
        "dodo_subscription_id": sub_id,
        "plan_key": plan,
        "region": region,
        "currency": plan_cfg.get("currency"),
        "amount": plan_cfg.get("amount"),
        "region_source": "payment-country" if paid_region else "checkout-metadata",
        "status": "trialing" if is_trialing else "active",
        "expires_at": trial_end if is_trialing else next_billing,
        "next_billing_date": next_billing,
        "trial_end": trial_end,
        "updated_at": datetime.utcnow().isoformat(),
    })

    supabase.table("pending_verifications")\
        .delete()\
        .eq("dodo_subscription_id", sub_id)\
        .execute()

    if background is None or was_live:
        return

    if is_trialing:
        until = _format_date(trial_end)
        body = (
            f"You have everything until {until}. No rush — talk when you're ready."
            if until
            else "You have everything for now. No rush — talk when you're ready."
        )
        background.add_task(_notify, user_id, "Your trial has started", body)
    else:
        background.add_task(
            _notify,
            user_id,
            "You're all set",
            "Your subscription is active. I'm here whenever you want to talk.",
        )


async def _on_subscription_renewed(data: dict):
    sub_id = data.get("subscription_id")
    next_billing = data.get("next_billing_date")

    # Covers both trial→paid conversion and regular renewal
    supabase.table("dodo_subscriptions").update({
        "status": "active",
        "expires_at": next_billing,
        "next_billing_date": next_billing,
        "trial_end": None,              # clear trial on conversion
    }).eq("dodo_subscription_id", sub_id).execute()


async def _on_subscription_cancelled(data: dict, background: Optional[BackgroundTasks] = None):
    sub_id = data.get("subscription_id")

    # Read before the update: this is the only place the user_id and the paid-up
    # date are available, and the write below would not change them anyway.
    row = _subscription_row(sub_id)

    supabase.table("dodo_subscriptions").update({
        "status": "cancelled",
    }).eq("dodo_subscription_id", sub_id).execute()

    if background is None or row.get("status") == "cancelled":
        return

    # Cancellation is scheduled, not immediate — /billing/cancel-subscription
    # answers "Cancellation scheduled" and access runs to the end of the paid
    # period, with a separate subscription.expired event when it actually ends.
    # Telling someone their access has stopped while they can still use the app
    # would be wrong, so the date does the work.
    until = _format_date(row.get("expires_at"))
    body = (
        f"You'll have full access until {until}. I'll be here if you come back."
        if until
        else "You'll keep full access until the end of your billing period. "
        "I'll be here if you come back."
    )
    background.add_task(_notify, row.get("user_id"), "Subscription cancelled", body)


async def _on_subscription_expired(data: dict):
    sub_id = data.get("subscription_id")

    supabase.table("dodo_subscriptions").update({
        "status": "expired",
    }).eq("dodo_subscription_id", sub_id).execute()


async def _on_subscription_failed(data: dict):
    sub_id = data.get("subscription_id")

    supabase.table("dodo_subscriptions").update({
        "status": "past_due",
    }).eq("dodo_subscription_id", sub_id).execute()