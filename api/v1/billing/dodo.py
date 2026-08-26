from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional, TypedDict

from fastapi import APIRouter, Depends, HTTPException, Request
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

@router.post("/billing/dodo/webhook")
async def dodo_webhook(request: Request):
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

    # ✅ IDEMPOTENCY
    event_id = payload.get("id")
    if event_id:
        exists = supabase.table("webhook_events").select("id").eq("id", event_id).execute()
        if exists.data:
            return {"received": True}
        supabase.table("webhook_events").insert({"id": event_id}).execute()

    event_type = payload.get("type")
    data = payload.get("data", {})

    if event_type in ("subscription.active", "subscription.created"):
        await _on_subscription_activated(data)

    elif event_type == "subscription.renewed":
        await _on_subscription_renewed(data)

    elif event_type == "subscription.cancelled":
        await _on_subscription_cancelled(data)

    elif event_type == "subscription.expired":
        await _on_subscription_expired(data)

    elif event_type in ("subscription.failed", "subscription.past_due"):
        await _on_subscription_failed(data)

    elif event_type == "subscription.on_hold":
        await _on_subscription_failed(data)  # treat on_hold same as past_due

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
# WEBHOOK HANDLERS
# ──────────────────────────────────────────────────────────────────────────────

async def _on_subscription_activated(data: dict):
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


async def _on_subscription_cancelled(data: dict):
    sub_id = data.get("subscription_id")

    supabase.table("dodo_subscriptions").update({
        "status": "cancelled",
    }).eq("dodo_subscription_id", sub_id).execute()


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