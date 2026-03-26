from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, TypedDict

import dodopayments
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
import structlog

from core.config import settings
from core.security import verify_clerk_token
from core.billing.dodo_client import DodoClient
from db.supabase import supabase

router = APIRouter()
dodo   = DodoClient()
logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# PLAN CONFIG
# ──────────────────────────────────────────────────────────────────────────────

PLAN_CONFIG: Dict[str, Dict[str, Any]] = {
    "clarity": {
        "name": "Clarity",
        "price_usd": 15,
        "sessions_per_month": 10,
        "minutes_per_session": 40,
        "trial_period_days": 1,
    },
    "insight": {
        "name": "Insight",
        "price_usd": 20,
        "sessions_per_month": 15,
        "minutes_per_session": 40,
        "trial_period_days": 1,
    },
}


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
    plan_key: str
    customer_name: Optional[str] = None
    return_url: Optional[str] = settings.DODO_DEFAULT_RETURN_URL


class SubscriptionStatusResponse(BaseModel):
    status: str
    plan: str
    expires_at: Optional[str] = None
    next_billing_date: Optional[str] = None
    trial_end: Optional[str] = None
    is_trialing: bool = False
    sessions_per_month: int
    minutes_per_session: int


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
    user: dict = Depends(verify_clerk_token),
):
    user_id = user.get("user_id") or user.get("sub")
    if not user_id:
        raise HTTPException(400, "User ID not found")

    if body.plan_key not in PLAN_CONFIG:
        raise HTTPException(400, "Invalid plan")

    email = await _get_user_email(user, user_id)
    if not email:
        raise HTTPException(400, "Email not found")

    plan_cfg = PLAN_CONFIG[body.plan_key]

    try:
        checkout_url, subscription_id = dodo.get_checkout_url(
            plan_key=body.plan_key,
            email=email,
            customer_name=body.customer_name or email.split("@")[0],
            return_url=body.return_url,
            user_id=user_id,
            trial_period_days=plan_cfg.get("trial_period_days", 0),
        )
    except Exception as e:
        logger.error("checkout_failed", exc_info=True)
        raise HTTPException(502, str(e))

    supabase.table("pending_verifications").insert({
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "plan_key": body.plan_key,
        "dodo_subscription_id": subscription_id,
        "created_at": datetime.utcnow().isoformat(),
    }).execute()

    return {
        "url": checkout_url,
        "subscription_id": subscription_id,
        "plan": plan_cfg["name"],
        "sessions_per_month": plan_cfg["sessions_per_month"],
        "minutes_per_session": plan_cfg["minutes_per_session"],
        "trial_period_days": plan_cfg.get("trial_period_days", 0),
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

    # STEP 1: already in DB (webhook already fired)
    existing = (
        supabase.table("dodo_subscriptions")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )

    if existing.data:
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
            "sessions_per_month": PLAN_CONFIG[plan]["sessions_per_month"],
            "minutes_per_session": PLAN_CONFIG[plan]["minutes_per_session"],
        }

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
    plan_cfg = PLAN_CONFIG.get(plan_key, {})
    trial_days = plan_cfg.get("trial_period_days", 0)

    if trial_days > 0:
        # Optimistically write trialing so user gets access immediately.
        # subscription.active webhook will upsert over this with verified status.
        trial_end = (datetime.utcnow() + timedelta(days=trial_days)).isoformat()

        _upsert_subscription(user_id, {
            "user_id": user_id,
            "dodo_subscription_id": sub_id,
            "plan_key": plan_key,
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
            "sessions_per_month": plan_cfg["sessions_per_month"],
            "minutes_per_session": plan_cfg["minutes_per_session"],
        }

    # No trial — wait for webhook
    return {"found": True, "activated": False, "pending": True}


# ── 4. SUBSCRIPTION STATUS ────────────────────────────────────────────────────

@router.get("/billing/me/subscription", response_model=SubscriptionStatusResponse)
async def get_my_subscription(user: dict = Depends(verify_clerk_token)):
    user_id = user.get("user_id") or user.get("sub")

    res = supabase.table("dodo_subscriptions").select("*").eq("user_id", user_id).execute()

    if not res.data:
        return SubscriptionStatusResponse(
            status="none",
            plan="none",
            sessions_per_month=0,
            minutes_per_session=0,
        )

    row = res.data[0]
    plan = row.get("plan_key")
    status = row.get("status")
    cfg = PLAN_CONFIG.get(plan, {})

    return SubscriptionStatusResponse(
        status=status,
        plan=plan,
        expires_at=row.get("expires_at"),
        next_billing_date=row.get("next_billing_date"),
        trial_end=row.get("trial_end"),
        is_trialing=status == "trialing",
        sessions_per_month=cfg.get("sessions_per_month", 0),
        minutes_per_session=cfg.get("minutes_per_session", 0),
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


# ──────────────────────────────────────────────────────────────────────────────
# WEBHOOK HANDLERS
# ──────────────────────────────────────────────────────────────────────────────

async def _on_subscription_activated(data: dict):
    user_id = data.get("metadata", {}).get("user_id")
    plan = data.get("metadata", {}).get("plan_key")
    sub_id = data.get("subscription_id")
    next_billing = data.get("next_billing_date")

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