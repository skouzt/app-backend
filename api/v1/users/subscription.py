# api/v1/users/subscription.py

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from pydantic import BaseModel
import logging

from core.security import verify_clerk_token
from db.supabase import supabase

router = APIRouter()
logger = logging.getLogger(__name__)


# -------------------------------
# Response Model
# -------------------------------

class SubscriptionResponse(BaseModel):
    plan: str
    status: str
    expires_at: Optional[str] = None


# -------------------------------
# Endpoint
# -------------------------------

@router.get("/me/subscription", response_model=SubscriptionResponse)
async def get_my_subscription(user: dict = Depends(verify_clerk_token)):
    """
    Returns the current user's subscription.
    If no row exists, returns plan='none'.
    """

    clerk_id = user["sub"]

    try:
        result = (
            supabase
            .table("dodo_subscriptions")
            .select("plan_key, status, expires_at, next_billing_date")
            .eq("user_id", clerk_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as e:
        logger.error("subscription_fetch_failed", extra={"user_id": clerk_id}, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch subscription",
        )

    # NO ROW FOUND
    if not result.data:
        logger.info("subscription_not_found", extra={"user_id": clerk_id})
        return SubscriptionResponse(
            plan="none",
            status="inactive",
            expires_at=None,
        )

    row = result.data[0]

    # Log successful fetch without exposing sensitive data

    return SubscriptionResponse(
        plan=row.get("plan_key", "none"),
        status=row.get("status", "inactive"),
        expires_at=row.get("expires_at"),
    )