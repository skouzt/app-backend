"""Subscription gate.

Kept at /usage/check because the app already calls it. It no longer meters anything:
the plans sell unlimited conversations, so the only question is whether the
subscription is live. The old daily_usage counting and /usage/record are gone.
"""

import traceback

from fastapi import APIRouter, Depends, HTTPException

from core.security import get_current_user_id
from services.chat_service import in_thread
from services.subscription_service import get_subscription_state

router = APIRouter()


@router.get("/usage/check")
async def check_access(user_id: str = Depends(get_current_user_id)):
    try:
        return await in_thread(get_subscription_state, user_id)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to check subscription")
