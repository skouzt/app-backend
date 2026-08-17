"""Patterns Lily has noticed — the Summaries screen's insight cards."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel

from core.rate_limit import INSIGHTS, rate_limited
from services.chat_service import in_thread
from services.insights_service import get_insights
from services.subscription_service import get_subscription_state

router = APIRouter()


class InsightCard(BaseModel):
    kind: str
    icon: str
    title: str
    body: str


class InsightsResponse(BaseModel):
    ready: bool
    sessions_needed: int
    insights: List[InsightCard]


@router.get("/insights", response_model=InsightsResponse)
async def read_insights(
    tz_offset: int = Query(
        0,
        ge=-840,
        le=840,
        description="Device UTC offset in minutes, so 'late at night' means late for them",
    ),
    user_id: str = Depends(rate_limited("insights", INSIGHTS)),
):
    # Theme extraction is an LLM call behind a 6h cache. Cheap per user, but it
    # was the one paid path any signed-up account could reach without a plan.
    if not (await in_thread(get_subscription_state, user_id))["allowed"]:
        raise HTTPException(status_code=402, detail="Subscription required")

    try:
        data = await get_insights(user_id, tz_offset_min=tz_offset)
    except Exception as e:
        logger.error(f"insights failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Could not load insights")

    return InsightsResponse(**data)
