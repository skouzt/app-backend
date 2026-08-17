"""What this customer will be charged.

The app ships a copy of the catalogue for first paint, but the device only knows
its locale — which is a setting, not a location. The server sees the edge geo
header, which is the same signal Dodo's checkout keys off, so this endpoint is
what makes the displayed price agree with the charged one.

Display strings are rendered here rather than on the client: Hermes builds without
full ICU silently render "INR 999.00" instead of "₹999", and that bug is invisible
in a simulator with full ICU.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from core.billing.plans import TRIAL_DAYS, get_plan, yearly_saving_percent
from core.billing.region import is_trusted, resolve_billing_country
from core.rate_limit import READ, rate_limited

router = APIRouter()


class IntervalPrice(BaseModel):
    amount: float
    display: str
    period: str


class PricingResponse(BaseModel):
    country: str
    currency: str
    monthly: IntervalPrice
    yearly: IntervalPrice
    yearly_saving_percent: int
    trial_days: int
    # Lets the client tell "the server knows where you are" from "the server is
    # echoing the hint I sent it", which is the difference between a price we can
    # stand behind and a guess.
    source: str
    trusted: bool


@router.get("/billing/pricing", response_model=PricingResponse)
async def read_pricing(
    request: Request,
    country: Optional[str] = Query(
        None,
        min_length=2,
        # 4 rather than 2 so an older app build sending the legacy "INTL" tier gets
        # a price instead of a 422. resolve_billing_country maps it to the base.
        max_length=4,
        description="Device region hint, overridden whenever edge geo is available",
    ),
    _user_id: str = Depends(rate_limited("pricing", READ)),
):
    resolved, source = resolve_billing_country(request, country)

    monthly = get_plan("monthly", resolved)
    yearly = get_plan("yearly", resolved)

    return PricingResponse(
        country=monthly["country"],
        currency=monthly["currency"],
        monthly=IntervalPrice(**{k: monthly[k] for k in ("amount", "display", "period")}),
        yearly=IntervalPrice(**{k: yearly[k] for k in ("amount", "display", "period")}),
        yearly_saving_percent=yearly_saving_percent(resolved),
        trial_days=TRIAL_DAYS,
        source=source,
        trusted=is_trusted(source),
    )
