"""Plan catalogue.

Lily sells one thing — unlimited conversations — billed monthly or yearly. There is no
metering, so a plan carries a price and nothing else.

There is ONE Dodo product per interval. Each product carries per-country prices via
Dodo's Localized Pricing, so Dodo picks the currency from the customer's real billing
country at checkout. That makes Dodo — not our region guess — the authority on what
someone is charged, which is exactly where that decision belongs.

Prices here are read from `core.billing.catalog`, the same table that
`scripts/sync_dodo_prices.py` pushes to Dodo. That is the point: display and charge
now come from one source, so they cannot drift the way they did when India showed a
PPP-discounted USD price the app knew nothing about.

`region` is historical naming. It holds an ISO-3166 country code where we have one,
and the legacy "INTL" for rows written before the catalogue existed; both resolve.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.billing.catalog import BASE, by_country, format_amount
from core.config import settings

INTERVALS = ("monthly", "yearly")

TRIAL_DAYS = 3

_PERIOD = {"monthly": "month", "yearly": "year"}


def _product_map() -> Dict[str, str]:
    # Read at call time, not import time, so a missing env var is a clear 4xx at
    # checkout rather than an import crash on boot.
    return {
        "monthly": settings.DODO_PRODUCT_MONTHLY,
        "yearly": settings.DODO_PRODUCT_YEARLY,
    }


def is_valid_interval(interval: str) -> bool:
    return interval in INTERVALS


def get_plan(interval: str, region: str) -> Optional[Dict[str, Any]]:
    """Price for one interval in one country.

    `region` accepts a country code ("JP"), or "INTL"/anything unknown, which falls
    through to the base USD price — the same fallback Dodo applies for a country
    with no localized rule, so display matches the charge.
    """
    if not is_valid_interval(interval):
        return None

    price = by_country().get((region or "").upper(), BASE)
    amount = price.monthly if interval == "monthly" else price.yearly

    return {
        "currency": price.currency,
        "amount": float(amount),
        "display": format_amount(amount, price.currency),
        "period": _PERIOD[interval],
        "country": price.country,
    }


def get_product_id(interval: str) -> Optional[str]:
    """Region-independent — Dodo localises the price on its side."""
    return _product_map().get(interval) or None


def yearly_saving_percent(region: str) -> int:
    """How much paying yearly saves versus twelve monthly charges."""
    monthly = get_plan("monthly", region)
    yearly = get_plan("yearly", region)
    if not monthly or not yearly or monthly["amount"] <= 0:
        return 0
    twelve = monthly["amount"] * 12
    return max(0, round((1 - yearly["amount"] / twelve) * 100))


def describe(interval: str, region: str) -> Dict[str, Any]:
    """Plan detail for API responses."""
    plan = get_plan(interval, region) or {}
    return {
        "plan": interval,
        "region": region,
        "currency": plan.get("currency"),
        "amount": plan.get("amount"),
        "display": plan.get("display"),
        "period": plan.get("period"),
        "trial_period_days": TRIAL_DAYS,
        "unlimited": True,
    }


def missing_products() -> List[str]:
    """Which intervals have no Dodo product configured. For startup diagnostics."""
    return [interval for interval, pid in _product_map().items() if not pid]
