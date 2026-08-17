"""Which price tier a customer is actually billed at.

The client sends a `region` in the checkout body, but that is derived from the device
locale — one tap in phone settings and anyone can ask for the ₹ tier. So the client
value is treated as a *hint of last resort*.

Preference order:
  1. A country header injected by the edge/proxy (Cloudflare, Vercel, App Engine…).
     The user cannot forge these; the infrastructure sets them from the TCP peer.
  2. The client hint, recorded as untrusted so mismatches can be audited later.

The webhook cross-checks the country the payment actually came from, which is the
only fully authoritative signal — but it arrives after checkout, so it can't pick the
product.
"""

from __future__ import annotations

from typing import Optional, Tuple

from fastapi import Request

# ISO-3166 codes billed in the local tier.
LOCAL_TIER_COUNTRIES = {"IN"}

REGION_LOCAL = "IN"
REGION_INTL = "INTL"

# Headers set by common edge providers. Order matters — first match wins.
GEO_HEADERS = (
    "cf-ipcountry",            # Cloudflare
    "x-vercel-ip-country",     # Vercel
    "x-appengine-country",     # Google App Engine
    "x-country-code",          # assorted proxies
    "x-geo-country",
)

# Placeholders these providers use when they cannot resolve a country.
UNKNOWN_COUNTRY = {"XX", "T1", "ZZ", ""}


def region_for_country(country: Optional[str]) -> str:
    if not country:
        return REGION_INTL
    return REGION_LOCAL if country.strip().upper() in LOCAL_TIER_COUNTRIES else REGION_INTL


def resolve_billing_region(
    request: Request, client_hint: Optional[str] = None
) -> Tuple[str, str]:
    """Return (region, source). `source` records how much to trust the result."""
    for header in GEO_HEADERS:
        raw = request.headers.get(header)
        if raw and raw.strip().upper() not in UNKNOWN_COUNTRY:
            return region_for_country(raw), header

    if client_hint:
        hint = client_hint.strip().upper()
        if hint in (REGION_LOCAL, REGION_INTL):
            return hint, "client-hint"

    return REGION_INTL, "default"


def is_trusted(source: str) -> bool:
    """True when the region came from infrastructure rather than the client."""
    return source in GEO_HEADERS


def resolve_billing_country(
    request: Request, client_hint: Optional[str] = None
) -> Tuple[str, str]:
    """Return (ISO-3166 country, source) — the finer-grained twin of
    `resolve_billing_region`.

    The two-tier IN/INTL split predates per-country pricing; with a full catalogue
    the country itself is what selects a price, so this preserves it instead of
    collapsing 100+ countries into "INTL". Same trust order: an edge header the
    user cannot forge first, then the client's hint, then nothing.

    Returns "ZZ" when unknown, which `get_plan` resolves to the base USD price —
    matching what Dodo charges a country with no localized rule.
    """
    for header in GEO_HEADERS:
        raw = request.headers.get(header)
        if raw and raw.strip().upper() not in UNKNOWN_COUNTRY:
            return raw.strip().upper(), header

    if client_hint:
        hint = client_hint.strip().upper()
        # Legacy clients send the tier, not a country. Map it back so an old app
        # build keeps working against a new server.
        if hint == REGION_INTL:
            return "ZZ", "client-hint-legacy"
        if len(hint) == 2 and hint.isalpha():
            return hint, "client-hint"

    return "ZZ", "default"
