"""Global price catalogue — the single source of truth for what Lily costs.

Every price lives here exactly once. Three things derive from this table — the Dodo
localized-price rules (scripts/sync_dodo_prices.py), what the API quotes
(core/billing/plans.py), and the app's bundled fallback (scripts/gen_pricing_ts.py)
— so a price change is one edit rather than three that can drift apart. (Drift is
not hypothetical here: India once showed a PPP-discounted USD price because the Dodo
product and the app disagreed about what ₹149 meant.)

AMOUNTS ARE WRITTEN THE WAY A HUMAN READS THEM — "7.99", "1000", "149". The
conversion to Dodo's integer minor units happens in `minor_units()`, which is the
only place that knows how many decimal places a currency has. Do not pre-multiply
by 100 here; that is the bug this module exists to prevent.

Countries absent from this table fall through to the product's base USD price.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, Iterator, NamedTuple, Tuple

# Currencies with no minor unit at all: ¥1,000 is 1000, not 100000. Getting this
# wrong overcharges by 100x, and it is silent — the API accepts the integer either
# way. This is the single most dangerous line in the file.
ZERO_DECIMAL = {
    "BIF", "CLP", "DJF", "GNF", "ISK", "JPY", "KMF", "KRW", "MGA",
    "PYG", "RWF", "UGX", "VND", "VUV", "XAF", "XOF", "XPF",
}

# Gulf currencies subdivide into 1000, not 100. KWD 2.49 is 2490 minor units.
THREE_DECIMAL = {"BHD", "IQD", "JOD", "KWD", "LYD", "OMR", "TND"}

# Dodo rejects these for localized pricing with UNSUPPORTED_CURRENCY, even though
# they appear in the SDK's Currency enum — the enum is the superset the platform
# knows about, not the subset localized pricing accepts. Determined empirically
# against the API; re-test before assuming any of them became available.
LOCALIZED_UNSUPPORTED = {"ARS", "COP", "GHS", "ISK", "KES", "NAD", "PKR", "UGX"}


def decimals_for(currency: str) -> int:
    if currency in ZERO_DECIMAL:
        return 0
    if currency in THREE_DECIMAL:
        return 3
    return 2


def minor_units(amount: str, currency: str) -> int:
    """'7.99' USD -> 799.  '1000' JPY -> 1000.  '2.49' KWD -> 2490."""
    value = Decimal(amount)
    scaled = value * (10 ** decimals_for(currency))
    if scaled != scaled.to_integral_value():
        raise ValueError(
            f"{amount} {currency} has more precision than the currency allows "
            f"({decimals_for(currency)} dp)"
        )
    return int(scaled)


class Price(NamedTuple):
    country: str      # ISO-3166 alpha-2
    currency: str     # ISO-4217
    monthly: str      # human-readable, e.g. "7.99"
    yearly: str


# Countries with no rule of their own are charged the product's base price.
BASE = Price("ZZ", "USD", "7.99", "79.99")

# Rendered server-side because the client cannot be trusted to have full ICU:
# Hermes builds without it do not raise, they quietly render "INR 999.00" instead
# of "₹999". Formatting here means the price string is identical everywhere.
SYMBOLS: Dict[str, str] = {
    "USD": "$", "EUR": "€", "GBP": "£", "INR": "₹", "JPY": "¥", "CNY": "¥",
    "KRW": "₩", "VND": "₫", "PHP": "₱", "THB": "฿", "TRY": "₺", "NGN": "₦",
    "ILS": "₪", "KZT": "₸", "GEL": "₾", "AMD": "֏", "PYG": "₲", "CRC": "₡",
    "GHS": "GH₵", "LKR": "LKR ", "PKR": "Rs ", "NPR": "NPR ", "BDT": "৳",
    "MVR": "MVR ", "AUD": "A$", "NZD": "NZ$", "CAD": "C$", "SGD": "S$",
    "HKD": "HK$", "TWD": "NT$", "BRL": "R$", "MXN": "MX$", "ZAR": "R",
    "CHF": "CHF ", "SEK": "", "NOK": "", "DKK": "", "PLN": "", "CZK": "",
    "HUF": "", "RON": "", "MYR": "RM", "IDR": "Rp", "AED": "AED ",
    "SAR": "SAR ", "QAR": "QAR ", "JOD": "JOD ", "KWD": "KWD ", "OMR": "OMR ",
    "BHD": "BHD ", "IQD": "IQD ", "BSD": "B$", "BZD": "BZ$", "DOP": "RD$",
    "GTQ": "Q", "HNL": "L", "PEN": "S/", "UYU": "UYU ", "BOB": "Bs ",
    "CLP": "CLP ", "COP": "COP ", "ARS": "ARS ", "KES": "KSh ", "TZS": "TZS ",
    "UGX": "UGX ", "ZMW": "ZMW ", "BWP": "BWP ", "NAD": "N$", "MUR": "MUR ",
    "SCR": "SCR ", "ETB": "ETB ", "EGP": "EGP ", "MAD": "MAD ", "BND": "B$",
    "FJD": "FJ$", "PGK": "PGK ", "WST": "WST ", "TOP": "TOP ", "SBD": "SBD ",
    "VUV": "VUV ", "RSD": "RSD ", "MKD": "MKD ", "BAM": "KM ", "BBD": "Bds$",
}

# Currencies conventionally written with the unit after the number.
SUFFIX = {"SEK": " kr", "NOK": " kr", "DKK": " kr", "PLN": " zł",
          "CZK": " Kč", "HUF": " Ft", "RON": " lei"}


def format_amount(amount: str, currency: str) -> str:
    """'1000' JPY -> '¥1,000'.  '7.99' USD -> '$7.99'.  '79' SEK -> '79 kr'."""
    # Format exactly the precision the catalogue was written with: "2.49" stays
    # "2.49" rather than being padded to KWD's three places, and "149" stays "149"
    # rather than gaining ".00". The written form is already the intended display.
    text = f"{Decimal(amount):,}"
    if currency in SUFFIX:
        return f"{text}{SUFFIX[currency]}"
    return f"{SYMBOLS.get(currency, currency + ' ')}{text}"


# Countries the sync script refuses to push until a human confirms them. Being in
# the catalogue records the intended price; being here says "not verified yet".
HOLD: Dict[str, str] = {
    "IQ": (
        "IQD is officially 3-decimal, but the fils is long obsolete and most "
        "processors treat the dinar as 0-decimal. We would send 7,500,000 minor "
        "units for IQD 7,500 — if Dodo expects 7,500, that is a 1000x overcharge. "
        "Confirm with Dodo support which they expect before enabling Iraq."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# The catalogue. Ordered by region for review; order is not significant.
# ─────────────────────────────────────────────────────────────────────────────

CATALOG: Tuple[Price, ...] = (
    # ── North America ────────────────────────────────────────────────────────
    Price("US", "USD", "7.99", "79.99"),
    Price("CA", "CAD", "10.99", "109.99"),
    Price("MX", "MXN", "99", "999"),
    Price("BS", "BSD", "7.99", "79.99"),
    # Barbados deliberately omitted: BBD is pegged 2:1 to USD, so a Bds$ price set
    # to the USD number charges half. Falls through to the base USD price until
    # someone decides what Barbados should actually cost.
    Price("BM", "USD", "7.99", "79.99"),
    Price("BZ", "BZD", "15", "149"),
    Price("CR", "CRC", "3990", "39900"),
    Price("DO", "DOP", "399", "3999"),
    Price("GT", "GTQ", "59", "599"),
    Price("HN", "HNL", "199", "1999"),

    # ── Europe ───────────────────────────────────────────────────────────────
    Price("GB", "GBP", "6.99", "69.99"),
    Price("DE", "EUR", "7.99", "79.99"),
    Price("FR", "EUR", "7.99", "79.99"),
    Price("IT", "EUR", "7.99", "79.99"),
    Price("ES", "EUR", "7.99", "79.99"),
    Price("NL", "EUR", "7.99", "79.99"),
    Price("BE", "EUR", "7.99", "79.99"),
    Price("AT", "EUR", "7.99", "79.99"),
    Price("IE", "EUR", "7.99", "79.99"),
    Price("FI", "EUR", "7.99", "79.99"),
    Price("PT", "EUR", "6.99", "69.99"),
    Price("GR", "EUR", "6.99", "69.99"),
    Price("CY", "EUR", "7.99", "79.99"),
    Price("LU", "EUR", "7.99", "79.99"),
    Price("MT", "EUR", "7.99", "79.99"),
    Price("EE", "EUR", "6.99", "69.99"),
    Price("LV", "EUR", "6.99", "69.99"),
    Price("LT", "EUR", "6.99", "69.99"),
    Price("SK", "EUR", "6.99", "69.99"),
    Price("SI", "EUR", "6.99", "69.99"),
    Price("HR", "EUR", "6.99", "69.99"),
    Price("CH", "CHF", "7.99", "79.99"),
    Price("SE", "SEK", "79", "799"),
    Price("NO", "NOK", "79", "799"),
    Price("DK", "DKK", "59", "599"),
    # Iceland: Dodo does not support ISK (it is absent from their 145-currency
    # list), so the krona price cannot be created. EUR is the closest correct
    # option — Icelandic cards handle it routinely.
    Price("IS", "EUR", "7.99", "79.99"),
    Price("PL", "PLN", "29.99", "299"),
    Price("CZ", "CZK", "179", "1790"),
    Price("HU", "HUF", "2490", "24900"),
    Price("RO", "RON", "34.99", "349.99"),
    Price("BG", "EUR", "5.99", "59.99"),
    Price("RS", "RSD", "699", "6990"),
    Price("MK", "MKD", "399", "3990"),
    Price("BA", "BAM", "7.99", "79.99"),

    # ── Asia / Pacific ───────────────────────────────────────────────────────
    Price("JP", "JPY", "1000", "9900"),
    Price("KR", "KRW", "9900", "99000"),
    Price("SG", "SGD", "9.99", "99.99"),
    Price("HK", "HKD", "59", "599"),
    Price("TW", "TWD", "249", "2490"),
    Price("AU", "AUD", "11.99", "119.99"),
    Price("NZ", "NZD", "12.99", "129.99"),
    Price("MY", "MYR", "19.90", "199"),
    Price("TH", "THB", "199", "1990"),
    Price("ID", "IDR", "49000", "490000"),
    Price("PH", "PHP", "249", "2490"),
    Price("VN", "VND", "79000", "790000"),
    Price("CN", "CNY", "49", "499"),
    Price("TR", "TRY", "149", "1490"),
    Price("KZ", "KZT", "3990", "39900"),
    Price("GE", "GEL", "19.90", "199"),
    Price("AM", "AMD", "2990", "29900"),

    # ── South Asia ───────────────────────────────────────────────────────────
    Price("IN", "INR", "149", "999"),
    Price("BD", "BDT", "199", "1999"),
    Price("PK", "USD", "1.99", "19.99"),    # was PKR 499/4999 (~$1.79) — PKR unsupported
    Price("NP", "NPR", "399", "3999"),
    Price("LK", "LKR", "999", "9999"),
    Price("MV", "MVR", "99", "999"),

    # ── Middle East ──────────────────────────────────────────────────────────
    Price("AE", "AED", "29", "299"),
    Price("SA", "SAR", "29", "299"),
    Price("QA", "QAR", "29", "299"),
    Price("IL", "ILS", "29", "299"),
    Price("JO", "JOD", "5.49", "54.90"),
    Price("IQ", "IQD", "7500", "75000"),
    Price("KW", "KWD", "2.49", "24.90"),
    Price("OM", "OMR", "2.49", "24.90"),
    Price("BH", "BHD", "2.99", "29.90"),
    Price("LB", "USD", "3.99", "39.99"),

    # ── Latin America ────────────────────────────────────────────────────────
    Price("BR", "BRL", "29.90", "299"),
    Price("AR", "USD", "4.99", "49.99"),    # was ARS 6999/69990 (~$4.80) — ARS unsupported
    Price("CL", "CLP", "5990", "59900"),
    Price("CO", "USD", "4.99", "49.99"),    # was COP 19900/199000 (~$5.00) — COP unsupported
    Price("PE", "PEN", "19.90", "199"),
    Price("UY", "UYU", "299", "2990"),
    Price("BO", "BOB", "39", "399"),
    Price("PY", "PYG", "29900", "299000"),
    Price("EC", "USD", "4.99", "49.99"),
    Price("VE", "USD", "3.99", "39.99"),

    # ── Africa ───────────────────────────────────────────────────────────────
    Price("ZA", "ZAR", "99", "999"),
    Price("NG", "NGN", "4999", "49990"),
    Price("KE", "USD", "3.99", "39.99"),    # was KES 499/4990 (~$3.86) — KES unsupported
    Price("GH", "USD", "4.99", "49.99"),    # was GHS 59/599 (~$4.60) — GHS unsupported
    Price("EG", "EGP", "249", "2490"),
    Price("MA", "MAD", "49", "490"),
    Price("TZ", "TZS", "9900", "99000"),
    Price("UG", "USD", "3.99", "39.99"),    # was UGX 14900/149000 (~$3.95) — UGX unsupported
    Price("ZM", "ZMW", "49", "490"),
    Price("BW", "BWP", "69", "690"),
    Price("NA", "USD", "4.99", "49.99"),    # was NAD 79/790 (~$4.35) — NAD unsupported
    Price("MU", "MUR", "199", "1990"),
    Price("SC", "SCR", "79", "790"),
    Price("ET", "ETB", "499", "4990"),

    # ── Pacific / other ──────────────────────────────────────────────────────
    Price("BN", "BND", "9.99", "99"),
    Price("FJ", "FJD", "12.99", "129"),
    Price("PG", "PGK", "39", "390"),
    Price("WS", "WST", "19", "190"),
    Price("TO", "TOP", "19", "190"),
    Price("SB", "SBD", "39", "390"),
    Price("VU", "VUV", "799", "7990"),
    Price("MG", "USD", "2.99", "29.99"),
)


def by_country() -> Dict[str, Price]:
    return {p.country: p for p in CATALOG}


def for_interval(interval: str) -> Iterator[Tuple[str, str, int]]:
    """Yield (country_code, currency, amount_in_minor_units) for one interval."""
    if interval not in ("monthly", "yearly"):
        raise ValueError(f"Unknown interval: {interval}")
    for p in CATALOG:
        amount = p.monthly if interval == "monthly" else p.yearly
        yield p.country, p.currency, minor_units(amount, p.currency)
