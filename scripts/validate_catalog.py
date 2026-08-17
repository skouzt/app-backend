"""Check the price catalogue before anything is pushed to Dodo.

Run this after every catalogue edit. It is cheap, offline, and it catches the
class of mistake that is invisible in the dashboard afterwards — a 100x overcharge
in a zero-decimal currency looks like a perfectly ordinary integer.
"""

from __future__ import annotations

import re
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.billing.catalog import (  # noqa: E402
    CATALOG, HOLD, LOCALIZED_UNSUPPORTED, decimals_for, minor_units,
)

SDK_CURRENCY = Path(
    "/tmp/claude-1000/-home-skouzt-project-backend-server/"
    "0e5c3db3-8058-401d-9f27-68458977afef/scratchpad/sdk/x/dodopayments/types/currency.py"
)

# Hard pegs, so the USD value is exact rather than a guess at today's FX. These
# exist to catch prices that look right in local currency but are wildly off in
# dollars — a 2:1 pegged currency priced at the USD number is half price.
PEGS_USD = {
    "BSD": Decimal("1"), "BBD": Decimal("2"), "BZD": Decimal("2"),
    "BMD": Decimal("1"), "AED": Decimal("3.6725"), "SAR": Decimal("3.75"),
    "QAR": Decimal("3.64"), "OMR": Decimal("0.3845"), "BHD": Decimal("0.376"),
    "JOD": Decimal("0.709"), "HKD": Decimal("7.8"), "XCD": Decimal("2.7"),
}

USD_ANCHOR = Decimal("7.99")

errors: list[str] = []
warnings: list[str] = []


def supported_currencies() -> set[str]:
    if not SDK_CURRENCY.exists():
        warnings.append("SDK currency enum not found — skipped currency support check")
        return set()
    return set(re.findall(r'"([A-Z]{3})"', SDK_CURRENCY.read_text()))


def main() -> int:
    supported = supported_currencies()
    seen: set[str] = set()

    print(f"catalogue: {len(CATALOG)} countries\n")

    for p in CATALOG:
        tag = f"{p.country}/{p.currency}"

        if p.country in seen:
            errors.append(f"{tag}: duplicate country row")
        seen.add(p.country)

        if not re.fullmatch(r"[A-Z]{2}", p.country):
            errors.append(f"{tag}: not an ISO-3166 alpha-2 code")
        if p.currency in LOCALIZED_UNSUPPORTED:
            errors.append(
                f"{tag}: Dodo rejects {p.currency} for localized prices "
                f"(UNSUPPORTED_CURRENCY) — price this market in USD instead"
            )
        elif supported and p.currency not in supported:
            errors.append(f"{tag}: currency absent from Dodo's currency list")

        # Decimal-place correctness — the 100x bug.
        for interval, amount in (("monthly", p.monthly), ("yearly", p.yearly)):
            try:
                minor_units(amount, p.currency)
            except ValueError as e:
                errors.append(f"{tag} {interval}: {e}")

        # A yearly price above 12x monthly means the "discount" is a penalty.
        m, y = Decimal(p.monthly), Decimal(p.yearly)
        if y > m * 12:
            errors.append(f"{tag}: yearly {y} exceeds 12x monthly ({m * 12})")
        months = (y / m).quantize(Decimal("0.1"))
        if months > Decimal("11"):
            warnings.append(f"{tag}: yearly is {months} months of monthly — thin discount")

        # Pegged currencies convert exactly, so an outlier here is a real error.
        if p.currency in PEGS_USD:
            usd = (m / PEGS_USD[p.currency]).quantize(Decimal("0.01"))
            ratio = usd / USD_ANCHOR
            if ratio < Decimal("0.55"):
                errors.append(
                    f"{tag}: monthly {p.monthly} {p.currency} = ${usd} "
                    f"— {int((1 - ratio) * 100)}% below the ${USD_ANCHOR} anchor"
                )
            elif ratio > Decimal("1.35"):
                warnings.append(
                    f"{tag}: monthly {p.monthly} {p.currency} = ${usd} "
                    f"— {int((ratio - 1) * 100)}% above the ${USD_ANCHOR} anchor"
                )

    # Show how the trickiest currencies actually serialise.
    print("minor-unit conversion for non-2-decimal currencies:")
    for p in CATALOG:
        d = decimals_for(p.currency)
        if d != 2:
            print(f"  {p.country} {p.currency:4} {d}dp  "
                  f"monthly {p.monthly:>8} -> {minor_units(p.monthly, p.currency):>9}  "
                  f"yearly {p.yearly:>8} -> {minor_units(p.yearly, p.currency):>9}")

    if HOLD:
        print("\non hold — excluded from sync until verified:")
        for cc, why in HOLD.items():
            print(f"  {cc}: {why}")

    print()
    for w in warnings:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  FAIL  {e}")

    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
