"""Generate the app's offline price table from the catalogue.

    python scripts/gen_pricing_ts.py

The app needs a price on first paint, before any network call returns, and it needs
one at all if the request fails. That fallback used to be hand-maintained, which is
exactly how the app came to show USD to Indian users while Dodo charged INR. Now it
is generated, so it cannot disagree with what we push to Dodo.

The generated file is still only a fallback. `/api/v1/billing/pricing` is
authoritative because the server sees the edge geo header and the device only sees
its own locale setting.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.billing.catalog import BASE, CATALOG, format_amount  # noqa: E402

OUT = Path("/home/skouzt/app/constants/pricing.generated.ts")


def rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1], text=True,
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    rows = []
    for p in sorted(CATALOG, key=lambda x: x.country):
        rows.append(
            f"  {p.country}: {{ currency: '{p.currency}', "
            f"monthly: {p.monthly}, yearly: {p.yearly}, "
            f"monthlyLabel: '{format_amount(p.monthly, p.currency)}', "
            f"yearlyLabel: '{format_amount(p.yearly, p.currency)}' }},"
        )

    body = f"""/**
 * GENERATED FILE — DO NOT EDIT.
 *
 * Produced by backend/server/scripts/gen_pricing_ts.py from core/billing/catalog.py,
 * the same table that scripts/sync_dodo_prices.py pushes to Dodo. Editing this file
 * by hand reintroduces the drift it exists to prevent.
 *
 * Regenerate:  python scripts/gen_pricing_ts.py
 * Catalogue rev: {rev()}   Countries: {len(CATALOG)}
 *
 * This is the OFFLINE FALLBACK only. The server decides the real price — it can see
 * the edge geo header, whereas the device only knows its own locale setting, which
 * a user can change in Settings.
 */

export interface CountryPricing {{
  currency: string;
  monthly: number;
  yearly: number;
  monthlyLabel: string;
  yearlyLabel: string;
}}

/** Charged to any country without a rule of its own — matches Dodo's base price. */
export const BASE_PRICING: CountryPricing = {{
  currency: '{BASE.currency}',
  monthly: {BASE.monthly},
  yearly: {BASE.yearly},
  monthlyLabel: '{format_amount(BASE.monthly, BASE.currency)}',
  yearlyLabel: '{format_amount(BASE.yearly, BASE.currency)}',
}};

export const PRICING_BY_COUNTRY: Record<string, CountryPricing> = {{
{chr(10).join(rows)}
}};

export function pricingForCountry(code?: string | null): CountryPricing {{
  if (!code) return BASE_PRICING;
  return PRICING_BY_COUNTRY[code.toUpperCase()] ?? BASE_PRICING;
}}
"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body)
    print(f"wrote {OUT} — {len(CATALOG)} countries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
