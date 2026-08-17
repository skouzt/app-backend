"""Push the price catalogue to Dodo's localized-price rules.

  Dry run (default, read-only — shows the diff and writes nothing):
      python scripts/sync_dodo_prices.py
      python scripts/sync_dodo_prices.py --env live

  Apply:
      python scripts/sync_dodo_prices.py --apply
      python scripts/sync_dodo_prices.py --env live --apply

Idempotent: it lists what Dodo already has, then creates what is missing and
patches what differs. Re-running after a partial failure is safe, and a country
removed from the catalogue is reported but never auto-deleted — silently dropping
a price would move those customers to the base USD rate without anyone noticing.

Talks to the REST API directly rather than through the SDK: the pinned
dodopayments==1.87.1 predates the localized-prices resource, and upgrading a
payment SDK is not something to do as a side effect of a pricing change.

Environments are entirely separate objects in Dodo. Syncing test does nothing
for live; both need running, with their own product ids.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.billing.catalog import CATALOG, HOLD, minor_units  # noqa: E402

load_dotenv()

BASE_URL = {
    "test": "https://test.dodopayments.com",
    "live": "https://live.dodopayments.com",
}

INTERVAL_ENV = {
    "monthly": "DODO_PRODUCT_MONTHLY",
    "yearly": "DODO_PRODUCT_YEARLY",
}


def product_id_for(interval: str, env: str) -> Tuple[str | None, str]:
    """Test and live are separate products with separate ids, so the plain
    DODO_PRODUCT_* vars cannot address both. Prefer an env-suffixed var when it
    exists; fall back to the unsuffixed one the app already uses."""
    base = INTERVAL_ENV[interval]
    suffixed = f"{base}_{env.upper()}"
    if os.getenv(suffixed):
        return os.getenv(suffixed), suffixed
    return os.getenv(base), base


class Dodo:
    def __init__(self, api_key: str, env: str) -> None:
        self._c = httpx.Client(
            base_url=BASE_URL[env],
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

    def _raise(self, r: httpx.Response) -> None:
        if r.is_error:
            raise RuntimeError(f"{r.status_code} {r.request.method} {r.request.url} — {r.text}")

    def get_product(self, pid: str) -> dict:
        r = self._c.get(f"/products/{pid}")
        self._raise(r)
        return r.json()

    def set_pricing_mode(self, pid: str, mode: str) -> None:
        r = self._c.patch(f"/products/{pid}", json={"pricing_mode": mode})
        self._raise(r)

    def list_prices(self, pid: str) -> Dict[str, Tuple[str, str, int]]:
        """country_code -> (rule_id, currency, amount)"""
        r = self._c.get(f"/products/{pid}/localized-prices")
        self._raise(r)
        body = r.json()
        items = body.get("items", body) if isinstance(body, dict) else body
        out = {}
        for it in items:
            cc = it.get("country_code")
            if cc:
                out[cc] = (it["id"], it["currency"], int(it["amount"]))
        return out

    def create_price(self, pid: str, country: str, currency: str, amount: int) -> None:
        r = self._c.post(
            f"/products/{pid}/localized-prices",
            json={"country_code": country, "currency": currency, "amount": amount},
        )
        self._raise(r)

    def update_price(self, pid: str, rule_id: str, amount: int) -> None:
        r = self._c.patch(f"/products/{pid}/localized-prices/{rule_id}", json={"amount": amount})
        self._raise(r)


def sync_interval(dodo: Dodo, interval: str, product_id: str, apply: bool) -> int:
    print(f"\n{'=' * 70}\n{interval.upper()}  product={product_id}\n{'=' * 70}")

    product = dodo.get_product(product_id)
    mode = product.get("pricing_mode")
    base = product.get("price", {})
    print(f"base price: {base.get('price')} {base.get('currency')}   pricing_mode: {mode}")

    if mode != "by_country":
        print(f"  {'SET ' if apply else 'WOULD SET '}pricing_mode -> by_country")
        if apply:
            dodo.set_pricing_mode(product_id, "by_country")

    existing = dodo.list_prices(product_id)
    print(f"existing country rules: {len(existing)}\n")

    create, update, ok, held = [], [], 0, []

    for p in CATALOG:
        if p.country in HOLD:
            held.append(p.country)
            continue
        amount = minor_units(p.monthly if interval == "monthly" else p.yearly, p.currency)
        cur = existing.get(p.country)
        if cur is None:
            create.append((p.country, p.currency, amount))
        elif cur[2] != amount or cur[1] != p.currency:
            update.append((p.country, cur[0], cur[1], cur[2], p.currency, amount))
        else:
            ok += 1

    stale = sorted(set(existing) - {p.country for p in CATALOG})

    for cc, currency, amount in create:
        print(f"  {'CREATE' if apply else 'create'}  {cc}  {currency} {amount}")
    for cc, _rid, old_cur, old_amt, new_cur, new_amt in update:
        print(f"  {'UPDATE' if apply else 'update'}  {cc}  "
              f"{old_cur} {old_amt} -> {new_cur} {new_amt}")
    for cc in stale:
        print(f"  ORPHAN  {cc}  in Dodo but not in the catalogue — remove by hand if intended")
    for cc in held:
        print(f"  SKIP    {cc}  on hold: {HOLD[cc].split('.')[0]}.")

    print(f"\n  unchanged {ok} | create {len(create)} | update {len(update)} "
          f"| orphan {len(stale)} | held {len(held)}")

    if not apply:
        return 0

    failures = 0
    for cc, currency, amount in create:
        try:
            dodo.create_price(product_id, cc, currency, amount)
        except Exception as e:
            print(f"  FAILED create {cc}: {e}")
            failures += 1
    for cc, rid, _oc, _oa, _nc, new_amt in update:
        try:
            dodo.update_price(product_id, rid, new_amt)
        except Exception as e:
            print(f"  FAILED update {cc}: {e}")
            failures += 1

    print(f"  applied: {len(create) + len(update) - failures} ok, {failures} failed")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=("test", "live"), default="test")
    ap.add_argument("--apply", action="store_true", help="actually write (default is a dry run)")
    ap.add_argument("--interval", choices=("monthly", "yearly"), help="limit to one product")
    args = ap.parse_args()

    api_key = os.getenv("DODO_PAYMENTS_API_KEY")
    if not api_key:
        print("DODO_PAYMENTS_API_KEY not set")
        return 2

    if args.apply and args.env == "live":
        print("About to modify LIVE pricing. This changes what real customers are charged.")
        if input('Type "apply live" to continue: ').strip() != "apply live":
            print("aborted")
            return 1

    intervals = [args.interval] if args.interval else ["monthly", "yearly"]
    dodo = Dodo(api_key, args.env)

    print(f"env={args.env}  mode={'APPLY' if args.apply else 'DRY RUN (no writes)'}")

    failures = 0
    for interval in intervals:
        pid, var = product_id_for(interval, args.env)
        if not pid:
            print(f"\n{var} not set — skipping {interval}")
            failures += 1
            continue
        print(f"\n{interval}: product id from {var}")
        failures += sync_interval(dodo, interval, pid, args.apply)

    if not args.apply:
        print("\nDry run only. Re-run with --apply to write.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
