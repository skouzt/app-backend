"""
core/billing/dodo_client.py
Drop-in replacement for GumroadClient.
Uses the official Dodo Payments Python SDK.
"""

from __future__ import annotations

import hmac
import hashlib
import base64
import os
import time
from typing import Any, Optional

from dodopayments import DodoPayments

from core.config import settings
import structlog

logger = structlog.get_logger(__name__)

from core.billing.plans import get_product_id, is_valid_interval
from core.billing.region import REGION_INTL


class DodoClient:
    """Thin wrapper around the Dodo Payments SDK."""

    def __init__(self) -> None:
        self._client = DodoPayments(
            bearer_token=settings.DODO_PAYMENTS_API_KEY,
            environment=settings.DODO_ENVIRONMENT, 
        )

    # ── Checkout ───────────────────────────────────────────────────────────────

    def get_checkout_url(
        self,
        plan_key: str,
        email: str,
        customer_name: str,
        return_url: str,
        user_id: str,
        region: str = REGION_INTL,
        region_trusted: bool = False,
        trial_period_days: int = 0,
    ) -> tuple[str, str]:

        if not is_valid_interval(plan_key):
            raise ValueError(f"Unknown plan: {plan_key}")

        product_id = get_product_id(plan_key)
        if not product_id:
            raise ValueError(f"No Dodo product configured for '{plan_key}'")

        # Dodo's Localized Pricing keys off the billing country, so this seed decides
        # which currency the customer is first shown.
        #
        # This deliberately trusts the resolved region even when it came from the
        # client hint. Requiring a trusted edge header meant every user behind a proxy
        # that injects no geo header (ngrok, plain uvicorn) was shown ₹149 in the app
        # and then charged $7.99 at checkout — telling customers one price and taking
        # another is a worse failure than the spoofing it prevented.
        #
        # Spoofing stays contained because the seed is only a prefill: Dodo re-prices
        # when the customer sets their real billing country, a mismatched card tends to
        # fail address verification, and the webhook logs billing_region_mismatch
        # against the country the payment actually came from.
        # Seed with the resolved country itself. This was `"IN" if region == "IN"
        # else "US"`, written when pricing was a two-tier IN/INTL split. Once the
        # catalogue grew to 109 countries that line silently broke every one of
        # them: a GB customer was quoted £69.99 in the app, seeded as US, and shown
        # $79.99 by Dodo — the display/charge mismatch this system exists to
        # prevent, for everyone except India.
        #
        # "ZZ" is the catalogue's unknown-country marker and "INTL" the legacy tier
        # name; both mean "no localized price", which is the base USD product, so
        # they seed as US.
        seed_country = (
            region.upper()
            if region and len(region) == 2 and region.isalpha() and region.upper() != "ZZ"
            else "US"
        )

        create_kwargs = {
            "billing": {
                "city": "",
                "country": seed_country,
                "state": "",
                "street": "",
                "zipcode": "0",
            },
            "customer": {
                "email": email,
                "name": customer_name,
            },
            "product_id": product_id,
            "quantity": 1,
            "return_url": return_url or settings.DODO_DEFAULT_RETURN_URL,
            "payment_link": True,
            "metadata": {
                "plan_key": plan_key,
                "region": region,
                "user_id": user_id,
                "app": "lily",
            },
        }

        if trial_period_days > 0:
            create_kwargs["trial_period_days"] = trial_period_days

        session = self._client.subscriptions.create(**create_kwargs)

        return str(session.payment_link), str(session.subscription_id)

    # ── Subscription retrieval ─────────────────────────────────────────────────

    def get_subscription(self, subscription_id: str) -> Any:
        """Fetch a subscription object from Dodo."""
        return self._client.subscriptions.retrieve(subscription_id)

    def is_subscription_active(self, subscription: Any) -> bool:
        """Check if a Dodo subscription object represents an active subscription."""
        return str(getattr(subscription, "status", "")).lower() in {"active", "trialing"}

    def get_plan_from_subscription(self, subscription: Any) -> Optional[str]:
        """Derive our internal interval from the subscription's metadata."""
        meta = getattr(subscription, "metadata", {}) or {}
        if not isinstance(meta, dict):
            meta = dict(meta)
        plan_key = meta.get("plan_key")
        return plan_key if is_valid_interval(str(plan_key)) else None

    @staticmethod
    def billing_country(data: dict) -> Optional[str]:
        """The country the payment actually came from — the authoritative signal.

        Only available after checkout, so it can't choose the product; it exists to
        catch a client that lied about its region to reach the cheaper tier.
        """
        for path in (("billing", "country"), ("customer", "country"), ("country",)):
            node: Any = data
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
                if node is None:
                    break
            if isinstance(node, str) and node.strip():
                return node.strip().upper()
        return None

    def get_next_billing_date(self, subscription: Any) -> Optional[str]:
        """Return ISO-format next billing date or None."""
        nbd = getattr(subscription, "next_billing_date", None)
        if nbd:
            return str(nbd)
        return None

    # ── Webhook signature verification ────────────────────────────────────────

    @staticmethod
    def verify_webhook_signature(
        raw_body: bytes,
        webhook_id: str,
        webhook_timestamp: str,
        webhook_signature: str,
        secret: str,
    ) -> bool:
       
        if not all([webhook_id, webhook_timestamp, webhook_signature]):
            return False

        # Replay attack guard – reject events older than 5 minutes
        try:
            ts = int(webhook_timestamp)
            age = abs(time.time() - ts)
            max_age = int(os.getenv("DODO_WEBHOOK_MAX_AGE", "300"))
            if age > max_age:
                logger.warning("dodo_webhook_replay_attack_detected")
                return False
        except ValueError:
            return False

        signed_content = f"{webhook_id}.{webhook_timestamp}.{raw_body.decode()}"

        try:
            # Strip the "whsec_" prefix before base64 decoding
            raw_secret = secret.removeprefix("whsec_")
            secret_bytes = base64.b64decode(raw_secret)
        except Exception:
            return False

        expected = base64.b64encode(
            hmac.new(secret_bytes, signed_content.encode(), hashlib.sha256).digest()
        ).decode()

        # Dodo may send multiple space-separated sigs
        return any(
            hmac.compare_digest(f"v1,{expected}", sig)
            for sig in webhook_signature.split(" ")
        )