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
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import dodopayments
from dodopayments import DodoPayments

from core.config import settings
import structlog

logger = structlog.get_logger(__name__)

# ── Plan → Dodo Product ID mapping ────────────────────────────────────────────
# Create these products in the Dodo dashboard (Products → New, billing=monthly)
# then paste the pdt_xxx IDs into your .env
PLAN_PRODUCT_MAP: Dict[str, str] = {
    "clarity": settings.DODO_PLAN_CLARITY_ID,   
    "insight": settings.DODO_PLAN_INSIGHT_ID,   
}


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
        trial_period_days: int = 0,
    ) -> tuple[str, str]:

        product_id = PLAN_PRODUCT_MAP.get(plan_key)
        if not product_id:
            raise ValueError(f"Unknown plan: {plan_key}")

        create_kwargs = {
            "billing": {
                "city": "Test",
                "country": "US",
                "state": "CA",
                "street": "Test Street",
                "zipcode": "94103",
            },
            "customer": {
                "email": email,
                "name": customer_name,
            },
            "product_id": product_id,
            "quantity": 1,
            "return_url": return_url,
            "payment_link": True,
            "metadata": {
                "plan_key": plan_key,
                "user_id": user_id,
                "app": "aletheia",
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
        """
        Derive our internal plan_key from the Dodo subscription's metadata
        or product_id.
        """
        meta = getattr(subscription, "metadata", {}) or {}
        if not isinstance(meta, dict):
            meta = dict(meta)
        plan_key = meta.get("plan_key")
        if plan_key in PLAN_PRODUCT_MAP:
            return plan_key

        # Fallback: match product_id
        product_id = str(getattr(subscription, "product_id", ""))
        for key, pid in PLAN_PRODUCT_MAP.items():
            if pid == product_id:
                return key

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