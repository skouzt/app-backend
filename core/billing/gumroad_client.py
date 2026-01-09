import httpx
from typing import Dict, Any, Optional
from core.config import settings
from loguru import logger

class GumroadClient:
    base_url = "https://api.gumroad.com/v2"

    def __init__(self):
        self.api_key = settings.GUMROAD_API_KEY
        self.product_id = settings.GUMROAD_PRODUCT_ID

    async def verify_license(self, license_key: str) -> Dict[str, Any]:
        """
        Verify a license key with Gumroad API.
        
        Returns the full response from Gumroad including purchase details.
        """
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    f"{self.base_url}/licenses/verify",
                    data={
                        "product_id": self.product_id,
                        "license_key": license_key,
                    },
                    timeout=10,
                )
                res.raise_for_status()
                response_data = res.json()
                
                # ✅ SAFE: Log action only, not response content
                logger.debug(f"Gumroad license verification attempted for {license_key[:8]}***")
                
                return response_data
                
        except httpx.HTTPStatusError as e:
            # ✅ SAFE: Log status code only, not full error
            logger.error(f"Gumroad API error: HTTP {e.response.status_code}")
            # Return error response instead of raising
            return {"success": False, "message": "API request failed"}
        except Exception as e:
            # ✅ SAFE: Log exception type only
            logger.error(f"License verification failed: {type(e).__name__}")
            return {"success": False, "message": "Internal error"}

    def is_subscription_active(self, data: Dict[str, Any]) -> bool:
        """
        Check if a subscription is currently active.
        
        Args:
            data: Response from verify_license()
        
        Returns:
            True if subscription is active and valid
        """
        if not data.get("success"):
            return False

        purchase = data.get("purchase", {})
        
        # Check for refunds or chargebacks
        if purchase.get("refunded") or purchase.get("chargebacked"):
            return False

        # Check if subscription is cancelled
        if purchase.get("cancelled"):
            return False

        # Check subscription-specific fields
        if purchase.get("subscription_cancelled_at"):
            return False

        if purchase.get("subscription_failed_at"):
            return False

        return True

    def get_plan_from_purchase(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Determine which plan (guided/extended) based on purchase price.
        
        Args:
            data: Response from verify_license()
        
        Returns:
            "guided" | "extended" | None
        """
        if not data.get("success"):
            return None
        
        purchase = data.get("purchase", {})
        price = purchase.get("price")
        
        # Match price to plan (prices in cents)
        if price == settings.GUIDED_PRICE:
            return "guided"
        elif price == settings.EXTENDED_PRICE:
            return "extended"
        
        # ✅ SAFE: Log only that an unknown price was encountered
        logger.warning("License verification: Unknown price encountered")
        return None

    def calculate_expiry(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Get subscription expiry date from purchase data.
        
        For active subscriptions, returns None (ongoing).
        For cancelled, returns the end date.
        """
        if not data.get("success"):
            return None
        
        purchase = data.get("purchase", {})
        
        # Check for explicit end date first
        return (
            purchase.get("subscription_ended_at")
            or purchase.get("subscription_cancelled_at")
            or purchase.get("subscription_failed_at")
        )

    def get_checkout_url(self, plan_key: str, email: str) -> str:
        """
        Generate Gumroad checkout URL for a specific plan.
        
        Args:
            plan_key: "guided" or "extended"
            email: Pre-fill user's email
        
        Returns:
            Full checkout URL
        """
        # Map plan to price
        price_map = {
            "guided": 15,   # $15
            "extended": 50  # $50
        }
        
        price = price_map.get(plan_key)
        if not price:
            raise ValueError(f"Invalid plan: {plan_key}")

        # Build checkout URL with wanted=true for subscription
        url = (
            f"https://skouzt.gumroad.com/l/{settings.GUMROAD_PRODUCT_PERMALINK}"
             f"?wanted=true&price={price}"
        )
        
        if email:
            url += f"&email={email}"
        
        # ✅ Note: URL contains email - ensure this method isn't called in contexts where URLs get logged
        return url

    def extract_subscription_id(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Extract Gumroad subscription ID from purchase data.
        """
        if not data.get("success"):
            return None
        
        purchase = data.get("purchase", {})
        return purchase.get("subscription_id")

    def extract_email(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Extract purchaser email from purchase data.
        """
        if not data.get("success"):
            return None
        
        purchase = data.get("purchase", {})
        return purchase.get("email")