# tasks/verify_gumroad_subscriptions.py
import asyncio
import hashlib  # ⭐ Added for safe user referencing
from datetime import datetime, timedelta
from typing import Dict, Any, List, cast, TypedDict
from loguru import logger
from db.supabase import supabase
from core.billing.gumroad_client import GumroadClient

gumroad_client = GumroadClient()

# -------------------------------
# Type Definitions
# -------------------------------

class SubscriptionRow(TypedDict, total=False):
    user_id: str
    gumroad_license_key: str  # ✅ Changed from gumroad_purchase_id
    plan_key: str

def safe_get_str(data: Any, key: str, default: str = "") -> str:
    """Safely get string value with type checking"""
    if not isinstance(data, dict):
        return default
    value = data.get(key)
    return str(value) if value is not None else default

def _safe_user_ref(user_id: str) -> str:
    """Create a safe, non-reversible reference for logging."""
    if not user_id or user_id == "anonymous":
        return "anonymous"
    # Create a deterministic but non-reversible hash
    return f"user_{hashlib.sha256(user_id.encode()).hexdigest()[:8]}"

# -------------------------------
# Main Function
# -------------------------------

async def verify_all_gumroad_subscriptions():
    """
    Daily job: Verify all active Gumroad subscriptions
    Run this every 24 hours via Railway scheduler or cron
    """
    logger.info("Starting Gumroad subscription verification sweep")
    
    # Get all active subscriptions
    result = supabase.table("gumroad_subscriptions").select(
        "user_id",
        "gumroad_license_key",  # ✅ Changed from gumroad_purchase_id
        "plan_key"
    ).eq("status", "active").execute()
    
    if not result.data or not isinstance(result.data, list):
        logger.info("No active subscriptions to verify")
        return
    
    verified = 0
    cancelled = 0
    errors = 0
    
    # Iterate through subscriptions with type safety
    for sub_raw in result.data:
        # Cast to proper type
        sub = cast(SubscriptionRow, sub_raw)
        
        try:
            user_id = safe_get_str(sub, "user_id")
            license_key = safe_get_str(sub, "gumroad_license_key")  # ✅ Changed
            
            if not user_id or not license_key:
                # ✅ SAFE: Log anonymized warning
                user_ref = _safe_user_ref(user_id)
                logger.warning(f"Skipping invalid subscription for {user_ref}")
                continue
            
            # ✅ FIX: Use verify_license() instead of verify_purchase()
            purchase_data = await gumroad_client.verify_license(license_key)
            
            if gumroad_client.is_subscription_active(purchase_data):
                # Update expiry
                expires_at = gumroad_client.calculate_expiry(purchase_data)
                supabase.table("gumroad_subscriptions").update({
                    "expires_at": expires_at,
                    "updated_at": datetime.utcnow().isoformat()  # ✅ Changed column name
                }).eq("user_id", user_id).execute()
                
                verified += 1
                # ✅ SAFE: Log with anonymized user reference
                user_ref = _safe_user_ref(user_id)
                logger.info(f"Subscription verified as active for {user_ref}")
            else:
                # Subscription is no longer active
                supabase.table("gumroad_subscriptions").update({
                    "status": "cancelled",
                    "updated_at": datetime.utcnow().isoformat()  # ✅ Changed column name
                }).eq("user_id", user_id).execute()
                
                cancelled += 1
                # ✅ SAFE: Log with anonymized user reference
                user_ref = _safe_user_ref(user_id)
                logger.info(f"Subscription marked as cancelled for {user_ref}")
                
        except Exception as e:
            # ✅ SAFE: Log error with anonymized references
            user_ref = _safe_user_ref(user_id)
            license_prefix = license_key[:8] + "..." if license_key and len(license_key) > 8 else "invalid_key"
            logger.error(f"Verification failed for user {user_ref} (license: {license_prefix}): {type(e).__name__}")
            errors += 1
    
    logger.info(f"Verification complete: {verified} active, {cancelled} cancelled, {errors} errors")

if __name__ == "__main__":
    asyncio.run(verify_all_gumroad_subscriptions())