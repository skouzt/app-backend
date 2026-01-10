# api/v1/billing/gumroad.py
import json
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, RedirectResponse
import httpx
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, cast, TypedDict, List
import structlog

from core.config import settings
from core.security import verify_clerk_token
from core.billing.gumroad_client import GumroadClient
from db.supabase import supabase
from postgrest.types import ReturnMethod

router = APIRouter()
gumroad_client = GumroadClient()
logger = structlog.get_logger(__name__)


# -------------------------------
# Type Definitions
# -------------------------------

class UserInfoRow(TypedDict, total=False):
    user_id: str
    email: str

class GumroadSubscriptionRow(TypedDict, total=False):
    id: str
    user_id: str
    gumroad_license_key: str
    gumroad_subscription_id: str
    plan_key: str
    status: str
    expires_at: str

class CreateCheckoutRequest(BaseModel):
    plan_key: str

class ActivateLicenseRequest(BaseModel):
    license_key: str

class SubscriptionStatusResponse(BaseModel):
    status: str
    plan: str
    expires_at: Optional[str] = None
    daily_minutes: int

class GumroadPingPayload(BaseModel):
    """Gumroad ping payload structure"""
    seller_id: str
    product_id: str
    product_name: str
    permalink: str
    product_permalink: str
    email: str
    price: int
    gumroad_fee: int
    currency: str
    quantity: int
    discover_fee_charged: bool
    can_contact: bool
    referrer: str
    card: Dict[str, Any]
    order_number: int
    sale_id: str
    sale_timestamp: str
    purchaser_id: str
    subscription_id: Optional[str] = None
    license_key: str
    ip_country: str
    recurrence: Optional[str] = None
    is_gift_receiver_purchase: bool
    refunded: bool
    disputed: bool
    dispute_won: bool
    cancelled: bool
    ended: bool

# -------------------------------
# Configuration
# -------------------------------

PLAN_CONFIG: Dict[str, Dict[str, Any]] = {
    "guided": {"daily_minutes": 60, "name": "Guided"},
    "extended": {"daily_minutes": 480, "name": "Extended"}
}

# -------------------------------
# Helper Functions
# -------------------------------

def safe_get_str(data: Any, key: str, default: str = "") -> str:
    """Safely get string value with type checking"""
    if not isinstance(data, dict):
        return default
    value = data.get(key)
    return str(value) if value is not None else default

def safe_get_list(data: Any) -> List[Any]:
    """Safely get list with type checking"""
    if not isinstance(data, list):
        return []
    return data

# -------------------------------
# Core Helper: Verify and Activate
# -------------------------------

async def verify_and_activate_license(
    license_key: str, 
    user_id: str,
    user_email: str
) -> Dict[str, Any]:
    """
    Verify a license key with Gumroad and activate subscription.
    
    Returns:
        {
            "success": bool,
            "plan": str | None,
            "message": str
        }
    """
    try:
        # Verify with Gumroad API
        purchase_data = await gumroad_client.verify_license(license_key)
        
        # Check if verification succeeded
        if not purchase_data.get("success"):
            logger.warning("license_verification_failed", extra={"user_id": user_id, "reason": "invalid_key"})
            return {
                "success": False,
                "plan": None,
                "message": "Invalid license key"
            }
        
        # Check if subscription is active
        if not gumroad_client.is_subscription_active(purchase_data):
            logger.warning("subscription_not_active", extra={"user_id": user_id})
            return {
                "success": False,
                "plan": None,
                "message": "Subscription is not active (cancelled, refunded, or expired)"
            }
        
        # Get plan from price
        plan_key = gumroad_client.get_plan_from_purchase(purchase_data)
        if not plan_key:
            logger.error("plan_determination_failed", extra={"user_id": user_id})
            return {
                "success": False,
                "plan": None,
                "message": "Unable to determine subscription plan"
            }
        
        # Extract subscription details
        purchase = purchase_data.get("purchase", {})
        subscription_id = gumroad_client.extract_subscription_id(purchase_data)
        expires_at = gumroad_client.calculate_expiry(purchase_data)
        
        # Store/update subscription in database
        subscription_data = {
            "user_id": user_id,
            "gumroad_license_key": license_key,
            "gumroad_subscription_id": subscription_id or "",
            "plan_key": plan_key,
            "status": "active",
            "expires_at": expires_at,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Check if subscription already exists
        existing = supabase.table("gumroad_subscriptions")\
            .select("id")\
            .eq("user_id", user_id)\
            .execute()
        
        if existing.data and len(existing.data) > 0:
            # Update existing
            supabase.table("gumroad_subscriptions")\
                .update(subscription_data)\
                .eq("user_id", user_id)\
                .execute()
        else:
            # Insert new
            subscription_data["id"] = str(uuid.uuid4())
            supabase.table("gumroad_subscriptions")\
                .insert(subscription_data)\
                .execute()
        
        # Clean up any pending verifications
        try:
            supabase.table("pending_verifications")\
                .delete()\
                .eq("user_id", user_id)\
                .execute()
        except:
            pass  # Non-critical
        
        logger.info("subscription_activated", extra={"user_id": user_id, "plan": plan_key})
        
        return {
            "success": True,
            "plan": plan_key,
            "message": f"Successfully activated {PLAN_CONFIG[plan_key]['name']} plan"
        }
        
    except Exception as e:
        logger.error("license_activation_error", extra={"user_id": user_id}, exc_info=True)
        return {
            "success": False,
            "plan": None,
            "message": f"Verification error: {str(e)}"
        }

# -------------------------------
# Routes
# -------------------------------
@router.post("/billing/create-checkout")
async def create_gumroad_checkout(
    request: CreateCheckoutRequest,
    user: dict = Depends(verify_clerk_token)
):
    # Extract user identifiers (never log full JWT or email)
    user_id = user.get("user_id") or user.get("sub")
    email = user.get("email")
    
    if not user_id:
        logger.error("checkout_failed_missing_user_id")
        raise HTTPException(400, "User ID not found")
    
    if not email:
        # Fetch from database if not in JWT
        user_result = supabase.table("user_info")\
            .select("email")\
            .eq("user_id", user_id)\
            .execute()
        
        if user_result.data and len(user_result.data) > 0:
            user_row = user_result.data[0]
            if isinstance(user_row, dict):
                email = str(user_row.get("email", ""))
    
    if not email or not isinstance(email, str):
        logger.error("checkout_failed_missing_email", extra={"user_id": user_id})
        raise HTTPException(400, "Email not found")

    if request.plan_key not in PLAN_CONFIG:
        logger.warning("checkout_invalid_plan", extra={"user_id": user_id, "plan_key": request.plan_key})
        raise HTTPException(400, f"Invalid plan: {request.plan_key}")

    # Generate checkout URL (URL itself contains sensitive params, don't log it)
    checkout_url = gumroad_client.get_checkout_url(request.plan_key, email)
    logger.info("checkout_url_generated", extra={"user_id": user_id, "plan": request.plan_key})

    # Store pending intent
    try:
        supabase.table("pending_verifications").insert({
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "plan_key": request.plan_key,
            "created_at": datetime.utcnow().isoformat(),
        }).execute()
        logger.info("pending_verification_stored", extra={"user_id": user_id})
    except Exception as e:
        logger.warning("pending_verification_store_failed", extra={"user_id": user_id}, exc_info=True)

    return {"url": checkout_url}


@router.post("/billing/activate-license")
async def activate_license(
    request: ActivateLicenseRequest,
    user: dict = Depends(verify_clerk_token)
):
    """
    Activate a subscription using a Gumroad license key.
    User pastes their license key from Gumroad email.
    """
    user_id = user.get("user_id") or user.get("sub")
    email = user.get("email", "")
    
    if not user_id:
        raise HTTPException(400, "User ID not found")
    
    result = await verify_and_activate_license(
        request.license_key,
        user_id,
        email
    )
    
    if not result["success"]:
        raise HTTPException(400, result["message"])
    
    return {
        "status": "success",
        "plan": result["plan"],
        "message": result["message"],
        "daily_minutes": PLAN_CONFIG[result["plan"]]["daily_minutes"]
    }


@router.post("/billing/check-and-activate")
async def check_and_activate_subscription(
    user: dict = Depends(verify_clerk_token)
):
    """
    Check if user has a new Gumroad subscription (including free trials).
    Called after user returns from checkout.
    """
    user_id = user.get("user_id") or user.get("sub")
    
    if not user_id:
        raise HTTPException(400, "User ID not found")
    
    # Get user email
    email: Optional[str] = user.get("email")
    
    if not email:
        user_result = supabase.table("user_info")\
            .select("email")\
            .eq("user_id", user_id)\
            .execute()
        
        if user_result.data and len(user_result.data) > 0:
            user_row = user_result.data[0]
            if isinstance(user_row, dict):
                email = str(user_row.get("email", ""))
    
    if not email:
        raise HTTPException(400, "Email not found")
    
    try:
        # ✅ FIX: Use GUMROAD_ACCESS_TOKEN instead of GUMROAD_API_KEY
        async with httpx.AsyncClient() as client:
            res = await client.get(
                "https://api.gumroad.com/v2/sales",
                params={
                    "access_token": settings.GUMROAD_ACCESS_TOKEN,  # ✅ Changed this
                    "email": email,
                },
                timeout=10,
            )
            
            # ✅ Add better error logging
            if not res.is_success:
                logger.error(
                    f"Gumroad API error: {res.status_code}",
                    extra={
                        "user_id": user_id,
                        "status": res.status_code,
                        "response": res.text[:500]  # Log first 500 chars
                    }
                )
                raise HTTPException(500, f"Gumroad API error: {res.status_code}")
            
            data = res.json()
        
        if not data.get("success"):
            logger.info("gumroad_sales_query_no_results", extra={"user_id": user_id})
            return {"found": False, "message": "No subscription found"}
        
        sales = data.get("sales", [])
        
        if not sales or len(sales) == 0:
            logger.info("gumroad_no_purchases_found", extra={"user_id": user_id})
            return {"found": False, "message": "No purchases found"}
        
        # Get the most recent sale
        latest_sale = sales[0]
        
        # Check if it's a subscription
        subscription_id = latest_sale.get("subscription_id")
        if not subscription_id:
            logger.info("gumroad_purchase_not_subscription", extra={"user_id": user_id})
            return {"found": False, "message": "Not a subscription"}
        
        # Extract details
        # Extract price (Gumroad returns it as cents)
        price_value = latest_sale.get("price", 0)
        try:
            price = int(price_value) if isinstance(price_value, (int, str)) else 0
        except (ValueError, TypeError):
            price = 0

        # Determine plan - check exact prices first
        plan_key = None

        if price == settings.EXTENDED_PRICE:  # 5000 cents = $50
            plan_key = "extended"
            logger.info(f"Detected Extended plan from price: {price}")
            
        elif price == settings.GUIDED_PRICE:  # 1500 cents = $15
            plan_key = "guided"
            logger.info(f"Detected Guided plan from price: {price}")
            
        elif price == 0:
            # Free trial - check variants to determine plan
            variants = latest_sale.get("variants", {})
            tier = str(variants.get("Tier", ""))
            
            logger.info(f"Free trial detected, checking variants: {tier}")
            
            if "Extended" in tier:
                plan_key = "extended"
            elif "Guided" in tier:
                plan_key = "guided"
            else:
                logger.warning(f"Cannot determine plan from variants: {variants}")
                return {
                    "found": False,
                    "message": "Cannot determine plan for free trial"
                }
        else:
            # Unknown price
            logger.warning(f"Unknown price encountered: {price} (expected 1500 or 5000)")
            return {
                "found": False,
                "message": f"Unknown price: {price}"
            }

        if not plan_key:
            return {"found": False, "message": "Failed to determine plan"}

        logger.info(f"Final plan determined: {plan_key}")
        return {
            "found": True,
            "activated": True,
            "plan": plan_key,
            "daily_minutes": PLAN_CONFIG[plan_key]["daily_minutes"]
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(
            "check_and_activate_error",
            extra={"user_id": user_id, "error": str(e)},
            exc_info=True
        )
        raise HTTPException(500, f"Failed to check subscription: {str(e)}")
    

@router.post("/billing/gumroad/ping")
async def gumroad_ping_handler(request: Request):
    """
    Gumroad Ping endpoint - receives notifications about purchases/cancellations.
    
    Set this URL in Gumroad Dashboard:
    Settings → Advanced → Ping → https://your-api.com/api/v1/billing/gumroad/ping 
    """
    try:
        # Parse the ping payload (never log full payload)
        data = await request.form()
        
        # Validate seller_id for security
        seller_id = data.get("seller_id")
        if seller_id != settings.GUMROAD_SELLER_ID:
            logger.warning("gumroad_ping_invalid_seller", extra={"seller_id": seller_id})
            return {"status": "error", "message": "Invalid seller"}
        
        # Extract key fields
        email = data.get("email")
        license_key = data.get("license_key")
        subscription_id = data.get("subscription_id")
        sale_id = data.get("sale_id")
        
        # Handle test pings (no license_key)
        is_test = data.get("test") == "true"
        
        if not email:
            logger.error("gumroad_ping_missing_email")
            return {"status": "error", "message": "Missing email"}
        
        # Test ping - acknowledge without processing
        if is_test and not license_key:
            logger.info("gumroad_test_ping_received")
            return {"status": "ok", "message": "Test ping acknowledged"}
        
        # Real purchase requires license_key
        if not license_key:
            logger.error("gumroad_ping_missing_license_key")
            return {"status": "error", "message": "Missing license_key"}
        
        # Get price
        price_value = data.get("price", "0")
        try:
            price = int(str(price_value))
        except (ValueError, TypeError):
            logger.error("gumroad_ping_invalid_price", extra={"price_value": price_value})
            price = 0
        
        cancelled = data.get("cancelled") == "true"
        refunded = data.get("refunded") == "true"
        ended = data.get("ended") == "true"
        
        # Find user by email (email is query param, not logged)
        user_result = supabase.table("user_info")\
            .select("user_id")\
            .eq("email", email)\
            .execute()
        
        if not user_result.data or len(user_result.data) == 0:
            logger.info("gumroad_ping_user_not_found")
            return {"status": "ok", "message": "User not found, will be activated when user signs up"}
        
        user_row = user_result.data[0] if user_result.data else {}
        user_id = user_row.get("user_id") if isinstance(user_row, dict) else None
        
        if not user_id:
            logger.warning("gumroad_ping_user_id_missing")
            return {"status": "ok", "message": "User not found"}
        
        # Determine plan from price
        plan_key = None
        if price == settings.GUIDED_PRICE:
            plan_key = "guided"
        elif price == settings.EXTENDED_PRICE:
            plan_key = "extended"
        
        if not plan_key:
            logger.error("gumroad_ping_unknown_price", extra={"price": price})
            return {"status": "error", "message": "Unknown price"}
        
        # Determine status
        status = "cancelled" if (cancelled or refunded or ended) else "active"
        
        # Update or insert subscription
        subscription_data = {
            "user_id": user_id,
            "gumroad_license_key": license_key,  # Stored in DB, never logged
            "gumroad_subscription_id": subscription_id or sale_id or "",
            "plan_key": plan_key,
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        existing = supabase.table("gumroad_subscriptions")\
            .select("id")\
            .eq("user_id", user_id)\
            .execute()
        
        if existing.data and len(existing.data) > 0:
            supabase.table("gumroad_subscriptions")\
                .update(subscription_data)\
                .eq("user_id", user_id)\
                .execute()
            logger.info("gumroad_subscription_updated", extra={"user_id": user_id, "status": status, "plan": plan_key})
        else:
            subscription_data["id"] = str(uuid.uuid4())
            subscription_data["created_at"] = datetime.utcnow().isoformat()
            supabase.table("gumroad_subscriptions")\
                .insert(subscription_data)\
                .execute()
            logger.info("gumroad_subscription_created", extra={"user_id": user_id, "plan": plan_key})
        
        return {"status": "ok"}
        
    except Exception as e:
        logger.error("gumroad_ping_handler_error", exc_info=True)
        return {"status": "error", "message": "Internal error"}

@router.post("/billing/verify-subscription")
async def verify_existing_subscription(
    user: dict = Depends(verify_clerk_token)
):
    """
    Manually verify current user's subscription with Gumroad.
    Useful for checking if subscription is still active.
    """
    user_id = user.get("user_id") or user.get("sub")
    
    if not user_id:
        raise HTTPException(400, "User ID not found")
    
    # Get existing subscription
    result = supabase.table("gumroad_subscriptions")\
        .select("*")\
        .eq("user_id", user_id)\
        .execute()
    
    if not result.data or len(result.data) == 0:
        logger.info("verify_subscription_no_subscription", extra={"user_id": user_id})
        return {"status": "none", "message": "No subscription found"}
    
    sub = cast(GumroadSubscriptionRow, result.data[0])
    license_key = safe_get_str(sub, "gumroad_license_key")
    
    if not license_key:
        logger.warning("verify_subscription_missing_license_key", extra={"user_id": user_id})
        return {"status": "error", "message": "No license key found"}
    
    try:
        # Verify with Gumroad
        purchase_data = await gumroad_client.verify_license(license_key)
        
        is_active = gumroad_client.is_subscription_active(purchase_data)
        
        if not is_active:
            # Update status to cancelled
            supabase.table("gumroad_subscriptions").update({
                "status": "cancelled",
                "updated_at": datetime.utcnow().isoformat()
            }).eq("user_id", user_id).execute()
            
            logger.info("subscription_marked_cancelled", extra={"user_id": user_id})
            return {"status": "cancelled", "message": "Subscription is no longer active"}
        
        # Update expiry date
        expires_at = gumroad_client.calculate_expiry(purchase_data)
        supabase.table("gumroad_subscriptions").update({
            "status": "active",
            "expires_at": expires_at,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("user_id", user_id).execute()
        
        logger.info("subscription_verified_active", extra={"user_id": user_id, "plan": safe_get_str(sub, "plan_key")})
        
        return {
            "status": "active",
            "plan": safe_get_str(sub, "plan_key"),
            "expires_at": expires_at
        }
        
    except Exception as e:
        logger.error("subscription_verification_error", extra={"user_id": user_id}, exc_info=True)
        return {"status": "error", "message": "Verification failed"}

@router.get("/billing/me/subscription", response_model=SubscriptionStatusResponse)
async def get_my_subscription(user: dict = Depends(verify_clerk_token)):
    """
    Get current user's subscription status.
    Used by frontend to display plan info.
    """
    user_id = user.get("user_id") or user.get("sub")
    
    if not user_id:
        raise HTTPException(400, "User ID not found")
    
    result = supabase.table("gumroad_subscriptions").select(
        "plan_key, status, expires_at"
    ).eq("user_id", user_id).execute()
    
    if not result.data or len(result.data) == 0:
        return SubscriptionStatusResponse(
            status="none",
            plan="none",
            daily_minutes=0
        )
    
    row = cast(GumroadSubscriptionRow, result.data[0])
    plan = safe_get_str(row, "plan_key", "none")
    
    if plan not in PLAN_CONFIG:
        logger.warning("unknown_plan_in_database", extra={"user_id": user_id, "plan": plan})
        plan = "none"
    
    return SubscriptionStatusResponse(
        status=safe_get_str(row, "status", "inactive"),
        plan=plan,
        expires_at=safe_get_str(row, "expires_at"),
        daily_minutes=PLAN_CONFIG.get(plan, {}).get("daily_minutes", 0)
    )