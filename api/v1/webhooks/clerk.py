# api/v1/webhooks/clerk.py
from fastapi import APIRouter, Request, HTTPException
import hmac
import hashlib
import base64
import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging

from core.config import settings
from db.supabase import supabase

router = APIRouter()
logger = logging.getLogger(__name__)

def verify_clerk_webhook(payload: bytes, headers: dict) -> dict:
    """Verifies Svix webhook signature"""
    svix_id = headers.get("svix-id", "")
    svix_timestamp = headers.get("svix-timestamp", "")
    svix_signature = headers.get("svix-signature", "")
    
    if not all([svix_id, svix_timestamp, svix_signature]):
        logger.warning("clerk_webhook_missing_headers")
        raise ValueError("Missing Svix headers")
    
    signed_content = f"{svix_id}.{svix_timestamp}.{payload.decode()}"
    secret = settings.CLERK_WEBHOOK_SECRET.encode()
    expected_signature = base64.b64encode(
        hmac.new(secret, signed_content.encode(), hashlib.sha256).digest()
    ).decode()
    
    signature_list = svix_signature.split(" ")
    for signature in signature_list:
        if signature == f"v1,{expected_signature}":
            return json.loads(payload)
    
    logger.warning("clerk_webhook_invalid_signature")
    raise ValueError("Invalid webhook signature")


@router.get("/webhooks/clerk/test")
async def test_webhook_endpoint():
    """Test endpoint to verify webhook route is working"""
    return {
        "status": "ok",
        "message": "Webhook endpoint is reachable",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/webhooks/clerk")
async def clerk_webhook_handler(request: Request) -> dict:
    """
    Receives Clerk events:
    - user.created: Creates user_info record with email
    - subscription.*: Updates subscription status
    """
    payload = await request.body()
    headers = dict(request.headers)
    
    try:
        event = verify_clerk_webhook(payload, headers)
        
        event_type = event.get("type")
        data = event.get("data", {})
        user_id = data.get("id") or data.get("user_id")
        
        if not user_id:
            logger.error("clerk_webhook_missing_user_id", extra={"event_type": event_type})
            return {"status": "error", "message": "No user_id"}
        
        logger.info(
            "clerk_webhook_received",
            extra={"user_id": user_id, "event_type": event_type}
        )
        
        # ✅ Handle user.created event
        if event_type == "user.created":
            return await handle_user_created(user_id, data)
        
        # ✅ Handle user.updated event
        elif event_type == "user.updated":
            return await handle_user_updated(user_id, data)
        
        # Handle subscription events
        elif event_type in ["subscription.created", "subscription.updated", "subscription.active"]:
            return await handle_subscription_update(user_id, data, event_type)
            
        elif event_type == "subscription.canceled":
            return await handle_subscription_canceled(user_id, data)
        
        # Unknown event type
        logger.info("clerk_webhook_unhandled_event", extra={"event_type": event_type})
        return {"status": "ok", "message": "Event type not handled"}
        
    except ValueError as e:
        logger.error("clerk_webhook_validation_error", extra={"error": str(e)})
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("clerk_webhook_handler_error", exc_info=True)
        return {"status": "error", "details": "Internal error"}


async def handle_user_created(user_id: str, data: dict) -> dict:
    """Create user_info record when user signs up"""
    
    # Extract email from Clerk webhook
    email_addresses = data.get("email_addresses", [])
    primary_email_id = data.get("primary_email_address_id")
    
    primary_email = None
    
    # Find primary email
    for email_obj in email_addresses:
        if email_obj.get("id") == primary_email_id:
            primary_email = email_obj.get("email_address")
            break
    
    # Fallback to first email if no primary
    if not primary_email and email_addresses:
        primary_email = email_addresses[0].get("email_address")
    
    if not primary_email:
        logger.warning(
            "no_email_in_user_created_webhook",
            extra={"user_id": user_id}
        )
        # Don't fail - onboarding will handle it as fallback
        return {"status": "ok", "message": "No email found"}
    
    try:
        # Check if user already exists
        existing = supabase.table("user_info")\
            .select("user_id")\
            .eq("user_id", user_id)\
            .execute()
        
        if existing.data and len(existing.data) > 0:
            logger.info("user_already_exists", extra={"user_id": user_id})
            return {"status": "ok", "message": "User already exists"}
        
        # Create user_info record with email
        # Onboarding will fill in the rest later
        result = supabase.table("user_info").insert({
            "user_id": user_id,
            "email": primary_email,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        
        if not result.data:
            logger.error("failed_to_create_user", extra={"user_id": user_id})
            return {"status": "error", "message": "Database insert failed"}
        
        logger.info(
            "user_created_from_webhook",
            extra={"user_id": user_id, "email": primary_email}
        )
        
        return {"status": "ok", "user_id": user_id}
        
    except Exception as e:
        logger.error(
            "user_creation_error",
            extra={"user_id": user_id, "error": str(e)},
            exc_info=True
        )
        # Don't fail the webhook
        return {"status": "ok", "message": "Error but continuing"}


async def handle_user_updated(user_id: str, data: dict) -> dict:
    """Update user email if it changed"""
    
    email_addresses = data.get("email_addresses", [])
    primary_email_id = data.get("primary_email_address_id")
    
    primary_email = None
    for email_obj in email_addresses:
        if email_obj.get("id") == primary_email_id:
            primary_email = email_obj.get("email_address")
            break
    
    if not primary_email:
        return {"status": "ok"}
    
    try:
        # Update email if user exists
        supabase.table("user_info")\
            .update({"email": primary_email})\
            .eq("user_id", user_id)\
            .execute()
        
        logger.info(
            "user_email_updated",
            extra={"user_id": user_id, "email": primary_email}
        )
        
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(
            "user_update_error",
            extra={"user_id": user_id, "error": str(e)}
        )
        return {"status": "ok"}


async def handle_subscription_update(user_id: str, data: dict, event_type: str) -> dict:
    """Handle subscription created/updated/active events"""
    
    try:
        # Store event metadata
        plan_name = data.get("plan", {}).get("name") if isinstance(data.get("plan"), dict) else None
        
        supabase.table("subscription_events").insert({
            "user_id": user_id,
            "event_type": event_type,
            "plan": plan_name,
            "status": data.get("status"),
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        
        # Update user_info
        plan = data.get("plan", {})
        current_period_end = data.get("current_period_end")
        
        update_data = {
            "subscription_plan": plan.get("name", "free"),
            "subscription_status": data.get("status", "active"),
            "subscription_updated_at": datetime.utcnow().isoformat()
        }
        
        if current_period_end:
            update_data["subscription_expires_at"] = datetime.fromtimestamp(
                current_period_end / 1000
            ).isoformat()
        
        supabase.table("user_info")\
            .update(update_data)\
            .eq("user_id", user_id)\
            .execute()
        
        logger.info(
            "subscription_updated",
            extra={"user_id": user_id, "status": update_data["subscription_status"]}
        )
        
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(
            "subscription_update_error",
            extra={"user_id": user_id, "error": str(e)},
            exc_info=True
        )
        return {"status": "error"}


async def handle_subscription_canceled(user_id: str, data: dict) -> dict:
    """Handle subscription canceled event"""
    
    try:
        supabase.table("user_info").update({
            "subscription_status": "canceled",
            "subscription_updated_at": datetime.utcnow().isoformat()
        }).eq("user_id", user_id).execute()
        
        logger.info("subscription_canceled", extra={"user_id": user_id})
        
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(
            "subscription_cancel_error",
            extra={"user_id": user_id, "error": str(e)}
        )
        return {"status": "error"}