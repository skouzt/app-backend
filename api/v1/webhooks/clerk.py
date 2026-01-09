# api/v1/webhooks/clerk.py
from fastapi import APIRouter, Request
import hmac
import hashlib
import base64
import json
from datetime import datetime
from typing import Dict, Any, cast
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
    # IMPORTANT: Add CLERK_WEBHOOK_SECRET to your .env and core.config
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

@router.post("/webhooks/clerk")
async def clerk_webhook_handler(request: Request) -> dict:
    """Receives Clerk subscription events and syncs to user_info table"""
    payload = await request.body()
    headers = dict(request.headers)
    
    try:
        event = verify_clerk_webhook(payload, headers)
        
        event_type = event.get("type")
        data = event.get("data", {})
        user_id = data.get("user_id") or data.get("id")
        
        if not user_id:
            logger.error("clerk_webhook_missing_user_id")
            raise ValueError("No user_id in webhook payload")
        
        # Store event metadata only - NEVER store full payload
        plan_name = data.get("plan", {}).get("name") if isinstance(data.get("plan"), dict) else None
        supabase.table("subscription_events").insert({
            "user_id": user_id,
            "event_type": event_type,
            "plan": plan_name,
            "status": data.get("status"),
        }).execute()
        
        logger.info("clerk_webhook_processed", extra={"user_id": user_id, "event_type": event_type})
        
        # Update user_info based on event type
        if event_type in ["subscription.created", "subscription.updated", "subscription.active"]:
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
            
            supabase.table("user_info").update(update_data).eq("user_id", user_id).execute()
            logger.info("subscription_updated", extra={"user_id": user_id, "status": update_data["subscription_status"]})
            
        elif event_type == "subscription.canceled":
            supabase.table("user_info").update({
                "subscription_status": "canceled",
                "subscription_updated_at": datetime.utcnow().isoformat()
            }).eq("user_id", user_id).execute()
            logger.info("subscription_canceled", extra={"user_id": user_id})
        
        return {"status": "ok"}
        
    except Exception as e:
        logger.error("clerk_webhook_handler_error", exc_info=True)
        return {"status": "error", "details": "Internal error"}