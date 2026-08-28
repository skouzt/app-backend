"""Push device registration.

    POST   /notifications/token        register this device
    DELETE /notifications/token        forget it (sign-out)
    GET    /notifications/preferences  what Settings should show
    PUT    /notifications/preferences  the Settings toggle

Registration is separate from the preference on purpose. A token is a fact about
a device and changes on reinstall; the preference is the person's answer and
should survive that.
"""

import traceback

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from core.security import get_current_user_id
from services import push_service
from services.chat_service import in_thread

router = APIRouter()


class TokenRequest(BaseModel):
    token: str = Field(min_length=1, max_length=255)
    platform: str = Field(pattern="^(ios|android)$")


class PreferenceRequest(BaseModel):
    enabled: bool


class PreferenceResponse(BaseModel):
    enabled: bool
    devices: int


@router.post("/notifications/token", status_code=204)
async def register_device(
    payload: TokenRequest, user_id: str = Depends(get_current_user_id)
) -> None:
    if not push_service.is_valid_token(payload.token):
        # A malformed token is an address nothing can deliver to. Rejecting it
        # here keeps the send path from failing later against a row that was
        # never usable.
        raise HTTPException(status_code=422, detail="Not an Expo push token")

    try:
        await in_thread(push_service.register_token, user_id, payload.token, payload.platform)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to register device")


@router.delete("/notifications/token", status_code=204)
async def unregister_device(
    token: str = Query(min_length=1, max_length=255),
    user_id: str = Depends(get_current_user_id),
) -> None:
    """The token is a query parameter, not a body.

    A request body on DELETE is legal but widely dropped by proxies and CDNs,
    which would make sign-out fail to detach the device in production and only
    sometimes — the worst kind of bug to chase.
    """
    try:
        await in_thread(push_service.unregister_token, user_id, token)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to remove device")


@router.get("/notifications/preferences", response_model=PreferenceResponse)
async def read_preferences(user_id: str = Depends(get_current_user_id)) -> PreferenceResponse:
    try:
        state = await in_thread(push_service.get_state, user_id)
        return PreferenceResponse(**state)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to read preferences")


@router.put("/notifications/preferences", response_model=PreferenceResponse)
async def update_preferences(
    payload: PreferenceRequest, user_id: str = Depends(get_current_user_id)
) -> PreferenceResponse:
    try:
        await in_thread(push_service.set_enabled, user_id, payload.enabled)
        state = await in_thread(push_service.get_state, user_id)
        return PreferenceResponse(**state)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to update preferences")
