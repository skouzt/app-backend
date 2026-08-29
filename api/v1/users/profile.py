from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import traceback
from core.security import get_current_user_id
from db.supabase import supabase
from services.user_info_service import set_timezone

router = APIRouter()

class ProfileResponse(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None

class ProfileUpdateRequest(BaseModel):
    name: str
    age: str
    gender: str


class TimezoneRequest(BaseModel):
    # An IANA zone name as the device reports it, e.g. "Asia/Kolkata".
    timezone: str = Field(min_length=1, max_length=64)

@router.get("/profile", response_model=ProfileResponse)
async def get_profile(user_id: str = Depends(get_current_user_id)):
    try:
        result = supabase.table("user_info") \
            .select("name, age, gender") \
            .eq("user_id", user_id) \
            .limit(1) \
            .execute()

        if result.data and len(result.data) > 0:
            row = result.data[0]
            return ProfileResponse(
                name=row.get("name"),
                age=row.get("age"),
                gender=row.get("gender")
            )

        return ProfileResponse()

    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to fetch profile")


@router.post("/profile")
async def update_profile(
    payload: ProfileUpdateRequest,
    user_id: str = Depends(get_current_user_id)
):
    try:
        existing = supabase.table("user_info") \
            .select("user_id") \
            .eq("user_id", user_id) \
            .limit(1) \
            .execute()

        if existing.data and len(existing.data) > 0:
            supabase.table("user_info") \
                .update({
                    "name": payload.name.strip(),
                    "age": payload.age,
                    "gender": payload.gender,
                }) \
                .eq("user_id", user_id) \
                .execute()
        else:
            supabase.table("user_info") \
                .insert({
                    "user_id": user_id,
                    "name": payload.name.strip(),
                    "age": payload.age,
                    "gender": payload.gender,
                }) \
                .execute()

        return {"status": "success"}

    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to save profile")

@router.put("/timezone", status_code=204)
async def update_timezone(
    payload: TimezoneRequest, user_id: str = Depends(get_current_user_id)
) -> None:
    """Record where the user is, so a day ends where they are.

    Sessions cover one local day and the reaper closes them once that date has
    passed, which it does while nobody is online to ask — so the zone has to be
    stored rather than read from a request header.
    """
    try:
        set_timezone(user_id, payload.timezone)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to save timezone")
