from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
import traceback
from core.security import get_current_user_id
from db.supabase import supabase

router = APIRouter()

class ProfileResponse(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None

class ProfileUpdateRequest(BaseModel):
    name: str
    age: str
    gender: str

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

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch profile: {str(e)}")


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

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save profile: {str(e)}")