# api/v1/users/profile.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
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
    """
    Get user profile info from user_info table.
    Returns empty defaults if no profile exists yet.
    """
    try:
        result = supabase.table("user_info") \
            .select("name, age, gender") \
            .eq("user_id", user_id) \
            .maybe_single() \
            .execute()
        
        if result.data:
            return ProfileResponse(
                name=result.data.get("name"),
                age=result.data.get("age"),
                gender=result.data.get("gender")
            )
        
        # Return defaults if no profile exists
        return ProfileResponse()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch profile: {str(e)}")

@router.post("/profile")
async def update_profile(
    payload: ProfileUpdateRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Create or update user profile in user_info table.
    """
    try:
        # Check if profile exists
        existing = supabase.table("user_info") \
            .select("user_id") \
            .eq("user_id", user_id) \
            .maybe_single() \
            .execute()
        
        if existing.data:
            # Update existing
            supabase.table("user_info") \
                .update({
                    "name": payload.name.strip(),
                    "age": payload.age,
                    "gender": payload.gender,
                }) \
                .eq("user_id", user_id) \
                .execute()
        else:
            # Insert new
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
        raise HTTPException(status_code=500, detail=f"Failed to save profile: {str(e)}")