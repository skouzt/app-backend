from fastapi import APIRouter, Depends, HTTPException
import traceback
from pydantic import BaseModel
from core.security import get_current_user_id
from db.supabase import supabase

router = APIRouter()

class OnboardingStatusResponse(BaseModel):
    completed: bool

@router.get("/onboarding-status", response_model=OnboardingStatusResponse)
async def get_onboarding_status(user_id: str = Depends(get_current_user_id)):
    try:
        result = supabase.table("user_info") \
            .select("id") \
            .eq("user_id", user_id) \
            .limit(1) \
            .execute()

        completed = result.data is not None and len(result.data) > 0
        return OnboardingStatusResponse(completed=completed)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to check onboarding status: {str(e)}")