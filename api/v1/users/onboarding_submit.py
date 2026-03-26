from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from core.security import get_current_user_id
from db.supabase import supabase

router = APIRouter()

class OnboardingSubmitRequest(BaseModel):
    name: str
    age: str
    gender: str
    support_style: str
    Current_Difficulty: str
    Duration: str
    Daily_Impact: str
    Coping_Style: str
    Support_Network: str
    Safety_Check: str
    email: Optional[str] = None

class OnboardingSubmitResponse(BaseModel):
    user_id: str
    completed: bool = True

@router.post("/onboarding-submit")
async def submit_onboarding(
    payload: OnboardingSubmitRequest,
    user_id: str = Depends(get_current_user_id)
):
    try:
        form_data = {
            "user_id": user_id,
            "name": payload.name,
            "age": payload.age,
            "gender": payload.gender,
            "support_style": payload.support_style,
            "Current_Difficulty": payload.Current_Difficulty,
            "Duration": payload.Duration,
            "Daily_Impact": payload.Daily_Impact,
            "Coping_Style": payload.Coping_Style,
            "Support_Network": payload.Support_Network,
            "Safety_Check": payload.Safety_Check,
            "email": payload.email,
        }

        result = supabase.table("user_info").insert(form_data).execute()

        return {"ok": True}

    except Exception as e:
        print("🔥 ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))