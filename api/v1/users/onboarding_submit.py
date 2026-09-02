from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
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

        # Upsert, not insert. Onboarding is re-run more often than it looks —
        # after a failed status check, a reinstall, or a double tap on Finish —
        # and inserting left a second row each time. Since fetch_user_info()
        # reads with limit(1), a duplicate meant Lily could pick up whichever
        # row Postgres happened to return, including answers the user replaced.
        # Columns absent here (timezone) are left untouched by the update.
        supabase.table("user_info").upsert(form_data, on_conflict="user_id").execute()

        return {"ok": True}

    except Exception as e:
        # The Postgres error names tables, columns and constraints. That belongs
        # in the log, not in a response body.
        logger.error(f"onboarding submit failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Could not save your answers")