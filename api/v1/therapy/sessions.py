from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict
from typing import Optional
import time
import traceback
from core.security import get_current_user_id
from db.supabase import supabase

router = APIRouter()

request_history: Dict[str, List[float]] = {}

def check_rate_limit(user_id: str, max_requests: int = 10, window_seconds: int = 60):
    now = time.time()
    user_requests = request_history.get(user_id, [])
    user_requests = [t for t in user_requests if now - t < window_seconds]

    if len(user_requests) >= max_requests:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {max_requests} requests per {window_seconds}s"
        )

    user_requests.append(now)
    request_history[user_id] = user_requests



class SessionRow(BaseModel):
    id: str
    user_id: str
    created_at: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_minutes: Optional[int] = None  # <-- ADD THIS
    session_intensity: int
    title: Optional[str] = None
    summary: Optional[str] = None

class JourneyPoint(BaseModel):
    date: str
    intensity: int

class SessionsResponse(BaseModel):
    sessions: List[SessionRow]
    has_more: bool
    page: int

class JourneyResponse(BaseModel):
    journey: List[JourneyPoint]

class Config:
        from_attributes = True

@router.get("/sessions", response_model=SessionsResponse)
async def get_sessions(
    page: int = Query(0, ge=0),
    page_size: int = Query(5, ge=1, le=50),
    user_id: str = Depends(get_current_user_id)
):
    check_rate_limit(user_id, max_requests=10, window_seconds=60)

    try:
        query = supabase.table("therapy_sessions") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True)  # ✅ fixed

        if page > 0:
            from_idx = page * page_size
            to_idx = from_idx + page_size - 1
            query = query.range(from_idx, to_idx)
        else:
            query = query.limit(page_size)

        result = query.execute()

        sessions = result.data or []
        has_more = len(sessions) == page_size

        normalized = [
            {
                **s,
                "session_intensity": int(s.get("session_intensity") or 1)  
            }
            for s in sessions
        ]

        return SessionsResponse(
            sessions=normalized,
            has_more=has_more,
            page=page
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch sessions: {str(e)}")


@router.get("/journey", response_model=JourneyResponse)
async def get_journey(
    user_id: str = Depends(get_current_user_id)
):
    check_rate_limit(user_id, max_requests=10, window_seconds=60)

    try:
        result = supabase.table("therapy_sessions") \
            .select("session_intensity, created_at") \
            .eq("user_id", user_id) \
            .order("created_at", desc=False)  

        result = result.execute()

        if not result.data:
            return JourneyResponse(journey=[])

        journey = [
            JourneyPoint(
                date=d["created_at"],
                intensity=int(d.get("session_intensity") or 1)  
            )
            for d in result.data
        ]

        return JourneyResponse(journey=journey)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch journey: {str(e)}")


@router.post("/sessions/clear")
async def clear_all_sessions(user_id: str = Depends(get_current_user_id)):
    try:
        result = supabase.table("therapy_sessions") \
            .delete() \
            .eq("user_id", user_id) \
            .execute()

        return {"status": "success", "deleted": len(result.data) if result.data else 0}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to clear sessions: {str(e)}")

@router.get("/sessions/recent", response_model=List[SessionRow])
async def get_recent_sessions(user_id: str = Depends(get_current_user_id)):
    check_rate_limit(user_id, max_requests=10, window_seconds=60)

    try:
        result = supabase.table("therapy_sessions") \
            .select("*") \
            .eq("user_id", user_id) \
            .not_.is_("summary", None) \
            .order("created_at", desc=True) \
            .limit(5) \
            .execute()

        sessions = result.data or []

        normalized = []
        for s in sessions:
            duration = s.get("duration_minutes")
            
            # Calculate if missing/null/0 but timestamps exist
            # FIXED: Use 'end_time' and 'start_time' (not 'ended_at'/'created_at')
            if (duration is None or duration == 0) and s.get("end_time") and s.get("start_time"):
                try:
                    start_str = s["start_time"].replace('Z', '+00:00')
                    end_str = s["end_time"].replace('Z', '+00:00')
                    
                    start = datetime.fromisoformat(start_str)
                    end = datetime.fromisoformat(end_str)
                    
                    calc_duration = int((end - start).total_seconds() / 60)
                    duration = max(1, calc_duration)  # Minimum 1 min
                except Exception as e:
                    print(f"Duration calc error: {e}")
                    duration = 0
            else:
                # Convert existing value to int, default 0 if null
                try:
                    duration = int(duration) if duration is not None else 0
                except (ValueError, TypeError):
                    duration = 0

            normalized.append({
                **s,
                "duration_minutes": duration,
                "session_intensity": int(s.get("session_intensity") or 1)
            })

        return normalized

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to fetch recent sessions")