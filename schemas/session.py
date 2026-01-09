from pydantic import BaseModel
from datetime import datetime


class EndSessionRequest(BaseModel):
    summary: str
    emotion_score: int
    emotion_label: str | None = None
    dominant_emotion: str | None = None
    intensity: int | None = None
    started_at: datetime
