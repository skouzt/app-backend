from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import ConfigDict 


class Settings(BaseSettings):
    model_config = ConfigDict(           
        env_file=".env",
        extra="allow"
    )

    # --- Supabase ---
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_KEY: str = ""
    
    # --- Clerk ---
    CLERK_SECRET_KEY: str = ""
    CLERK_WEBHOOK_SECRET: str = ""
    CLERK_JWT_ISSUER: Optional[str] = None
    
    # Dodo Payments
    DODO_PAYMENTS_API_KEY: str = ""
    DODO_WEBHOOK_SECRET: str = ""
    DODO_PLAN_CLARITY_ID: str = ""
    DODO_PLAN_INSIGHT_ID: str = ""
    DODO_ENVIRONMENT: str = "test_mode"
    DODO_DEFAULT_RETURN_URL: str = "aletheia://payment/result"
    
    # --- Application URLs ---
    API_BASE_URL: str = "http://localhost:8000"
    WEB_APP_URL: str = "http://localhost:3000"
    APP_DEEP_LINK: str = "aletheia://"
    
    # --- LiveKit (optional) ---
    LIVEKIT_API_KEY: Optional[str] = None
    LIVEKIT_API_SECRET: Optional[str] = None
    LIVEKIT_URL: Optional[str] = None
    
    # --- Security ---
    ALLOWED_ORIGINS: str = "*"
    
    # ← class Config block deleted


settings = Settings()