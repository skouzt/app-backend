# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # --- Supabase ---
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_KEY: str = ""
    
    # --- Clerk ---
    CLERK_SECRET_KEY: str = ""
    CLERK_WEBHOOK_SECRET: str = ""
    CLERK_JWT_ISSUER: Optional[str] = None
    
    # --- Gumroad ---
    GUMROAD_API_KEY: str = ""  # From Gumroad Settings → Advanced
    GUMROAD_PRODUCT_ID: str = "iauij"
    GUMROAD_PRODUCT_PERMALINK: str = "aletheia"  # Your product permalink
    GUMROAD_SELLER_ID: str = "JglTMXwp1tOqifxr0BhYdg==" 
    GUMROAD_ACCESS_TOKEN: str = "" 
    # Pricing tiers (in cents)
    GUIDED_PRICE: int = 1500   # $15
    EXTENDED_PRICE: int = 5000  # $50
    
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
    
    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()