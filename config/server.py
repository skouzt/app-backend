"""Server configuration management module."""

import os
from dotenv import load_dotenv


class ServerConfig:
    def __init__(self):
        load_dotenv()

        # Server settings
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("FAST_API_PORT", "7860"))
        self.reload: bool = os.getenv("RELOAD", "false").lower() == "true"

        # Daily API settings
        self.livekit_api_key = os.getenv("LIVEKIT_API_KEY", "")
        self.livekit_api_secret = os.getenv("LIVEKIT_API_SECRET", "")
        self.livekit_url = os.getenv("LIVEKIT_WS_URL", "").rstrip('/') 
        self.livekit_token = os.getenv("LIVEKIT_TOKEN", "")
        self.daily_api_key = os.getenv("DAILY_API_KEY")
        self.daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")

        # Bot settings
        self.max_bots_per_room: int = int(os.getenv("MAX_BOTS_PER_ROOM", "1"))

        # Validate required settings
        if not self.daily_api_key:
            raise ValueError("DAILY_API_KEY environment variable must be set")
        if self.livekit_api_key and not self.livekit_api_secret:
            raise ValueError("LIVEKIT_API_SECRET required when LIVEKIT_API_KEY is set")