# config/env_loader.py
import os
from dotenv import load_dotenv, find_dotenv

def load_env():
    """Load .env file once, safely for both local and cloud."""
    # Only load if not already loaded (check a sentinel var)
    if os.getenv("_ENV_LOADED"):
        return
    
    env_path = find_dotenv()
    if env_path and os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    
    # Mark as loaded
    os.environ["_ENV_LOADED"] = "1"

# Auto-load when imported
load_env()