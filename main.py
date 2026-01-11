"""
Main entry point for the FastAPI server.

This module defines the FastAPI application, its endpoints,
and lifecycle management. It now uses LiveKit instead of Daily.co.
"""

import os
import subprocess
import sys
import asyncio
import hashlib  # ⭐ Added for safe referencing
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, cast, List
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from loguru import logger
from pydantic import BaseModel
from core.security import get_current_user_id
from db.supabase import supabase
from fastapi import HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from core.config import settings  # Add this import
from config.server import ServerConfig
from datetime import date
from api.v1.users.subscription import router as user_subscription_router
from api.v1.billing.gumroad import router as billing_router

from fastapi.routing import APIRoute
from fastapi import APIRouter
from livekit import api
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Server configuration
server_config = ServerConfig()

# Check for required LiveKit env vars
if not server_config.livekit_api_key or not server_config.livekit_api_secret:
    logger.error("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in environment")
    raise ValueError("Missing LiveKit API credentials")

# Runtime state
bot_procs: Dict[int, tuple] = {}  # Track bot processes: {pid: (process, room_name, token)}

def _safe_room_ref(room_name: str) -> str:
    """Create a safe reference for room names in logs."""
    if not room_name:
        return "unknown_room"
    return f"room_{hashlib.sha256(room_name.encode()).hexdigest()[:8]}"

class ConversationMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None

class SessionEndRequest(BaseModel):
    user_id: str
    title: str
    summary: str
    session_intensity: int
    

# Configure loguru
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",  # ⭐ Changed from DEBUG to INFO for production
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    enqueue=True,
    backtrace=False,  # ⭐ Disabled for production (security)
    diagnose=False,   # ⭐ Disabled for production (security)
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager. Removed Daily.co session, kept cleanup task.
    """
    cleanup_task = asyncio.create_task(cleanup_finished_processes())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


async def cleanup_finished_processes() -> None:
    """
    Background task to clean up finished bot processes.
    For LiveKit, rooms auto-expire, so we just clean up process tracking.
    """
    while True:
        try:
            for pid in list(bot_procs.keys()):
                proc, room_name, token = bot_procs[pid]
                if proc.poll() is not None:
                    # ✅ SAFE: Log with anonymized room reference
                    room_ref = _safe_room_ref(room_name)
                    logger.info(f"Cleaning up finished bot process {pid} for {room_ref}")
                    # No need to delete LiveKit rooms - they auto-expire
                    del bot_procs[pid]
        except Exception as e:
            # ✅ SAFE: Log exception type only
            logger.error(f"Error during cleanup: {type(e).__name__}")
        await asyncio.sleep(5)


# Create the FastAPI app
app: FastAPI = FastAPI(lifespan=lifespan)


#subscription
app.include_router(user_subscription_router, prefix="/api/v1")

app.include_router(billing_router, prefix="/api/v1")

@app.get("/debug/routes")
def debug_routes():
    routes = []
    for route in app.routes:
        if isinstance(route, APIRoute):
            routes.append({
                "path": route.path,
                "methods": route.methods
            })
    return routes
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def create_livekit_token(room_name: str, identity: str) -> tuple[str, str, str]:
    """
    Create a LiveKit room token for bot and user.
    Returns: (room_name, bot_token, user_token)
    """
    try:
        # Bot token (can publish/subscribe)
        bot_token = api.AccessToken(
            api_key=server_config.livekit_api_key,
            api_secret=server_config.livekit_api_secret,
        )
        bot_token.with_identity(f"bot_{identity}")
        bot_token.with_name("Aletheia")
        bot_token.with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
                
            )
        )

        # User token (can publish/subscribe)
        user_token = api.AccessToken(
            api_key=server_config.livekit_api_key,
            api_secret=server_config.livekit_api_secret,
        )
        user_token.with_identity(f"user_{identity}")
        user_token.with_name("User")
        user_token.with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )

        return room_name, bot_token.to_jwt(), user_token.to_jwt()

    except Exception as e:
        # ✅ SAFE: Log exception type only, not identity details
        logger.error(f"Failed to create LiveKit token: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Token generation failed")


def parse_server_args():
    """Parse server-specific arguments and store remaining args for bot processes"""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host", help="Server host")
    parser.add_argument("--port", type=int, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # Parse known server args and keep remaining for bots
    server_args, remaining_args = parser.parse_known_args()

    # Update server config with parsed args
    global server_config
    if server_args.host:
        server_config.host = server_args.host
    if server_args.port:
        server_config.port = server_args.port
    if server_args.reload:
        server_config.reload = server_config.reload

    

    global bot_args
    bot_args = remaining_args


parse_server_args()


# In main.py

async def start_bot_process(room_name: str, token: str, user_id: str) -> int:
    """Start a bot subprocess with LiveKit credentials"""
    # Check room capacity
    num_bots_in_room = sum(
        1 for proc, name, _ in bot_procs.values() if name == room_name and proc.poll() is None
    )
    if num_bots_in_room >= server_config.max_bots_per_room:
        # ✅ SAFE: Log with anonymized room reference
        room_ref = _safe_room_ref(room_name)
        logger.warning(f"Room {room_ref} at capacity ({server_config.max_bots_per_room} bots)")
        raise HTTPException(
            status_code=429,
            detail=f"Room at capacity ({server_config.max_bots_per_room} bots)",
        )

    try:
        server_dir = os.path.dirname(os.path.abspath(__file__))
        run_helpers_path = os.path.join(server_dir, "runner.py")

        # Build command with LiveKit parameters
        cmd = [
            sys.executable,
            run_helpers_path,
            "-l",
            server_config.livekit_url,
            "-k",
            token,
            "-r",
            room_name,
            *bot_args,
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = server_dir
        env["BOT_USER_ID"] = user_id  # ⭐ Add this line

        proc = subprocess.Popen(cmd, bufsize=1, cwd=server_dir, env=env)
        bot_procs[proc.pid] = (proc, room_name, token)
        
        # ✅ SAFE: Log with anonymized references
        room_ref = _safe_room_ref(room_name)
        logger.info(f"Bot process started (pid: {proc.pid}) for {room_ref}")
        
        return proc.pid
    except Exception as e:
        # ✅ SAFE: Log exception type only
        logger.error(f"Bot startup failed: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Failed to start bot process")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Aletheia Therapy Bot",
        "timestamp": datetime.now().isoformat(),
        "bots_running": len([p for p, _, _ in bot_procs.values() if p.poll() is None]),
    }


@app.get("/")
async def start_agent(request: Request):
    """
    Legacy endpoint - redirects to LiveKit connect flow.
    """
    return RedirectResponse("/connect-livekit")


@app.post("/connect")
async def rtvi_connect(request: Request) -> Dict[str, Any]:
    """DEPRECATED: Daily.co endpoint. Use /connect-livekit instead."""
    raise HTTPException(
        status_code=410,
        detail="Daily.co is deprecated. Use POST /connect-livekit instead.",
    )

@app.post("/connect-livekit")
async def connect_livekit(
    request: Request,
    user_id: str = Depends(get_current_user_id)  # ⭐ Force JWT verification
) -> Dict[str, Any]:
    """
    Create LiveKit room and return connection credentials.
    Requires valid Clerk JWT.
    """
    try:
        data = await request.json()
        room_name = data.get("room", f"therapy-{uuid.uuid4().hex[:8]}")
        
        # ⭐ Use AUTHENTICATED user_id from JWT, NOT from request body
        identity = user_id

        # Create tokens
        room, bot_token, user_token = await create_livekit_token(room_name, identity)

        # Start bot process
        pid = await start_bot_process(room_name, bot_token, identity)

        return {
            "status": "success",
            "room_url": server_config.livekit_url,
            "room_name": room_name,
            "bot_token": bot_token,
            "user_token": user_token,
            "bot_pid": pid,
            "status_endpoint": f"/status/{pid}",
        }

    except Exception as e:
        # ✅ SAFE: Log exception type only
        logger.error(f"LiveKit connect failed: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Connection failed")

@app.get("/get-livekit-token")
async def get_livekit_token_endpoint(room: str = "therapy-room", identity: str = "user"):
    """
    Generate a LiveKit JWT token for a user to join a room.
    """
    try:
        room_name, _, user_token = await create_livekit_token(room, identity)
        return JSONResponse(
            {
                "token": user_token,
                "room": room_name,
                "identity": identity,
                "livekit_url": server_config.livekit_url,
            }
        )

    except Exception as e:
        # ✅ SAFE: Log exception type only
        logger.error(f"LiveKit token generation failed: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Token generation failed")


@app.get("/status/{pid}")
def get_status(pid: int):
    """
    Get the status of a specific bot process.
    """
    proc_tuple = bot_procs.get(pid)
    if not proc_tuple:
        raise HTTPException(status_code=404, detail=f"Bot with process id: {pid} not found")
    proc, room_name, _ = proc_tuple
    status = "running" if proc.poll() is None else "finished"
    return JSONResponse({"bot_id": pid, "status": status, "room": room_name})


@app.post("/session/start")
async def start_session(
    user_id: str = Depends(get_current_user_id),
):
    today = date.today().isoformat()

    supabase.table("therapy_sessions").upsert(
        {
            "user_id": user_id,
            "date": today,
        },
        on_conflict="user_id,date",
    ).execute()

    return {
        "status": "ok",
        "date": today,
    }

@app.post("/session/end")
async def end_session(payload: SessionEndRequest):
    today = date.today().isoformat()

    supabase.table("therapy_sessions").upsert(
        {
            "user_id": payload.user_id,
            "date": today,
            "title": payload.title,
            "summary": payload.summary,
            "session_intensity": payload.session_intensity,
        },
        on_conflict="user_id,date",
    ).execute()

    return {"status": "completed"}

if __name__ == "__main__":
    import os
    import uvicorn

    logger.info("Starting FastAPI server with LiveKit")

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )