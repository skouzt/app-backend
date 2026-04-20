import asyncio
import os
import subprocess
import sys
import hashlib
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from datetime import datetime, timezone, date

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.routing import APIRoute
from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel
from livekit import api
from dotenv import load_dotenv

from core.security import get_current_user_id, get_internal_or_user_id
from core.config import settings
from config.server import ServerConfig
from db.supabase import supabase

from api.v1.users.subscription import router as user_subscription_router
from api.v1.users.profile import router as profile_router
from api.v1.users.onboarding import router as onboarding_router
from api.v1.users.onboarding_submit import router as onboarding_submit_router
from api.v1.therapy.sessions import router as therapy_sessions_router
from api.v1.usage import router as usage_router
from api.v1.billing.dodo import router as billing_router

load_dotenv()

server_config = ServerConfig()

if not server_config.livekit_api_key or not server_config.livekit_api_secret:
    logger.error("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set")
    raise ValueError("Missing LiveKit API credentials")


# ──────────────────────────────────────────────────────────────────────────────
# BOT POOL
# ──────────────────────────────────────────────────────────────────────────────

POOL_SIZE = 3  # keep 3 bots pre-warmed at all times

# pid → (proc, room_name | None, token | None)
# room_name is None while bot is idle/pre-warmed
bot_procs: Dict[int, tuple] = {}

# Queue of PIDs that are warmed up and waiting for a room assignment
idle_bot_queue: asyncio.Queue = asyncio.Queue()


def _safe_room_ref(room_name: str) -> str:
    if not room_name:
        return "unknown_room"
    return f"room_{hashlib.sha256(room_name.encode()).hexdigest()[:8]}"


async def _spawn_idle_bot() -> int:
    """Spawn one bot process in idle/waiting mode (no room yet)."""
    server_dir = os.path.dirname(os.path.abspath(__file__))
    run_helpers_path = os.path.join(server_dir, "runner.py")

    cmd = [
        sys.executable,
        run_helpers_path,
        "--idle",           # ← runner.py must handle this flag (see below)
        "-l", server_config.livekit_url,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = server_dir

    proc = subprocess.Popen(cmd, bufsize=1, cwd=server_dir, env=env)
    bot_procs[proc.pid] = (proc, None, None)
    logger.info(f"Pre-warmed idle bot spawned (pid: {proc.pid})")
    return proc.pid


async def _assign_room_to_bot(pid: int, room_name: str, token: str, user_id: str) -> None:
    """Write room assignment to bot via a simple file signal."""
    server_dir = os.path.dirname(os.path.abspath(__file__))
    signal_path = os.path.join(server_dir, f".bot_assign_{pid}")

    # Write assignment file — runner.py polls for this
    with open(signal_path, "w") as f:
        import json
        json.dump({
            "room": room_name,
            "token": token,
            "user_id": user_id,
            "livekit_url": server_config.livekit_url,
        }, f)

    bot_procs[pid] = (bot_procs[pid][0], room_name, token)
    logger.info(f"Assigned {_safe_room_ref(room_name)} to bot pid {pid}")


async def _replenish_pool() -> None:
    """Keep idle_bot_queue topped up to POOL_SIZE."""
    current_idle = idle_bot_queue.qsize()
    needed = POOL_SIZE - current_idle
    for _ in range(needed):
        try:
            pid = await _spawn_idle_bot()
            # Give the process 3s to do its imports before marking as ready
            await asyncio.sleep(3)
            if bot_procs.get(pid) and bot_procs[pid][0].poll() is None:
                await idle_bot_queue.put(pid)
                logger.info(f"Bot pid {pid} is warmed and idle (queue size: {idle_bot_queue.qsize()})")
            else:
                logger.warning(f"Bot pid {pid} died during warmup")
        except Exception as e:
            logger.error(f"Pool replenish failed: {type(e).__name__}: {e}")


async def _pool_manager() -> None:
    """Background task: maintain the pre-warm pool."""
    while True:
        try:
            await _replenish_pool()
        except Exception as e:
            logger.error(f"Pool manager error: {type(e).__name__}")
        await asyncio.sleep(10)


async def cleanup_finished_processes() -> None:
    while True:
        try:
            for pid in list(bot_procs.keys()):
                proc, room_name, token = bot_procs[pid]
                if proc.poll() is not None:
                    room_ref = _safe_room_ref(room_name or "")
                    logger.info(f"Cleaning up finished bot {pid} for {room_ref}")
                    del bot_procs[pid]
        except Exception as e:
            logger.error(f"Cleanup error: {type(e).__name__}")
        await asyncio.sleep(5)


# ──────────────────────────────────────────────────────────────────────────────
# LIFESPAN
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the pool at startup before any requests come in
    logger.info(f"Pre-warming {POOL_SIZE} bot processes...")
    await _replenish_pool()
    logger.info("Bot pool ready")

    cleanup_task = asyncio.create_task(cleanup_finished_processes())
    pool_task = asyncio.create_task(_pool_manager())

    try:
        yield
    finally:
        cleanup_task.cancel()
        pool_task.cancel()
        for task in [cleanup_task, pool_task]:
            try:
                await task
            except asyncio.CancelledError:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────────────────────────────────────

app: FastAPI = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_subscription_router, prefix="/api/v1")
app.include_router(onboarding_router, prefix="/api/v1/users")
app.include_router(profile_router, prefix="/api/v1/users")
app.include_router(therapy_sessions_router, prefix="/api/v1/therapy")
app.include_router(usage_router, prefix="/api/v1")
app.include_router(billing_router, prefix="/api/v1")
app.include_router(onboarding_submit_router, prefix="/api/v1/users")


# ──────────────────────────────────────────────────────────────────────────────
# LIVEKIT HELPERS
# ──────────────────────────────────────────────────────────────────────────────

async def create_livekit_token(room_name: str, identity: str) -> tuple[str, str, str]:
    try:
        bot_token = api.AccessToken(
            api_key=server_config.livekit_api_key,
            api_secret=server_config.livekit_api_secret,
        )
        bot_token.with_identity(f"bot_{identity}")
        bot_token.with_name("Aletheia")
        bot_token.with_grants(api.VideoGrants(
            room_join=True, room=room_name,
            can_publish=True, can_subscribe=True, can_publish_data=True,
        ))

        user_token = api.AccessToken(
            api_key=server_config.livekit_api_key,
            api_secret=server_config.livekit_api_secret,
        )
        user_token.with_identity(f"user_{identity}")
        user_token.with_name("User")
        user_token.with_grants(api.VideoGrants(
            room_join=True, room=room_name,
            can_publish=True, can_subscribe=True, can_publish_data=True,
        ))

        return room_name, bot_token.to_jwt(), user_token.to_jwt()

    except Exception as e:
        logger.error(f"Token creation failed: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Token generation failed")


# ──────────────────────────────────────────────────────────────────────────────
# CONNECT ENDPOINT
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/connect-livekit")
async def connect_livekit(
    request: Request,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    try:
        data = await request.json()
        room_name = data.get("room", f"therapy-{uuid.uuid4().hex[:8]}")
        identity = user_id

        room, bot_token, user_token = await create_livekit_token(room_name, identity)

        # ── Try to grab a pre-warmed bot ──────────────────────────────────────
        try:
            pid = idle_bot_queue.get_nowait()
            await _assign_room_to_bot(pid, room_name, bot_token, user_id)
            logger.info(f"Used pre-warmed bot pid {pid} for {_safe_room_ref(room_name)}")
        except asyncio.QueueEmpty:
            # Pool exhausted — fall back to spawning fresh (and replenish async)
            logger.warning("Bot pool empty — spawning fresh process")
            pid = await _start_bot_fresh(room_name, bot_token, user_id)
            asyncio.create_task(_replenish_pool())

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
        logger.error(f"connect_livekit failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Connection failed")


async def _start_bot_fresh(room_name: str, token: str, user_id: str) -> int:
    """Fallback: spawn a bot with room args directly (old behaviour)."""
    server_dir = os.path.dirname(os.path.abspath(__file__))
    run_helpers_path = os.path.join(server_dir, "runner.py")

    cmd = [
        sys.executable, run_helpers_path,
        "-l", server_config.livekit_url,
        "-k", token,
        "-r", room_name,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = server_dir
    env["BOT_USER_ID"] = user_id

    proc = subprocess.Popen(cmd, bufsize=1, cwd=server_dir, env=env)
    bot_procs[proc.pid] = (proc, room_name, token)
    logger.info(f"Fresh bot spawned (pid: {proc.pid})")
    return proc.pid


# ──────────────────────────────────────────────────────────────────────────────
# OTHER ENDPOINTS (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "Lily Therapy Bot",
        "timestamp": datetime.now().isoformat(),
        "bots_running": len([p for p, _, _ in bot_procs.values() if p.poll() is None]),
        "bots_idle": idle_bot_queue.qsize(),  # ← useful for monitoring
    }


@app.get("/status/{pid}")
def get_status(pid: int):
    proc_tuple = bot_procs.get(pid)
    if not proc_tuple:
        raise HTTPException(status_code=404, detail=f"Bot {pid} not found")
    proc, room_name, _ = proc_tuple
    status = "running" if proc.poll() is None else "finished"
    return JSONResponse({"bot_id": pid, "status": status, "room": room_name})


@app.get("/")
async def root():
    return RedirectResponse("/connect-livekit")


@app.post("/connect")
async def rtvi_connect():
    raise HTTPException(status_code=410, detail="Daily.co deprecated. Use /connect-livekit")


@app.get("/debug/routes")
def debug_routes():
    return [
        {"path": r.path, "methods": r.methods}
        for r in app.routes if isinstance(r, APIRoute)
    ]


class SessionStartRequest(BaseModel):
    session_id: Optional[str] = None


class SessionEndRequest(BaseModel):
    session_id: str
    user_id: str
    minutes: int
    title: str
    summary: str
    session_intensity: int


@app.post("/session/start")
async def start_session(
    payload: SessionStartRequest,
    user_id: str = Depends(get_internal_or_user_id),
):
    now = datetime.utcnow()
    today = now.date().isoformat()

    existing = supabase.table("therapy_sessions") \
        .select("id, start_time") \
        .eq("user_id", user_id) \
        .eq("date", today) \
        .limit(1) \
        .execute()

    if existing.data:
        row = existing.data[0]
        return {
            "status": "resumed",
            "session_id": row["id"],
            "start_time": row["start_time"],
            "message": "Continuing today's session",
        }

    session_id = payload.session_id or str(uuid.uuid4())
    supabase.table("therapy_sessions").insert({
        "id": session_id,
        "user_id": user_id,
        "date": today,
        "start_time": now.isoformat(),
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }).execute()

    return {"status": "started", "session_id": session_id, "start_time": now.isoformat()}


@app.post("/session/end")
async def end_session(payload: SessionEndRequest):
    now = datetime.utcnow()
    today = date.today().isoformat()

    supabase.table("therapy_sessions").update({
        "end_time": now.isoformat(),
        "duration_minutes": payload.minutes,
        "title": payload.title,
        "summary": payload.summary,
        "session_intensity": payload.session_intensity,
        "updated_at": now.isoformat(),
    }).eq("id", payload.session_id).execute()

    existing = supabase.table("daily_usage") \
        .select("id, sessions_count, minutes_used") \
        .eq("user_id", payload.user_id) \
        .eq("usage_date", today) \
        .limit(1) \
        .execute()

    if existing.data:
        row = existing.data[0]
        supabase.table("daily_usage").update({
            "minutes_used": (row.get("minutes_used", 0) or 0) + payload.minutes,
            "updated_at": now.isoformat(),
        }).eq("id", row["id"]).execute()
    else:
        supabase.table("daily_usage").insert({
            "user_id": payload.user_id,
            "usage_date": today,
            "sessions_count": 1,
            "minutes_used": payload.minutes,
            "created_at": now.isoformat(),
        }).execute()

    return {"status": "completed"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)