"""Lily API.

Chat-first backend: the client sends text, Lily replies, and the server quietly
groups messages into sessions and summarises them so future conversations start
with context. There is no voice pipeline — see git history for the LiveKit/Pipecat
implementation this replaced.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from core.config import settings
from services.session_reaper import run_reaper

from api.v1.billing.dodo import router as billing_router
from api.v1.billing.pricing import router as pricing_router
from api.v1.chat.messages import router as chat_router
from api.v1.insights import router as insights_router
from api.v1.therapy.sessions import router as therapy_sessions_router
from api.v1.usage import router as usage_router
from api.v1.users.onboarding import router as onboarding_router
from api.v1.users.onboarding_submit import router as onboarding_submit_router
from api.v1.users.personalization import router as personalization_router
from api.v1.users.profile import router as profile_router
from api.v1.users.subscription import router as user_subscription_router

load_dotenv()


# The reaper is a singleton, not per-request work. With multiple workers every
# process would sweep the same sessions — the status='closing' guard keeps that
# correct rather than double-summarising, but it is wasted queries and wasted LLM
# calls racing each other. Set RUN_REAPER=false on the extra workers, or run it as
# its own process.
RUN_REAPER = os.environ.get("RUN_REAPER", "true").lower() not in ("false", "0", "no")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Lily API starting")

    reaper = asyncio.create_task(run_reaper()) if RUN_REAPER else None
    if not reaper:
        logger.info("Session reaper disabled in this process (RUN_REAPER=false)")

    try:
        yield
    finally:
        if reaper:
            reaper.cancel()
            try:
                await reaper
            except asyncio.CancelledError:
                pass
        logger.info("Lily API stopped")


app: FastAPI = FastAPI(title="Lily API", lifespan=lifespan)

# Auth is Bearer-token based, so credentials (cookies) are never needed. A wildcard
# origin with allow_credentials=True is rejected by browsers anyway.
#
# Unset now means "no origins" rather than "*". CORS only constrains browsers, and
# the native app is not a browser — so the wildcard protected nothing while letting
# any web page call this API on behalf of a user whose token it had obtained.
_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]

if not _origins:
    logger.info("CORS: no allowed origins — browser cross-origin requests are denied")
else:
    logger.info(f"CORS: allowing {_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_subscription_router, prefix="/api/v1")
app.include_router(onboarding_router, prefix="/api/v1/users")
app.include_router(onboarding_submit_router, prefix="/api/v1/users")
app.include_router(profile_router, prefix="/api/v1/users")
app.include_router(personalization_router, prefix="/api/v1/users")
app.include_router(therapy_sessions_router, prefix="/api/v1/therapy")
app.include_router(usage_router, prefix="/api/v1")
app.include_router(billing_router, prefix="/api/v1")
app.include_router(pricing_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(insights_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "Lily API",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Removed: an unauthenticated /debug/routes that published the whole API surface.
# Everything behind it is authenticated, so this was reconnaissance rather than
# access — but there is no reason to hand out the map. Use `fastapi routes` or
# /docs behind auth locally instead.


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))

    # supabase-py is synchronous, so DB work runs in the threadpool (see
    # chat_service.in_thread). Concurrency per worker is bounded by that pool;
    # workers multiply it across cores.
    workers = int(os.environ.get("WEB_CONCURRENCY", "1"))

    if workers > 1 and RUN_REAPER:
        logger.warning(
            "WEB_CONCURRENCY>1 with RUN_REAPER=true — every worker will sweep. "
            "Run one reaper process and set RUN_REAPER=false on the rest."
        )

    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, workers=workers)
