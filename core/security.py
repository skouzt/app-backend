import os
import time

import requests
from dotenv import load_dotenv
from fastapi import Depends, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt
from loguru import logger

load_dotenv()
security = HTTPBearer()

CLERK_JWT_ISSUER = os.getenv("CLERK_JWT_ISSUER")
CLERK_JWT_AUDIENCE = os.getenv("CLERK_JWT_AUDIENCE")

if not CLERK_JWT_ISSUER or not CLERK_JWT_AUDIENCE:
    raise RuntimeError("CLERK_JWT_ISSUER or CLERK_JWT_AUDIENCE not set")

JWKS_URL = f"{CLERK_JWT_ISSUER}/.well-known/jwks.json"

# Cached but refreshable. Fetching once at import meant a Clerk key rotation broke
# every request until restart — and looked identical to a forged token.
_JWKS_TTL = 3600
_jwks_cache: dict = {}
_jwks_fetched_at: float = 0.0


def _get_jwks(force: bool = False) -> dict:
    global _jwks_cache, _jwks_fetched_at
    if force or not _jwks_cache or (time.time() - _jwks_fetched_at) > _JWKS_TTL:
        _jwks_cache = requests.get(JWKS_URL, timeout=10).json()
        _jwks_fetched_at = time.time()
        logger.info(f"Fetched Clerk JWKS ({len(_jwks_cache.get('keys', []))} keys)")
    return _jwks_cache


def _decode(token: str) -> dict:
    """Decode, logging the real reason on failure and retrying once on key rotation."""
    try:
        return jwt.decode(
            token, _get_jwks(), algorithms=["RS256"],
            audience=CLERK_JWT_AUDIENCE, issuer=CLERK_JWT_ISSUER,
        )
    except Exception as first:
        reason = f"{type(first).__name__}: {first}"
        try:
            payload = jwt.decode(
                token, _get_jwks(force=True), algorithms=["RS256"],
                audience=CLERK_JWT_AUDIENCE, issuer=CLERK_JWT_ISSUER,
            )
            logger.warning("JWT verified only after JWKS refresh — Clerk keys rotated")
            return payload
        except Exception as second:
            logger.warning(
                f"JWT rejected: {reason} | after refresh: {type(second).__name__}: {second} "
                f"| expected aud={CLERK_JWT_AUDIENCE!r}"
            )
            raise second


_jwks = _get_jwks()


async def verify_clerk_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verifies Clerk JWT token and returns user payload with 'sub' field"""
    token = credentials.credentials
    
    try:
        payload = _decode(token)
        return payload  # Contains "sub" (user_id)
    except Exception:
        # _decode already logged the real reason. Returning it to the caller told
        # an attacker exactly which part of a forged token to fix next.
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user_id(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")

    token = authorization.replace("Bearer ", "")

    try:
        payload = _decode(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    return payload["sub"]


# Removed: get_internal_or_user_id().
#
# It returned an X-User-Id header verbatim to any caller presenting a shared
# INTERNAL_API_SECRET — full impersonation of any account, no JWT involved. It
# existed for the LiveKit bot process, which no longer exists, and had zero call
# sites by the time it was deleted. Identity now comes only from a verified
# Clerk token. Do not reintroduce a header-trusting auth path; if a future
# service needs to act for a user, give it its own token.