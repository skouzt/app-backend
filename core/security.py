import os
import threading
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

# Cached, refreshable, and fetched lazily. Fetching once at import meant a Clerk key
# rotation broke every request until restart — and looked identical to a forged
# token. Fetching at import at all meant the process could not start while Clerk was
# unreachable, so a brief outage on their side became downtime on ours, and CI could
# not boot the image without network.
_JWKS_TTL = 3600
_JWKS_TIMEOUT = 10

# DB work runs in the threadpool, so several requests can reach a cold cache at
# once. Without this they would each fetch the same key set.
_jwks_lock = threading.Lock()
_jwks_cache: dict = {}
_jwks_fetched_at: float = 0.0


class JWKSUnavailable(RuntimeError):
    """Clerk's key set could not be fetched and nothing is cached to fall back on.

    Distinct from a bad token: we cannot say whether the caller is authentic, which
    is a 503, not a 401.
    """


def _fresh() -> bool:
    return bool(_jwks_cache) and (time.time() - _jwks_fetched_at) <= _JWKS_TTL


def _get_jwks(force: bool = False) -> dict:
    global _jwks_cache, _jwks_fetched_at

    if _fresh() and not force:
        return _jwks_cache

    with _jwks_lock:
        # Re-check under the lock: whoever was waiting no longer needs to fetch.
        if _fresh() and not force:
            return _jwks_cache

        try:
            resp = requests.get(JWKS_URL, timeout=_JWKS_TIMEOUT)
            resp.raise_for_status()
            keys = resp.json()
        except Exception as e:
            if _jwks_cache:
                # Keys rotate rarely, so a stale set almost certainly still
                # verifies. Signing everyone out during a Clerk blip would be the
                # worse failure.
                logger.warning(
                    f"JWKS refresh failed ({type(e).__name__}), serving cached keys"
                )
                return _jwks_cache
            raise JWKSUnavailable(f"Could not fetch JWKS from {JWKS_URL}") from e

        _jwks_cache = keys
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
    except JWKSUnavailable:
        # Never retried and never downgraded to "invalid token" — the caller may be
        # perfectly authentic; we just cannot check.
        raise
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


# No import-time fetch. The first request that needs a key pays for it, everyone
# after that hits the cache, and the process starts whether or not Clerk is up.


async def verify_clerk_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verifies Clerk JWT token and returns user payload with 'sub' field"""
    token = credentials.credentials
    
    try:
        payload = _decode(token)
        return payload  # Contains "sub" (user_id)
    except JWKSUnavailable:
        raise HTTPException(status_code=503, detail="Cannot verify sign-in right now")
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
    except JWKSUnavailable:
        # Telling a signed-in user their token is invalid during a Clerk outage
        # sends them to re-authenticate, which cannot work either.
        raise HTTPException(status_code=503, detail="Cannot verify sign-in right now")
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