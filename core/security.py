from fastapi import Depends, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt
import requests
import os
from dotenv import load_dotenv

load_dotenv()
security = HTTPBearer()

CLERK_JWT_ISSUER = os.getenv("CLERK_JWT_ISSUER")
CLERK_JWT_AUDIENCE = os.getenv("CLERK_JWT_AUDIENCE")

if not CLERK_JWT_ISSUER or not CLERK_JWT_AUDIENCE:
    raise RuntimeError("CLERK_JWT_ISSUER or CLERK_JWT_AUDIENCE not set")

JWKS_URL = f"{CLERK_JWT_ISSUER}/.well-known/jwks.json"
_jwks = requests.get(JWKS_URL).json()


async def verify_clerk_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verifies Clerk JWT token and returns user payload with 'sub' field"""
    token = credentials.credentials
    
    try:
        payload = jwt.decode(
            token,
            _jwks,
            algorithms=["RS256"],
            audience=CLERK_JWT_AUDIENCE,
            issuer=CLERK_JWT_ISSUER,
        )
        return payload  # Contains "sub" (user_id)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


def get_current_user_id(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")

    token = authorization.replace("Bearer ", "")

    try:
        payload = jwt.decode(
            token,
            _jwks,
            algorithms=["RS256"],
            audience=CLERK_JWT_AUDIENCE,
            issuer=CLERK_JWT_ISSUER,
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    return payload["sub"]


INTERNAL_SECRET = os.getenv("INTERNAL_API_SECRET")

def get_internal_or_user_id(
    authorization: str = Header(default=None),
    x_internal_secret: str = Header(default=None),
    x_user_id: str = Header(default=None),
) -> str:
    # Bot path — internal secret + user_id header
    if x_internal_secret:
        if not INTERNAL_SECRET or x_internal_secret != INTERNAL_SECRET:
            raise HTTPException(status_code=401, detail="Invalid internal secret")
        if not x_user_id:
            raise HTTPException(status_code=401, detail="x-user-id required")
        return x_user_id

    # Normal user path — Clerk JWT
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")

    token = authorization.replace("Bearer ", "")
    try:
        payload = jwt.decode(
            token, _jwks,
            algorithms=["RS256"],
            audience=CLERK_JWT_AUDIENCE,
            issuer=CLERK_JWT_ISSUER,
        )
        return payload["sub"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")