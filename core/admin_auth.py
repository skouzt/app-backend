"""Admin login for the operations dashboard.

The dashboard renders names, emails and Safety_Check answers — self-harm
disclosures — so this gate is the only thing between that page and the open
internet. It fails closed: if the environment is not fully configured, no
credential is accepted at all rather than falling back to a default.

No new dependency. scrypt is in hashlib, constant-time comparison is in hmac,
and the session cookie is a short-lived JWT signed with python-jose, which is
already used for Clerk verification.

Generate a password hash with:

    python -m core.admin_auth
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from jose import jwt
from loguru import logger

load_dotenv()

ALGO = "HS256"
SESSION_HOURS = 12
COOKIE_NAME = "lily_admin"

# scrypt cost. n=2**15 keeps a single verification around a tenth of a second,
# which is irrelevant for one operator and expensive for anyone guessing.
_N, _R, _P, _DKLEN = 2 ** 15, 8, 1, 32


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(txt: str) -> bytes:
    return base64.urlsafe_b64decode(txt + "=" * (-len(txt) % 4))


def _maxmem(n: int, r: int) -> int:
    """scrypt needs 128*n*r bytes; OpenSSL refuses above its own 32MB default,
    so the budget is stated explicitly and derived from the stored parameters."""
    return 128 * n * r * 2


def hash_password(password: str) -> str:
    """Return a self-describing scrypt hash: scrypt$n$r$p$salt$key."""
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(
        password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=_DKLEN,
        maxmem=_maxmem(_N, _R),
    )
    return f"scrypt${_N}${_R}${_P}${_b64e(salt)}${_b64e(key)}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check against a stored scrypt hash."""
    try:
        scheme, n, r, p, salt, key = stored.split("$")
        if scheme != "scrypt":
            return False
        want = _b64d(key)
        n_i, r_i, p_i = int(n), int(r), int(p)
        got = hashlib.scrypt(
            password.encode(), salt=_b64d(salt),
            n=n_i, r=r_i, p=p_i, dklen=len(want),
            maxmem=_maxmem(n_i, r_i),
        )
    except Exception:
        # A malformed hash is a configuration error, not a valid login.
        logger.error("ADMIN_PASSWORD_HASH is malformed — refusing all logins")
        return False
    return hmac.compare_digest(got, want)


@dataclass(frozen=True)
class AdminConfig:
    email: str
    password_hash: str
    secret: str


def config() -> AdminConfig | None:
    """The configured admin, or None when the environment is incomplete."""
    email = os.getenv("ADMIN_EMAIL", "").strip().lower()
    pw = os.getenv("ADMIN_PASSWORD_HASH", "").strip()
    secret = os.getenv("ADMIN_SESSION_SECRET", "").strip()
    if not (email and pw and secret):
        return None
    if len(secret) < 32:
        logger.error("ADMIN_SESSION_SECRET is too short — refusing all logins")
        return None
    return AdminConfig(email=email, password_hash=pw, secret=secret)


def check_credentials(email: str, password: str) -> bool:
    """Both halves are compared even when the email is wrong, so a bad address
    and a bad password take the same time and cannot be told apart."""
    cfg = config()
    if cfg is None:
        logger.error("Admin login attempted but ADMIN_* env is not configured")
        return False
    email_ok = hmac.compare_digest(email.strip().lower(), cfg.email)
    password_ok = verify_password(password, cfg.password_hash)
    return email_ok and password_ok


def issue_session() -> str:
    """Always subjects the token to the configured address, never the raw form
    input — otherwise a login typed with different capitalisation mints a token
    that valid_session() then refuses, and the page bounces straight back to
    the form with no error to explain it."""
    cfg = config()
    if cfg is None:
        raise RuntimeError("admin not configured")
    now = int(time.time())
    return jwt.encode(
        {"sub": cfg.email, "iat": now, "exp": now + SESSION_HOURS * 3600, "scope": "admin"},
        cfg.secret,
        algorithm=ALGO,
    )


def valid_session(token: str | None) -> bool:
    cfg = config()
    if cfg is None or not token:
        return False
    try:
        claims = jwt.decode(token, cfg.secret, algorithms=[ALGO])
    except Exception:
        return False
    return claims.get("scope") == "admin" and claims.get("sub") == cfg.email


# ── Login throttle ───────────────────────────────────────────────────────────
#
# In-process and per-IP. Enough to make online guessing useless against one
# operator account.
# ponytail: in-memory dict, fine for one process — move to Redis if this ever
# runs multi-worker and the throttle needs to be shared.
_MAX_ATTEMPTS = 5
_WINDOW = 15 * 60
_attempts: dict[str, list[float]] = {}


def throttled(ip: str) -> bool:
    now = time.time()
    hits = [t for t in _attempts.get(ip, []) if now - t < _WINDOW]
    _attempts[ip] = hits
    return len(hits) >= _MAX_ATTEMPTS


def record_failure(ip: str) -> None:
    _attempts.setdefault(ip, []).append(time.time())


def clear_attempts(ip: str) -> None:
    _attempts.pop(ip, None)


if __name__ == "__main__":
    import getpass

    pw = getpass.getpass("Admin password: ")
    if pw != getpass.getpass("Confirm: "):
        raise SystemExit("Passwords do not match.")
    print("\nADMIN_PASSWORD_HASH=" + hash_password(pw))
    print("ADMIN_SESSION_SECRET=" + secrets.token_urlsafe(48))
