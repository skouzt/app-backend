"""The admin dashboard, behind a login.

Everything here renders names, emails and Safety_Check answers, so every
response is no-store and noindex, and every route checks the session first.
The page is rendered live from the database on each request — there is no
generated file on the server to leak.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from loguru import logger

from core.admin_auth import (
    COOKIE_NAME,
    SESSION_HOURS,
    check_credentials,
    clear_attempts,
    config,
    issue_session,
    record_failure,
    throttled,
    valid_session,
)
from services.dashboard_data import render

router = APIRouter()

# Private, never cached, never indexed, never framed.
SECURE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, private",
    "Pragma": "no-cache",
    "X-Robots-Tag": "noindex, nofollow, noarchive",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}

LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>Lily Admin</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#fff;--fg:#0a0a0a;--muted:#737373;--border:#e5e5e5;--primary:#171717;--danger:#dc2626}
@media (prefers-color-scheme:dark){:root{--bg:#0a0a0a;--fg:#fafafa;--muted:#a1a1a1;--border:#2a2a2a;--primary:#fafafa;--danger:#f87171}}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--fg);
 font-family:Inter,ui-sans-serif,system-ui,sans-serif;font-size:14px;padding:24px}
.card{width:100%;max-width:360px;border:1px solid var(--border);border-radius:14px;padding:24px;
 box-shadow:0 1px 2px rgb(0 0 0/.05)}
.brand{display:flex;align-items:center;gap:8px;font-weight:600;margin-bottom:18px}
.brand i{width:24px;height:24px;border-radius:6px;background:var(--primary);
 color:var(--bg);display:grid;place-items:center;font-style:normal;font-size:11px}
h1{margin:0 0 4px;font-size:16px;font-weight:600}
p.sub{margin:0 0 18px;color:var(--muted);font-size:13px}
label{display:block;font-size:13px;font-weight:500;margin:0 0 6px}
input{width:100%;height:36px;padding:0 10px;margin-bottom:14px;border:1px solid var(--border);
 border-radius:8px;background:var(--bg);color:var(--fg);font:inherit;font-size:13px}
input:focus{outline:none;border-color:var(--muted)}
button{width:100%;height:36px;border:none;border-radius:8px;background:var(--primary);
 color:var(--bg);font:inherit;font-size:13px;font-weight:500;cursor:pointer}
button:hover{opacity:.9}
.err{background:color-mix(in srgb,var(--danger) 10%,transparent);border:1px solid
 color-mix(in srgb,var(--danger) 30%,transparent);color:var(--danger);
 border-radius:8px;padding:8px 10px;font-size:12.5px;margin-bottom:14px}
</style></head><body>
<form class="card" method="post" action="/admin/login" autocomplete="on">
  <div class="brand"><i>L</i>Lily</div>
  <h1>Admin sign in</h1>
  <p class="sub">This console shows confidential user data.</p>
  __ERROR__
  <label for="email">Email</label>
  <input id="email" name="email" type="email" required autocomplete="username" autofocus>
  <label for="password">Password</label>
  <input id="password" name="password" type="password" required autocomplete="current-password">
  <button type="submit">Sign in</button>
</form></body></html>
"""


def _login_html(error: str | None = None, status: int = 200) -> HTMLResponse:
    block = f'<div class="err">{error}</div>' if error else ""
    return HTMLResponse(
        LOGIN_PAGE.replace("__ERROR__", block),
        status_code=status,
        headers=SECURE_HEADERS,
    )


def _is_https(request: Request) -> bool:
    """Railway terminates TLS at the edge, so request.url.scheme is http even
    when the browser is on https. Without consulting the forwarded header the
    session cookie ships without Secure and can be sent in clear text."""
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    return proto.split(",")[0].strip().lower() == "https"


def _client_ip(request: Request) -> str:
    # Railway terminates TLS in front of the app, so the real address arrives in
    # the forwarding header. Falls back to the socket for local runs.
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.get("/admin/login", response_class=HTMLResponse)
async def login_form(request: Request) -> Response:
    if valid_session(request.cookies.get(COOKIE_NAME)):
        return RedirectResponse("/admin", status_code=303, headers=SECURE_HEADERS)
    return _login_html()


@router.post("/admin/login")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
) -> Response:
    ip = _client_ip(request)

    if config() is None:
        # Fail closed and say so plainly — this is a deployment fault, not a
        # wrong password, and pretending otherwise wastes the operator's time.
        return _login_html("Admin login is not configured on this server.", 503)

    if throttled(ip):
        logger.warning(f"admin login throttled for {ip}")
        return _login_html("Too many attempts. Try again in 15 minutes.", 429)

    if not check_credentials(email, password):
        record_failure(ip)
        logger.warning(f"failed admin login for {email!r} from {ip}")
        # Deliberately does not say which half was wrong.
        return _login_html("Incorrect email or password.", 401)

    clear_attempts(ip)
    logger.info(f"admin login from {ip}")

    response = RedirectResponse("/admin", status_code=303, headers=SECURE_HEADERS)
    response.set_cookie(
        COOKIE_NAME,
        issue_session(),
        max_age=SESSION_HOURS * 3600,
        httponly=True,
        samesite="lax",
        secure=_is_https(request),
        path="/admin",
    )
    return response


@router.post("/admin/logout")
async def logout() -> Response:
    response = RedirectResponse("/admin/login", status_code=303, headers=SECURE_HEADERS)
    response.delete_cookie(COOKIE_NAME, path="/admin")
    return response


@router.get("/admin", response_class=HTMLResponse)
async def dashboard(request: Request) -> Response:
    if not valid_session(request.cookies.get(COOKIE_NAME)):
        return RedirectResponse("/admin/login", status_code=303, headers=SECURE_HEADERS)

    # supabase-py is synchronous; this is one operator hitting one page, so the
    # handful of blocking reads is not worth a threadpool hop.
    try:
        html = render()
    except Exception as e:
        logger.error(f"admin dashboard render failed: {type(e).__name__}: {e}")
        return HTMLResponse(
            "<p>Could not load the dashboard. Check the server logs.</p>",
            status_code=500,
            headers=SECURE_HEADERS,
        )
    return HTMLResponse(html, headers=SECURE_HEADERS)
