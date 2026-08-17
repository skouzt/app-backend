"""How the user wants Lily to sound.

Read on every prompt build (behind the prompt cache) and written from the
Personalization screen. Values are validated here rather than trusted from the
client, because they are interpolated into the system prompt — an unconstrained
string in that position is prompt injection with extra steps.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger

from db.supabase import supabase

TABLE = "user_personalization"

TONES = ("Gentle", "Friendly", "Reflective", "Direct")
TRAIT_LEVELS = ("Less", "Default", "More")
TRAIT_KEYS = ("warm", "encouraging", "questions", "brevity")

NOTE_MAX_CHARS = 2000

DEFAULTS: Dict[str, Any] = {
    "tone": "Gentle",
    "traits": {k: "Default" for k in TRAIT_KEYS},
    "check_ins": True,
    "note": "",
}


def normalise(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Coerce anything — a client payload or an old DB row — into a valid shape.

    Unknown tones and trait levels fall back to the default rather than raising:
    a preference is not worth a failed request, and silently ignoring a bad value
    is safer than letting it reach the prompt.
    """
    raw = raw or {}

    tone = str(raw.get("tone") or "").strip()
    if tone not in TONES:
        tone = DEFAULTS["tone"]

    incoming = raw.get("traits") or {}
    traits = {}
    for key in TRAIT_KEYS:
        level = str(incoming.get(key) or "").strip() if isinstance(incoming, dict) else ""
        traits[key] = level if level in TRAIT_LEVELS else "Default"

    note = raw.get("note") or ""
    if not isinstance(note, str):
        note = ""
    note = note.strip()[:NOTE_MAX_CHARS]

    check_ins = raw.get("check_ins")
    if check_ins is None:
        check_ins = DEFAULTS["check_ins"]

    return {"tone": tone, "traits": traits, "check_ins": bool(check_ins), "note": note}


def fetch_personalization(user_id: str) -> Dict[str, Any]:
    """Never raises — a preferences lookup must not be able to break the chat."""
    try:
        res = (
            supabase.table(TABLE)
            .select("tone, traits, check_ins, note")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        return normalise(rows[0]) if rows else dict(DEFAULTS)
    except Exception as e:
        logger.warning(f"personalization fetch failed, using defaults: {type(e).__name__}: {e}")
        return dict(DEFAULTS)


def save_personalization(user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    clean = normalise(payload)
    row = {"user_id": user_id, **clean, "updated_at": "now()"}

    # on_conflict so the first save inserts and later ones update, without a
    # read-then-write race between two devices saving at the same time.
    supabase.table(TABLE).upsert(row, on_conflict="user_id").execute()
    return clean
