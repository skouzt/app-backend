"""Per-user rate limiting.

Every limit has two layers, and both matter for different reasons:

  · The burst window keeps one client from saturating the shared threadpool.
    Throughput per worker is a fixed pool of DB threads, so a single scripted
    caller queues everybody else behind it.

  · The daily cap bounds what one account can *cost*. The burst limit does not
    do this — 20/min sustained is still 28,800 requests a day. Since `trialing`
    grants full access, the cost ceiling has to hold for an account that has
    paid nothing yet.

State is in-process. With WEB_CONCURRENCY>1 each worker keeps its own counters,
so the effective limit is roughly N x the configured value. That is a weaker
guarantee, not a broken one — the daily cap still holds within a factor of N,
which is the difference between "a bad day" and "an unbounded bill". When you
scale out, back `_hits` and `_daily` with Redis; the call sites do not change.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from fastapi import Depends, HTTPException

from core.security import get_current_user_id


@dataclass(frozen=True)
class Limit:
    burst: int      # requests allowed per window
    window: int     # window length in seconds
    daily: int      # requests allowed per UTC day


# A human in an intense conversation sends a message every 10-20s. 20/min is far
# above that, so the burst limit only ever fires on automation. The daily cap is
# the one doing the cost work: 300 messages/day is ~1 every 5 minutes, sustained
# for 24 hours, and it cuts the worst case by three orders of magnitude.
CHAT = Limit(burst=20, window=60, daily=300)

# Dictation is the fallback path — it only runs when on-device recognition
# produced nothing, and it bills per minute of audio. If a real user is hitting
# this 30 times a day, the on-device path is broken and that is the bug to fix.
TRANSCRIBE = Limit(burst=6, window=60, daily=30)

# Ending a conversation queues a summarisation call. Starting 40 distinct
# conversations in a day is already well past what the product is for.
CLOSE = Limit(burst=10, window=60, daily=40)

# Cached for 6h per user, so this is mostly a guard against cache-miss hammering.
INSIGHTS = Limit(burst=10, window=60, daily=60)

# Reads are cheap, but not free — this is threadpool protection, not cost control.
READ = Limit(burst=60, window=60, daily=5000)


_hits: Dict[Tuple[str, str], List[float]] = {}
_daily: Dict[Tuple[str, str, str], int] = {}

# Left unbounded, these dicts grow with every user who ever calls the API. The
# sweep is amortised rather than scheduled so there is no background task to own.
_SWEEP_EVERY = 512
_calls = 0


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _sweep(now: float) -> None:
    today = _today()
    for key, stamps in list(_hits.items()):
        if not stamps or now - stamps[-1] > 3600:
            _hits.pop(key, None)
    for key in list(_daily):
        if key[2] != today:
            _daily.pop(key, None)


def enforce(user_id: str, bucket: str, limit: Limit) -> None:
    """Raise 429 if `user_id` is over either layer of `limit` for `bucket`.

    Synchronous and free of awaits on purpose: the event loop cannot interleave
    another request midway through, so the read-modify-write needs no lock.
    """
    global _calls
    now = time.monotonic()

    _calls += 1
    if _calls % _SWEEP_EVERY == 0:
        _sweep(now)

    day_key = (user_id, bucket, _today())
    used_today = _daily.get(day_key, 0)
    if used_today >= limit.daily:
        raise HTTPException(
            status_code=429,
            detail="Daily limit reached. This resets at midnight UTC.",
            headers={"Retry-After": "3600"},
        )

    window_key = (user_id, bucket)
    stamps = [t for t in _hits.get(window_key, ()) if now - t < limit.window]
    if len(stamps) >= limit.burst:
        retry = max(1, int(limit.window - (now - stamps[0])))
        _hits[window_key] = stamps
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Slow down a moment.",
            headers={"Retry-After": str(retry)},
        )

    stamps.append(now)
    _hits[window_key] = stamps
    _daily[day_key] = used_today + 1


def rate_limited(bucket: str, limit: Limit):
    """Dependency that authenticates *and* rate-limits, returning the user_id.

    Used in place of `Depends(get_current_user_id)` so a route cannot acquire an
    identity without also passing through its limit.
    """

    def dependency(user_id: str = Depends(get_current_user_id)) -> str:
        enforce(user_id, bucket, limit)
        return user_id

    return dependency
