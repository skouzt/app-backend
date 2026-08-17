"""Supabase client — one per thread.

supabase-py is synchronous and wraps a single `httpx.Client`. DB work runs in the
threadpool (see `chat_service.in_thread`) so requests can overlap, and sharing one
client across those threads corrupts httpx's HTTP/2 connection state: under load it
raises LocalProtocolError, KeyError from the h2 state machine, and RemoteProtocolError
when a multiplexed connection is recycled mid-flight.

Serialised code never hit this because only one request was ever in flight. Giving
each thread its own client removes the shared mutable state entirely. The pool is
bounded by the threadpool size (~40), and threads are reused, so clients are created
once each rather than per request.

Call sites are unchanged: `supabase.table(...)` still works, resolved per-thread.
"""

import os
import threading
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client
from supabase.client import ClientOptions

load_dotenv()

SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]

_local = threading.local()


def _build() -> Client:
    return create_client(
        SUPABASE_URL,
        SUPABASE_SERVICE_ROLE_KEY,
        options=ClientOptions(
            postgrest_client_timeout=10,
            storage_client_timeout=10,
            schema="public",
        ),
    )


def get_client() -> Client:
    client = getattr(_local, "client", None)
    if client is None:
        client = _build()
        _local.client = client
    return client


class _ThreadLocalSupabase:
    """Transparent proxy so existing `supabase.table(...)` call sites keep working."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_client(), name)


supabase = _ThreadLocalSupabase()
