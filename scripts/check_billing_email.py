"""A NULL email column must never become the string "None".

.get("email", "") returns None for a key that exists with a NULL value — the
default does not fire — and str(None) is "None", which is truthy. That string
passed the caller's `if not email` guard, went to Dodo as the customer email,
came back 400 INVALID_REQUEST_PARAMETERS, and surfaced as a 502 on
/billing/create-checkout. Nobody in that state could pay.
"""

import asyncio
import sys

sys.path.insert(0, ".")

import api.v1.billing.dodo as dodo


def _stub(rows):
    res = type("Res", (), {"data": rows})()
    chain = type("Chain", (), {
        "select": lambda self, *a: self,
        "eq": lambda self, *a: self,
        "execute": lambda self: res,
    })()
    dodo.supabase = type("SB", (), {"table": staticmethod(lambda _: chain)})()


def email_for(user, rows):
    _stub(rows)
    return asyncio.run(dodo._get_user_email(user, "user_1"))


cases = [
    ({}, [{"email": None}], None),        # the bug: NULL column
    ({}, [{}], None),                     # column absent
    ({}, [], None),                       # no row
    ({}, [{"email": ""}], None),          # empty string
    ({}, [{"email": "a@b.com"}], "a@b.com"),
    ({"email": "jwt@b.com"}, [{"email": None}], "jwt@b.com"),
]

for user, rows, expected in cases:
    got = email_for(user, rows)
    assert got == expected, f"user={user} rows={rows}: expected {expected!r}, got {got!r}"

print(f"billing email: {len(cases)} cases OK")
