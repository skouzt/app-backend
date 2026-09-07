"""A Dodo delivery must be claimed, and a redelivery must not run twice.

Standard Webhooks puts the message id in the webhook-id header, not the body.
Reading payload["id"] found nothing, so event_id was always None, the claim was
skipped, and webhook_events stayed empty while handlers ran — every redelivery
free to reprocess a subscription change.
"""

import asyncio
import json
import sys

sys.path.insert(0, ".")

import api.v1.billing.dodo as dodo
from fastapi import BackgroundTasks

ROWS: dict = {}
CALLS: list = []


class _Q:
    def __init__(self, op, payload=None):
        self.op, self.payload, self.f = op, payload, []

    def eq(self, k, v):
        self.f.append(("eq", k, v)); return self

    def lt(self, k, v):
        self.f.append(("lt", k, v)); return self

    def limit(self, _):
        return self

    def _match(self, row):
        for op, k, v in self.f:
            got = row.get(k)
            if op == "eq" and got != v:
                return False
            if op == "lt" and not (got is not None and str(got) < str(v)):
                return False
        return True

    def execute(self):
        res = lambda d: type("R", (), {"data": d})()
        if self.op == "insert":
            rid = self.payload["id"]
            if rid in ROWS:
                raise Exception('duplicate key value violates unique constraint')
            ROWS[rid] = dict(self.payload)
            return res([ROWS[rid]])
        hit = [r for r in ROWS.values() if self._match(r)]
        if self.op == "select":
            return res([dict(r) for r in hit])
        if self.op == "update":
            for r in hit:
                r.update(self.payload)
            return res([dict(r) for r in hit])
        for r in hit:
            ROWS.pop(r["id"], None)
        return res([])


class _Tbl:
    def insert(self, row): return _Q("insert", row)
    def select(self, *a, **k): return _Q("select")
    def update(self, v, **k): return _Q("update", v)
    def delete(self): return _Q("delete")


dodo.supabase = type("SB", (), {"table": staticmethod(lambda _: _Tbl())})()
dodo.DodoClient.verify_webhook_signature = staticmethod(lambda **k: True)


async def _handler(data, background=None):
    CALLS.append("handled")


dodo._on_subscription_activated = _handler

BODY = json.dumps({"type": "subscription.active", "data": {}}).encode()


class _Req:
    """Dodo sends no top-level id in the body; the id is the webhook-id header."""

    def __init__(self, wid):
        self.headers = {"webhook-id": wid}

    async def body(self):
        return BODY


def post(wid):
    return asyncio.run(dodo.dodo_webhook(_Req(wid), BackgroundTasks()))


post("msg_1")
assert CALLS == ["handled"], "first delivery did not run"
assert "msg_1" in ROWS, "delivery was not claimed — webhook_events stays empty"
assert ROWS["msg_1"]["status"] == "done", "claim never marked done"

post("msg_1")
assert CALLS == ["handled"], "redelivery reprocessed the same event"

post("msg_2")
assert CALLS == ["handled", "handled"], "a distinct event was blocked"

print("webhook idempotency: 3 cases OK")
