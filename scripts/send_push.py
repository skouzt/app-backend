"""Send a push notification from the command line.

    python scripts/send_push.py "Lily" "I'm here whenever you want to talk."
    python scripts/send_push.py "Lily" "Your weekly look-back is ready." --user user_387MU...

Prints how many were sent, failed, and how many dead tokens were dropped.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.push_service import (  # noqa: E402
    all_active_tokens,
    active_tokens,
    send_to_all,
    send_to_user,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a push notification")
    parser.add_argument("title")
    parser.add_argument("body")
    parser.add_argument("--user", help="Send to one user id instead of everyone")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt",
    )
    args = parser.parse_args()

    if args.user:
        audience = len(active_tokens(args.user))
        target = f"user {args.user}"
    else:
        audience = len(all_active_tokens())
        target = "EVERYONE"

    if audience == 0:
        print("No reachable devices. Nothing sent.")
        return 0

    print(f"\n  To     : {target}  ({audience} device(s))")
    print(f"  Title  : {args.title}")
    print(f"  Body   : {args.body}\n")

    # A push cannot be recalled once it is away, and a broadcast lands on every
    # lock screen at once. One confirmation is cheap insurance against a typo.
    if not args.yes:
        if input("Send? [y/N] ").strip().lower() != "y":
            print("Cancelled.")
            return 1

    if args.user:
        result = asyncio.run(send_to_user(args.user, title=args.title, body=args.body))
    else:
        result = asyncio.run(send_to_all(title=args.title, body=args.body))

    print(
        f"\n  sent {result['sent']}   failed {result['failed']}   "
        f"dead tokens removed {result['removed']}\n"
    )
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
