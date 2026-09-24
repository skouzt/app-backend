#!/usr/bin/env python3
"""Write the admin dashboard to a single self-contained HTML file.

    python scripts/gen_dashboard.py
    python scripts/gen_dashboard.py --out ~/lily-dashboard.html

Offline snapshot of the same page /admin serves live. The output carries names,
emails and Safety_Check answers, which are self-harm disclosures — it is
gitignored on purpose. Keep it local; do not commit or upload it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.dashboard_data import collect, render  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the admin dashboard to disk")
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "lily-dashboard.html",
        help="Where to write the HTML (default: beside this script)",
    )
    args = parser.parse_args()

    data = collect()
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data))

    print(
        f"wrote {out} — {len(data['users'])} users, "
        f"{len(data['sessions'])} sessions, {len(data['payments'])} subscriptions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
