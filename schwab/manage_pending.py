"""
manage_pending.py  — CLI to inspect and cancel pending semi-auto orders.

Usage:
  python schwab/manage_pending.py list
  python schwab/manage_pending.py cancel <index>
  python schwab/manage_pending.py cancel all

Pending orders are stored in data/pending_orders.json and watched by the
check_pending_orders scan hook in auto_overseer.py.  Use this tool when:
  - An order timed out / price moved and you won't be placing it
  - You went out and couldn't place the trade in time
  - You want to clear stale entries before market open
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATA_DIR           = os.path.join(os.path.dirname(__file__), "..", "data")
PENDING_PATH       = os.path.join(DATA_DIR, "pending_orders.json")
SLACK_MENTION      = "<@U02DQJ9KKFZ>"


def _load() -> list:
    if not os.path.exists(PENDING_PATH):
        return []
    with open(PENDING_PATH) as f:
        return json.load(f)


def _save(orders: list):
    with open(PENDING_PATH, "w") as f:
        json.dump(orders, f, indent=2)


def _send_slack(msg: str):
    import requests
    webhook = os.getenv("SLACK_WEB_HOOK")
    if not webhook:
        print(f"  [Slack] no webhook configured — message:\n  {msg}")
        return
    try:
        r = requests.post(webhook, json={"text": msg}, timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"  [Slack] failed: {e}")


def cmd_list(orders: list):
    if not orders:
        print("No pending orders.")
        return
    print(f"{'#':>3}  {'Symbol':<12} {'Signal':<10} {'Strike':>8}  "
          f"{'Expiry':<12} {'Limit':>8}  {'Notified At'}")
    print("─" * 80)
    for i, o in enumerate(orders):
        print(f"{i:>3}  {o['symbol']:<12} {o['signal']:<10} "
              f"{o.get('strike', '?'):>8}  {o.get('expiry', '?'):<12} "
              f"{o.get('limit', '?'):>8}  {o.get('notified_at', '?')}")


def cmd_cancel(orders: list, target: str) -> list:
    if target.lower() == "all":
        cancelled = orders[:]
        remaining = []
    else:
        try:
            idx = int(target)
        except ValueError:
            print(f"Error: index must be a number or 'all', got '{target}'")
            sys.exit(1)
        if idx < 0 or idx >= len(orders):
            print(f"Error: index {idx} out of range (0–{len(orders)-1})")
            sys.exit(1)
        cancelled = [orders[idx]]
        remaining = [o for i, o in enumerate(orders) if i != idx]

    for o in cancelled:
        print(f"  Cancelled: {o['symbol']} {o['signal']} ${o.get('strike','?')}  "
              f"(notified {o.get('notified_at','?')})")

    if cancelled:
        lines = "\n".join(
            f"  • {o['symbol']} {o['signal']} ${o.get('strike','?')}  "
            f"(notified {o.get('notified_at','?')})"
            for o in cancelled
        )
        msg = (f"{SLACK_MENTION} 🚫 *Pending order(s) manually cancelled:*\n"
               f"{lines}\n"
               f"Removed from watch list — no fill logging will occur.")
        _send_slack(msg)

    return remaining


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()
    orders = _load()

    if cmd == "list":
        cmd_list(orders)

    elif cmd == "cancel":
        if len(sys.argv) < 3:
            print("Usage: manage_pending.py cancel <index|all>")
            sys.exit(1)
        remaining = cmd_cancel(orders, sys.argv[2])
        _save(remaining)
        print(f"\n{len(orders) - len(remaining)} order(s) removed. "
              f"{len(remaining)} remaining.")

    else:
        print(f"Unknown command '{cmd}'. Use: list | cancel <index|all>")
        sys.exit(1)


if __name__ == "__main__":
    main()
