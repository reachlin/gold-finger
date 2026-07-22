"""
place_order.py — Place a Schwab options order for a pending semi-auto trade.

Usage:
  python schwab/place_order.py <trade_id> [--price <limit>]
  python schwab/place_order.py list

Examples:
  python schwab/place_order.py T0001
  python schwab/place_order.py T0001 --price 2.35
  python schwab/place_order.py list

If --price is omitted the stored limit (from original Slack notification) is used.
The order stays in pending_orders.json after placement — the scan hook detects the
fill and logs it to the ledger automatically.
"""

import json
import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(os.path.expanduser("~/.claude/.env"), override=False)

DATA_DIR           = os.path.join(os.path.dirname(__file__), "..", "data")
PENDING_PATH       = os.path.join(DATA_DIR, "pending_orders.json")
TOKEN_PATH         = os.path.join(os.path.dirname(__file__), "schwab_token.json")
SLACK_MENTION      = "<@U02DQJ9KKFZ>"


def _load_pending() -> list:
    if not os.path.exists(PENDING_PATH):
        return []
    with open(PENDING_PATH) as f:
        return json.load(f)


def _save_pending(orders: list):
    with open(PENDING_PATH, "w") as f:
        json.dump(orders, f, indent=2)


def _send_slack(msg: str):
    import requests
    webhook = os.getenv("SLACK_WEB_HOOK")
    if not webhook:
        print(f"  [Slack] no webhook — message:\n  {msg}")
        return
    try:
        requests.post(webhook, json={"text": msg}, timeout=5).raise_for_status()
    except Exception as e:
        print(f"  [Slack] failed: {e}")


def _schwab_client():
    import sys
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _removed = [p for p in sys.path if os.path.abspath(p) == _project_root]
    for p in _removed:
        sys.path.remove(p)
    _local_schwab = sys.modules.pop("schwab", None)
    import schwab as _schwab_lib
    sys.modules["schwab"] = _local_schwab
    for p in _removed:
        sys.path.insert(0, p)

    token_path    = os.path.join(os.path.dirname(__file__), "schwab_token.json")
    client_id     = os.environ["SCHWAB_CLIENT_ID"]
    client_secret = os.environ["SCHWAB_CLIENT_SECRET"]
    return _schwab_lib.auth.client_from_token_file(token_path, client_id, client_secret)


def _get_account_hash(client) -> str | None:
    resp = client.get_account_numbers()
    resp.raise_for_status()
    accounts = resp.json()
    if not accounts:
        return None
    return accounts[0]["hashValue"]


def _get_schwab_option_positions(client, account_hash: str) -> list[dict]:
    """Fetch open option positions from Schwab. Returns list of position dicts."""
    try:
        resp = client.get_account(account_hash, fields=["positions"])
        resp.raise_for_status()
        data = resp.json()
        positions = data.get("securitiesAccount", {}).get("positions", [])
        return [
            p for p in positions
            if p.get("instrument", {}).get("assetType") == "OPTION"
            and p.get("longQuantity", 0) + p.get("shortQuantity", 0) > 0
        ]
    except Exception as e:
        print(f"  [Warning] Could not fetch Schwab positions: {e}")
        return []


def _check_conflict_position(schwab_positions: list[dict], entry: dict) -> str | None:
    """
    Check if Schwab already has an open position that conflicts with this order.
    Returns a warning string if conflict found, else None.
    """
    sig    = entry["signal"]
    symbol = entry["symbol"].upper()
    strike = float(entry.get("strike", 0))

    opt_type = None
    if sig == "SELL_PUT":
        opt_type = "P"
    elif sig in ("SELL_CALL", "BUY_CALL"):
        opt_type = "C"

    for pos in schwab_positions:
        inst = pos.get("instrument", {})
        desc = inst.get("description", "")
        sym  = inst.get("underlyingSymbol", "").upper()
        occ  = inst.get("symbol", "")
        short_qty = float(pos.get("shortQuantity", 0))
        long_qty  = float(pos.get("longQuantity",  0))
        qty = short_qty + long_qty
        if sym != symbol or qty == 0:
            continue
        # Check option type matches
        if opt_type and len(occ) > 15:
            occ_type = occ[15] if len(occ) > 15 else ""
            if occ_type != opt_type:
                continue
        return (f"Schwab already has open {symbol} option position: "
                f"{desc or occ}  qty={qty:.0f}  "
                f"(short={short_qty:.0f}, long={long_qty:.0f})")
    return None


def _check_amateur_hour() -> tuple[bool, str]:
    """
    Block order placement during amateur hour (9:30–10:30 ET) and pre-market.
    Pre-market DAY orders execute at open = same problem.
    Returns (blocked, reason).
    """
    from zoneinfo import ZoneInfo
    from datetime import time as dtime
    now_et = datetime.now(ZoneInfo("America/New_York")).time()
    market_open  = dtime(9, 30)
    safe_open    = dtime(10, 30)
    market_close = dtime(16, 0)
    if now_et < market_open:
        return True, (f"Pre-market ({now_et.strftime('%H:%M')} ET) — DAY orders "
                      f"execute at open (9:30 ET = amateur hour). Wait until 10:30 ET.")
    if market_open <= now_et < safe_open:
        return True, (f"Amateur hour ({now_et.strftime('%H:%M')} ET) — first 60 min "
                      f"are volatile. Wait until 10:30 ET.")
    if now_et >= market_close:
        return True, (f"Market closed ({now_et.strftime('%H:%M')} ET).")
    return False, ""


def cmd_list():
    orders = _load_pending()
    if not orders:
        print("No pending orders.")
        return
    print(f"{'ID':<8} {'Symbol':<12} {'Signal':<10} {'Strike':>8}  "
          f"{'Expiry':<12} {'Limit':>8}  {'Notified At'}")
    print("─" * 85)
    for o in orders:
        print(f"{o.get('trade_id','?'):<8} {o['symbol']:<12} {o['signal']:<10} "
              f"{o.get('strike','?'):>8}  {o.get('expiry','?'):<12} "
              f"{o.get('limit','?'):>8}  {o.get('notified_at','?')}")


def cmd_place(trade_id: str, price_override: float | None, skip_confirm: bool = False):
    orders = _load_pending()
    match = [o for o in orders if o.get("trade_id", "").upper() == trade_id.upper()]
    if not match:
        print(f"Error: trade {trade_id} not found in pending orders.")
        print("Run `python schwab/place_order.py list` to see pending trades.")
        sys.exit(1)
    entry = match[0]

    sig     = entry["signal"]
    symbol  = entry["symbol"]
    occ_sym = entry["occ_sym"]
    limit   = price_override if price_override is not None else entry["limit"]

    blocked, reason = _check_amateur_hour()
    if blocked:
        print(f"\n  ⛔ Cannot place order: {reason}")
        sys.exit(1)

    print(f"\n  Trade:    {trade_id}  {sig}  {symbol}")
    print(f"  OCC:      {occ_sym}")
    print(f"  Limit:    ${limit:.2f}  (original: ${entry['limit']:.2f})")
    print(f"  Expiry:   {entry.get('expiry','?')}  ({entry.get('dte','?')} DTE)")
    print(f"  Strike:   ${entry.get('strike','?')}")
    if entry.get("bid"):
        print(f"  Bid/Ask:  ${entry['bid']:.2f} / ${entry.get('ask','?')}")
    print()

    if not skip_confirm:
        confirm = input("  Place this order? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("  Cancelled — no order placed.")
            return

    try:
        client = _schwab_client()
    except Exception as e:
        print(f"  Error: could not connect to Schwab: {e}")
        sys.exit(1)

    account_hash = _get_account_hash(client)
    if not account_hash:
        print("  Error: no accounts found on this token.")
        sys.exit(1)

    # Guard: check Schwab for existing conflicting positions before placing
    schwab_positions = _get_schwab_option_positions(client, account_hash)
    conflict = _check_conflict_position(schwab_positions, entry)
    if conflict:
        print(f"\n  ⛔ BLOCKED — {conflict}")
        print("  Close or reconcile the existing position before placing a new one.")
        sys.exit(1)

    try:
        from schwab.orders.options import (
            option_sell_to_open_limit,
            option_buy_to_open_limit,
        )
        from schwab.orders.common import Duration, Session

        if sig in ("SELL_PUT", "SELL_CALL"):
            order = (
                option_sell_to_open_limit(occ_sym, 1, f"{limit:.2f}")
                .set_duration(Duration.DAY)
                .set_session(Session.NORMAL)
                .build()
            )
        elif sig == "BUY_CALL":
            order = (
                option_buy_to_open_limit(occ_sym, 1, f"{limit:.2f}")
                .set_duration(Duration.DAY)
                .set_session(Session.NORMAL)
                .build()
            )
        else:
            print(f"  Error: unsupported signal type '{sig}' for automated placement.")
            sys.exit(1)

        resp = client.place_order(account_hash, order)
        resp.raise_for_status()

    except Exception as e:
        print(f"  ✗ Order placement failed: {e}")
        sys.exit(1)

    print(f"\n  ✅ Order placed: {trade_id}  {sig}  {occ_sym}  limit ${limit:.2f}")
    print(f"  The scan hook will detect the fill and log it to the ledger.")

    _send_slack(
        f"{SLACK_MENTION} ✅ *Order placed via console: {trade_id}*\n"
        f"{sig}  `{occ_sym}`  limit ${limit:.2f}"
    )


def cmd_cancel(trade_id: str):
    """Cancel a live Schwab order by trade ID and remove it from pending."""
    from datetime import timedelta

    orders = _load_pending()
    match = [o for o in orders if o.get("trade_id", "").upper() == trade_id.upper()]
    if not match:
        print(f"  Error: trade {trade_id} not found in pending orders.")
        print("  Run `python schwab/place_order.py list` to see pending trades.")
        sys.exit(1)
    entry = match[0]
    occ_sym = entry["occ_sym"]
    occ_norm = occ_sym.replace(" ", "")

    print(f"\n  Cancelling: {trade_id}  {entry['signal']}  {entry['symbol']}")
    print(f"  OCC: {occ_sym}")

    try:
        client = _schwab_client()
    except Exception as e:
        print(f"  Error: could not connect to Schwab: {e}")
        sys.exit(1)

    account_hash = _get_account_hash(client)
    if not account_hash:
        print("  Error: no accounts found on this token.")
        sys.exit(1)

    # Find the matching open order on Schwab
    now = datetime.now()
    try:
        resp = client.get_orders_for_account(
            account_hash,
            from_entered_datetime=now - timedelta(hours=8),
            to_entered_datetime=now + timedelta(minutes=5),
        )
        resp.raise_for_status()
        schwab_orders = resp.json()
    except Exception as e:
        print(f"  Error fetching Schwab orders: {e}")
        sys.exit(1)

    order_id = None
    for o in schwab_orders:
        if o.get("status") not in ("WORKING", "QUEUED", "ACCEPTED", "PENDING_ACTIVATION"):
            continue
        for leg in o.get("orderLegCollection", []):
            sym = leg.get("instrument", {}).get("symbol", "").replace(" ", "")
            if occ_norm in sym or sym in occ_norm:
                order_id = o["orderId"]
                break
        if order_id:
            break

    if order_id is None:
        print(f"  ⚠ No open Schwab order found for {occ_sym}.")
        print(f"    It may already be filled or cancelled. Removing from pending only.")
    else:
        try:
            resp = client.cancel_order(order_id, account_hash)
            resp.raise_for_status()
            print(f"  ✅ Schwab order {order_id} cancelled.")
        except Exception as e:
            print(f"  Error cancelling Schwab order: {e}")
            sys.exit(1)

    # Remove from pending_orders.json
    remaining = [o for o in orders if o.get("trade_id", "").upper() != trade_id.upper()]
    _save_pending(remaining)
    print(f"  Removed {trade_id} from pending orders.")

    _send_slack(
        f"{SLACK_MENTION} 🚫 *Order cancelled: {trade_id}*\n"
        f"{entry['signal']}  `{occ_sym}`  — removed from watch list."
    )


def main():
    parser = argparse.ArgumentParser(description="Manage pending semi-auto trades.")
    parser.add_argument("command", nargs="?", help="trade_id, 'list', or 'cancel <id>'")
    parser.add_argument("trade_id", nargs="?", help="Trade ID for cancel subcommand")
    parser.add_argument("--price", type=float, help="Override limit price")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    cmd = (args.command or "list").lower()

    if cmd == "list":
        cmd_list()
    elif cmd == "cancel":
        if not args.trade_id:
            print("Usage: place_order.py cancel <trade_id>")
            sys.exit(1)
        cmd_cancel(args.trade_id.upper())
    else:
        # treat first arg as trade_id for placement
        cmd_place(args.command.upper(), args.price, skip_confirm=args.yes)


if __name__ == "__main__":
    main()
