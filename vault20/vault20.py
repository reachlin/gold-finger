"""
Vault 20 — Manual Position Tracker

Vault 20 has no API. The Dweller manages it by hand.
Tracks positions across any broker without automation.

Usage:
  python vault20/vault20.py add AAPL 10 185.50 [--target 200] [--stop 175] [--note "swing trade"]
  python vault20/vault20.py close AAPL 195.00
  python vault20/vault20.py status
  python vault20/vault20.py report        # status + post to Slack
"""
import argparse
import json
import os
import sys
from datetime import date, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "vault20_positions.json")


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

class Ledger:
    def __init__(self, path: str = DATA_FILE):
        self.path = path
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            data = json.loads(open(self.path).read())
        else:
            data = {"open": [], "closed": []}
        self.open   = data.get("open",   [])
        self.closed = data.get("closed", [])

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"open": self.open, "closed": self.closed}, f, indent=2)

    def add(self, symbol: str, shares: float, entry: float,
            entry_date: str, target: float | None = None,
            stop: float | None = None, note: str = "") -> dict:
        symbol = symbol.upper()
        if any(p["symbol"] == symbol for p in self.open):
            raise ValueError(f"{symbol} already open — close it first or use a different symbol")

        pos = {
            "symbol":     symbol,
            "shares":     shares,
            "entry":      entry,
            "entry_date": entry_date,
            "target":     target,
            "stop":       stop,
            "note":       note,
        }
        self.open.append(pos)
        self._save()
        return pos

    def close(self, symbol: str, exit_price: float, exit_date: str) -> dict:
        symbol = symbol.upper()
        match  = [p for p in self.open if p["symbol"] == symbol]
        if not match:
            raise ValueError(f"{symbol} not found in open positions")

        pos        = match[0]
        pnl_dollar = round((exit_price - pos["entry"]) * pos["shares"], 2)
        pnl_pct    = round((exit_price - pos["entry"]) / pos["entry"] * 100, 3)

        trade = {**pos,
                 "exit":       exit_price,
                 "exit_date":  exit_date,
                 "pnl_dollar": pnl_dollar,
                 "pnl_pct":    pnl_pct}
        self.closed.append(trade)
        self.open = [p for p in self.open if p["symbol"] != symbol]
        self._save()
        return trade

    def summary(self, prices: dict | None = None) -> dict:
        total_cost     = sum(p["entry"] * p["shares"] for p in self.open)
        realized_pnl   = sum(t["pnl_dollar"] for t in self.closed if t.get("pnl_dollar") is not None)
        unrealized_pnl = 0.0

        if prices:
            for p in self.open:
                cur = prices.get(p["symbol"])
                if cur is not None:
                    unrealized_pnl += (cur - p["entry"]) * p["shares"]

        wins = [t for t in self.closed if t.get("pnl_dollar") is not None and t["pnl_dollar"] > 0]
        return {
            "open_count":     len(self.open),
            "closed_count":   len(self.closed),
            "total_cost":     round(total_cost, 2),
            "realized_pnl":   round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_pnl":      round(realized_pnl + unrealized_pnl, 2),
            "win_count":      len(wins),
            "win_rate":       round(len(wins) / len(self.closed) * 100, 1) if self.closed else 0.0,
        }


# ---------------------------------------------------------------------------
# Price fetch
# ---------------------------------------------------------------------------

def _fetch_prices(symbols: list[str]) -> dict:
    """Fetch latest close prices via yfinance. Returns {} on failure.
    Skips option/synthetic symbols (contain underscores) — they have no yfinance ticker.
    """
    # filter to plain equity tickers only
    equities = [s for s in symbols if "_" not in s]
    if not equities:
        return {}
    try:
        import contextlib, io, yfinance as yf
        with contextlib.redirect_stderr(io.StringIO()):
            tickers = yf.download(equities, period="2d", interval="1d",
                                  progress=False, auto_adjust=True,
                                  multi_level_index=len(equities) > 1)
        if tickers.empty:
            return {}
        closes = tickers["Close"] if len(equities) > 1 else tickers[["Close"]]
        return {sym: float(closes[sym].dropna().iloc[-1])
                for sym in equities if sym in closes.columns}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_status(ledger: Ledger, prices: dict):
    s = ledger.summary(prices)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  VAULT 20 — POSITIONS  {now}")
    print(sep)

    if ledger.open:
        print(f"  Open ({s['open_count']}):")
        for pos in ledger.open:
            cur = prices.get(pos["symbol"])
            pnl_str = ""
            if cur is not None:
                usd = (cur - pos["entry"]) * pos["shares"]
                pct = (cur - pos["entry"]) / pos["entry"] * 100
                pnl_str = f"  → ${cur:.2f} ({pct:+.1f}%  ${usd:+.0f})"
            tgt_str  = f"  tgt ${pos['target']:.2f}" if pos["target"] else ""
            stp_str  = f"  stp ${pos['stop']:.2f}"   if pos["stop"]   else ""
            note_str = f"  [{pos['note']}]"           if pos["note"]   else ""
            print(f"    {pos['symbol']:<6} {pos['shares']} sh"
                  f" @ ${pos['entry']:.2f}{tgt_str}{stp_str}"
                  f"  [{pos['entry_date']}]{note_str}{pnl_str}")
    else:
        print("  Open:   none")

    print(f"  {'─'*56}")
    print(f"  Invested:        ${s['total_cost']:>10,.2f}")
    print(f"  Unrealized P&L:  ${s['unrealized_pnl']:>+10,.2f}")
    print(f"  Realized P&L:    ${s['realized_pnl']:>+10,.2f}")
    print(f"  Total P&L:       ${s['total_pnl']:>+10,.2f}")
    if ledger.closed:
        print(f"  Win rate:        {s['win_rate']:.0f}%"
              f"  ({s['win_count']}/{s['closed_count']} closed trades)")
    print(sep)


def _slack_status(ledger: Ledger, prices: dict) -> str:
    s   = ledger.summary(prices)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"*VAULT 20 — Positions* {now}"]

    if ledger.open:
        lines.append(f"*Open ({s['open_count']}):*")
        for pos in ledger.open:
            cur = prices.get(pos["symbol"])
            pnl_str = ""
            if cur is not None:
                usd = (cur - pos["entry"]) * pos["shares"]
                pct = (cur - pos["entry"]) / pos["entry"] * 100
                pnl_str = f" → ${cur:.2f} ({pct:+.1f}%, ${usd:+.0f})"
            note_str = f" [{pos['note']}]" if pos["note"] else ""
            lines.append(f"  • {pos['symbol']} {pos['shares']}sh"
                         f" @${pos['entry']:.2f} [{pos['entry_date']}]{note_str}{pnl_str}")
    else:
        lines.append("*Open:* none")

    lines.append(f"*Unrealized:* ${s['unrealized_pnl']:+,.2f}"
                 f"  |  *Realized:* ${s['realized_pnl']:+,.2f}"
                 f"  |  *Total P&L:* ${s['total_pnl']:+,.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _today() -> str:
    return date.today().isoformat()


def cmd_add(args, ledger: Ledger):
    pos = ledger.add(
        symbol     = args.symbol,
        shares     = args.shares,
        entry      = args.entry,
        entry_date = args.date or _today(),
        target     = args.target,
        stop       = args.stop,
        note       = args.note or "",
    )
    print(f"  ✓ Added {pos['symbol']} — {pos['shares']} sh @ ${pos['entry']:.2f}"
          f"  [{pos['entry_date']}]")
    if pos["target"]:
        print(f"    Target ${pos['target']:.2f}  |  Stop ${pos['stop'] or '—'}")
    if pos["note"]:
        print(f"    Note: {pos['note']}")


def cmd_close(args, ledger: Ledger):
    trade = ledger.close(
        symbol     = args.symbol,
        exit_price = args.exit_price,
        exit_date  = args.date or _today(),
    )
    icon = "✓" if trade["pnl_dollar"] >= 0 else "✗"
    print(f"  {icon} Closed {trade['symbol']} — ${trade['entry']:.2f} → ${trade['exit']:.2f}"
          f"  {trade['pnl_pct']:+.2f}%  ${trade['pnl_dollar']:+,.2f}")


def cmd_status(args, ledger: Ledger):
    syms   = [p["symbol"] for p in ledger.open]
    prices = _fetch_prices(syms)
    _print_status(ledger, prices)


def cmd_report(args, ledger: Ledger):
    syms   = [p["symbol"] for p in ledger.open]
    prices = _fetch_prices(syms)
    _print_status(ledger, prices)

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from notify_slack import send
    send(_slack_status(ledger, prices))
    print("  → Sent to Slack.")


def main():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    parser = argparse.ArgumentParser(prog="vault20",
                                     description="Vault 20 — manual position tracker")
    sub    = parser.add_subparsers(dest="cmd", required=True)

    # add
    p_add = sub.add_parser("add", help="Add a new position")
    p_add.add_argument("symbol")
    p_add.add_argument("shares",    type=float)
    p_add.add_argument("entry",     type=float)
    p_add.add_argument("--target",  type=float, default=None)
    p_add.add_argument("--stop",    type=float, default=None)
    p_add.add_argument("--date",    default=None, help="Entry date YYYY-MM-DD")
    p_add.add_argument("--note",    default="")

    # close
    p_cl = sub.add_parser("close", help="Close an open position")
    p_cl.add_argument("symbol")
    p_cl.add_argument("exit_price", type=float)
    p_cl.add_argument("--date",     default=None, help="Exit date YYYY-MM-DD")

    # status
    sub.add_parser("status", help="Show all positions with live prices")

    # report
    sub.add_parser("report", help="Show status and post to Slack")

    args   = parser.parse_args()
    ledger = Ledger()

    dispatch = {
        "add":    cmd_add,
        "close":  cmd_close,
        "status": cmd_status,
        "report": cmd_report,
    }
    dispatch[args.cmd](args, ledger)


if __name__ == "__main__":
    main()
