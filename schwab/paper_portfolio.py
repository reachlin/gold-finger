"""
Paper trading portfolio tracker with balance tracking and event logging.

State:  data/paper_trades.json   — positions, cash, balance history
Log:    data/paper_trading.log   — JSONL event log (one event per line)
"""
import os
import json
import uuid
from datetime import datetime, timedelta

MAX_HOLD_DAYS = 30


class PaperPortfolio:

    def __init__(self, path: str, starting_capital: float = 30_000.0):
        self.path     = path
        self.log_path = path.replace(".json", ".log")
        self._load(starting_capital)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self, starting_capital: float = 30_000.0):
        if os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)
            bal = data.get("balance", {})
            self.starting_capital = bal.get("starting_capital", starting_capital)
            self.cash             = bal.get("cash", self.starting_capital)
        else:
            self.starting_capital = starting_capital
            self.cash             = starting_capital
            data                  = {}
        self.open_positions   = data.get("open", [])
        self.closed_positions = data.get("closed", [])

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "balance": {
                    "starting_capital": self.starting_capital,
                    "cash":             round(self.cash, 4),
                },
                "open":   self.open_positions,
                "closed": self.closed_positions,
            }, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # JSONL event log
    # ------------------------------------------------------------------

    def _log(self, event: dict):
        event["ts"] = datetime.now().isoformat(timespec="seconds")
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def log_scan(self, scan_num: int, symbols_scanned: int, signals_found: int):
        self._log({
            "event":           "SCAN",
            "scan_num":        scan_num,
            "symbols_scanned": symbols_scanned,
            "signals_found":   signals_found,
            "cash":            round(self.cash, 2),
            "open_positions":  len(self.open_positions),
        })

    def log_signal(self, symbol: str, entry: float, target: float, stop: float,
                   rsi: float, adx: float, verdict: str, reason: str = ""):
        self._log({
            "event":   "SIGNAL",
            "symbol":  symbol,
            "entry":   entry,
            "target":  target,
            "stop":    stop,
            "rsi":     rsi,
            "adx":     adx,
            "verdict": verdict,
            "reason":  reason,
        })

    # ------------------------------------------------------------------
    # Open a position
    # ------------------------------------------------------------------

    def open_position(self, symbol: str, entry: float, target: float,
                      stop: float, shares: int) -> dict:
        # Cap shares to available cash
        max_shares = max(1, int(self.cash / entry)) if self.cash >= entry else 0
        if max_shares == 0:
            self._log({"event": "BUY_REJECTED", "symbol": symbol,
                       "reason": "insufficient cash", "cash": round(self.cash, 2)})
            return {}
        shares = min(shares, max_shares)
        cost   = round(shares * entry, 4)

        self.cash -= cost
        pos = {
            "trade_id":   str(uuid.uuid4())[:8],
            "symbol":     symbol,
            "entry":      entry,
            "target":     target,
            "stop":       stop,
            "shares":     shares,
            "cost":       cost,
            "entry_date": datetime.now().isoformat(timespec="seconds"),
        }
        self.open_positions.append(pos)
        self._save()
        self._log({
            "event":      "BUY",
            "trade_id":   pos["trade_id"],
            "symbol":     symbol,
            "shares":     shares,
            "entry":      entry,
            "target":     target,
            "stop":       stop,
            "cost":       cost,
            "cash_after": round(self.cash, 2),
        })
        return pos

    # ------------------------------------------------------------------
    # Check all open positions
    # ------------------------------------------------------------------

    def check_positions(self, price_fetcher) -> list[dict]:
        """
        price_fetcher(symbol) → {"high", "low", "close", "ema20", "ema50"}
        Returns list of newly-closed positions.
        """
        closed_now: list[dict] = []
        remaining:  list[dict] = []

        for pos in self.open_positions:
            try:
                prices = price_fetcher(pos["symbol"])
            except Exception:
                remaining.append(pos)
                continue

            exit_price  = None
            exit_reason = None

            if prices["high"] >= pos["target"]:
                exit_price, exit_reason = pos["target"], "target"
            elif prices["low"] <= pos["stop"]:
                exit_price, exit_reason = pos["stop"], "stop"
            elif prices.get("ema20", 1.0) < prices.get("ema50", 0.0):
                exit_price, exit_reason = prices["close"], "trend_end"
            else:
                hold = (datetime.now() - datetime.fromisoformat(pos["entry_date"])).days
                if hold >= MAX_HOLD_DAYS:
                    exit_price, exit_reason = prices["close"], "timeout"

            if exit_price is not None:
                exit_price  = round(exit_price, 4)
                pnl_dollar  = round(pos["shares"] * (exit_price - pos["entry"]), 4)
                pnl_pct     = round((exit_price - pos["entry"]) / pos["entry"] * 100, 2)
                proceeds    = round(pos["shares"] * exit_price, 4)
                self.cash  += proceeds

                closed = {
                    **pos,
                    "exit":        exit_price,
                    "exit_reason": exit_reason,
                    "exit_date":   datetime.now().isoformat(timespec="seconds"),
                    "pnl_pct":     pnl_pct,
                    "pnl_dollar":  pnl_dollar,
                    "proceeds":    proceeds,
                }
                self.closed_positions.append(closed)
                closed_now.append(closed)
                self._log({
                    "event":       "EXIT",
                    "trade_id":    pos["trade_id"],
                    "symbol":      pos["symbol"],
                    "exit":        exit_price,
                    "exit_reason": exit_reason,
                    "pnl_dollar":  pnl_dollar,
                    "pnl_pct":     pnl_pct,
                    "proceeds":    proceeds,
                    "cash_after":  round(self.cash, 2),
                })
            else:
                remaining.append(pos)

        self.open_positions = remaining
        self._save()
        return closed_now

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------

    def summary(self, current_prices: dict | None = None) -> dict:
        closed = self.closed_positions
        wins   = [t for t in closed if t["pnl_dollar"] > 0]

        realized_pnl = sum(t["pnl_dollar"] for t in closed)
        win_rate     = len(wins) / len(closed) * 100 if closed else 0.0

        # Unrealized P&L and invested value
        unrealized = 0.0
        invested   = 0.0
        for pos in self.open_positions:
            invested += pos["cost"]
            if current_prices and pos["symbol"] in current_prices:
                cur        = current_prices[pos["symbol"]]
                unrealized += pos["shares"] * (cur - pos["entry"])

        total_value = round(self.cash + invested + unrealized, 2)

        return {
            "starting_capital":     self.starting_capital,
            "cash":                 round(self.cash, 2),
            "invested":             round(invested, 2),
            "total_value":          total_value,
            "realized_pnl_dollar":  round(realized_pnl, 2),
            "unrealized_pnl_dollar": round(unrealized, 2),
            "total_pnl_dollar":     round(realized_pnl + unrealized, 2),
            "total_pnl_pct":        round((total_value - self.starting_capital)
                                          / self.starting_capital * 100, 2),
            "open_count":           len(self.open_positions),
            "closed_count":         len(closed),
            "win_count":            len(wins),
            "win_rate":             round(win_rate, 1),
        }

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_status(self, current_prices: dict | None = None):
        s = self.summary(current_prices)
        pnl_sign = "+" if s["total_pnl_dollar"] >= 0 else ""

        print("\n  ┌─ Paper Portfolio " + "─" * 40)
        print(f"  │ Capital:    ${s['starting_capital']:>10,.2f}  starting")
        print(f"  │ Cash:       ${s['cash']:>10,.2f}  available")
        print(f"  │ Invested:   ${s['invested']:>10,.2f}  in {s['open_count']} position(s)")
        print(f"  │ Total:      ${s['total_value']:>10,.2f}")
        print(f"  │ P&L:        ${pnl_sign}{s['total_pnl_dollar']:>9,.2f}  "
              f"({pnl_sign}{s['total_pnl_pct']:.2f}%)"
              f"  [realized ${s['realized_pnl_dollar']:+,.2f} /"
              f" unrealized ${s['unrealized_pnl_dollar']:+,.2f}]")

        if self.open_positions:
            print(f"  ├─ Open ({s['open_count']})")
            for pos in self.open_positions:
                cur   = current_prices.get(pos["symbol"]) if current_prices else None
                unreal_str = ""
                if cur is not None:
                    usd = pos["shares"] * (cur - pos["entry"])
                    pct = (cur - pos["entry"]) / pos["entry"] * 100
                    unreal_str = f"  cur ${cur:.2f} ({pct:+.1f}%  ${usd:+.2f})"
                since = pos["entry_date"][:10]
                print(f"  │   {pos['symbol']:<6} {pos['shares']} sh"
                      f" @ ${pos['entry']:.2f}"
                      f"  tgt ${pos['target']:.2f}"
                      f"  stp ${pos['stop']:.2f}"
                      f"  [{since}]{unreal_str}")

        if self.closed_positions:
            recent = self.closed_positions[-8:]
            print(f"  ├─ Closed ({s['closed_count']} total,"
                  f" {s['win_count']} wins,"
                  f" win rate {s['win_rate']:.0f}%)")
            for t in recent:
                icon = "✓" if t["pnl_dollar"] > 0 else "✗"
                print(f"  │   {icon} {t['symbol']:<6}"
                      f" {t['pnl_pct']:>+6.1f}%  ${t['pnl_dollar']:>+8.2f}"
                      f"  [{t['exit_reason']:<9}]"
                      f"  entry ${t['entry']:.2f} → exit ${t['exit']:.2f}"
                      f"  {t['entry_date'][:10]}")

        print("  └" + "─" * 57)
