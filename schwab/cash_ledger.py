"""
Cash ledger — append-only audit log for every cash movement.

File: data/cash_ledger.csv
Columns: timestamp, event_type, symbol, amount, running_balance, description

Event types
-----------
STARTING_CAPITAL   initial balance (written once on first run)
DEPOSIT            external cash added to the account (raises invested capital)
WITHDRAWAL         external cash removed (negative; lowers invested capital)
OPTION_SELL        premium received when SELL_PUT / SELL_CALL is approved
OPTION_BUYBACK     cost to close a position early (negative)
OPTION_EXPIRED     expired worthless — $0 delta (premium already credited)
OPTION_ASSIGNED    cash paid for assigned shares = -(strike × 100) (negative)
OPTION_CALLED_AWAY cash received when shares called away = +(call_strike × 100)
STOCK_BUY          Raider / Maggie share purchase (negative)
STOCK_SELL         Raider / Maggie share sale (positive)
"""
import csv
import os
from datetime import datetime
from zoneinfo import ZoneInfo

FIELDNAMES = [
    "timestamp", "event_type", "symbol", "amount", "running_balance", "description"
]

_ET = ZoneInfo("America/New_York")


def _now() -> str:
    return datetime.now(_ET).strftime("%Y-%m-%d %H:%M:%S ET")


class CashLedger:
    def __init__(self, path: str, starting_capital: float = 30_000.0):
        self.path = path
        self.starting_capital = starting_capital
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
            self._append("STARTING_CAPITAL", "", starting_capital,
                         f"starting capital ${starting_capital:,.2f}", starting_capital)

    def _append(self, event_type: str, symbol: str, amount: float,
                description: str, running_balance: float):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow({
                "timestamp":       _now(),
                "event_type":      event_type,
                "symbol":          symbol,
                "amount":          round(amount, 2),
                "running_balance": round(running_balance, 2),
                "description":     description,
            })

    def balance(self) -> float:
        rows = self.rows()
        if not rows:
            return 0.0
        return float(rows[-1]["running_balance"])

    def record(self, event_type: str, symbol: str, amount: float, description: str):
        bal = self.balance() + amount
        self._append(event_type, symbol, amount, description, bal)

    def record_deposit(self, amount: float, description: str = ""):
        """External cash in (+) / out (-). Moves both the running balance AND
        invested capital — it is contributed money, not trading P&L."""
        etype = "DEPOSIT" if amount >= 0 else "WITHDRAWAL"
        self.record(etype, "", amount,
                    description or f"{etype.lower()} ${abs(amount):,.2f}")

    def invested_capital(self) -> float:
        """Net contributed capital = starting capital + deposits − withdrawals.
        This is the basis to measure P&L against — unlike balance(), it does not
        move with trades. P&L = balance() − invested_capital()."""
        total = self.starting_capital
        for r in self.rows():
            if r.get("event_type") in ("DEPOSIT", "WITHDRAWAL"):
                try:
                    total += float(r.get("amount", 0) or 0)
                except (ValueError, TypeError):
                    pass
        return round(total, 2)

    def rows(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, newline="") as f:
            return list(csv.DictReader(f))


def from_options_event(ledger: CashLedger, ev: dict, opening_row: dict | None = None):
    """
    Record a cash entry from an options_ledger settlement event dict:
      {"action", "symbol", "pnl", "detail"}

    action map:
      put_expired    → OPTION_EXPIRED  $0   (premium already credited at open)
      call_expired   → OPTION_EXPIRED  $0
      early_exit     → OPTION_BUYBACK  -(premium_ct - pnl)  [what we paid back]
      put_assigned   → OPTION_ASSIGNED -(strike × 100)
      called_away    → OPTION_CALLED_AWAY +(call_strike × 100)
    """
    action = ev.get("action", "")
    sym    = ev.get("symbol", "")
    detail = ev.get("detail", "")
    pnl    = float(ev.get("pnl", 0))

    if action in ("put_expired", "call_expired"):
        ledger.record("OPTION_EXPIRED", sym, 0.0,
                      f"{action}: {detail} (premium already credited)")

    elif action == "early_exit":
        premium_ct = 0.0
        if opening_row:
            try:
                premium_ct = float(opening_row.get("premium_ct", 0))
            except (ValueError, TypeError):
                pass
        buyback = premium_ct - pnl
        ledger.record("OPTION_BUYBACK", sym, -round(buyback, 2),
                      f"early_exit buyback: {detail}")

    elif action == "put_assigned":
        if opening_row:
            try:
                strike_cash = float(opening_row["strike"]) * 100
                ledger.record("OPTION_ASSIGNED", sym, -round(strike_cash, 2),
                              f"assigned: {detail}")
            except (ValueError, KeyError):
                pass

    elif action == "called_away":
        if opening_row:
            try:
                strike_cash = float(opening_row["strike"]) * 100
                ledger.record("OPTION_CALLED_AWAY", sym, round(strike_cash, 2),
                              f"called away: {detail}")
            except (ValueError, KeyError):
                pass
