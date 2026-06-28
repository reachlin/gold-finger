"""
Backtest for The Chemist role — bear put spread in NUKED_ZONE.

Only enters positions when VIX ≥ 30 (NUKED_ZONE). Holds through or exits
early on profit target / loss limit / regime change.

State machine per symbol:
  FLAT        → scan for BUY_PUT_SPREAD signal (only in NUKED_ZONE)
  SPREAD_OPEN → track spread value each bar; exit on one of:
                  (a) profit target: 70% of max profit
                  (b) loss limit:    50% of max loss
                  (c) expiry:        take intrinsic value
                  (d) regime exit:   NUKED_ZONE ended, close at market

Usage:
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/backtest_chemist.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from trend_scanner import compute_indicators
from vault76.armory.chemist import Chemist
from vault76.overseer import Overseer
from options_pricer import black_scholes_put, historical_vol

DTE_BARS      = 30     # trading bars ≈ 30 calendar DTE proxy
SHARES        = 100    # 1 contract = 100 shares
MIN_HISTORY   = 60
RISK_FREE     = 0.05
PROFIT_TARGET = 0.70   # buy back spread when it decays to 30% of credit (70% profit)
LOSS_LIMIT    = 2.00   # cut when spread value reaches 200% of credit received

FLAT        = "FLAT"
SPREAD_OPEN = "SPREAD_OPEN"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

WATCHLIST = [
    "NVDA", "AMD", "AAPL", "AMZN", "META", "MSFT", "GOOGL",
    "IBM", "INTC", "IONQ", "KO", "MMM", "XOM", "PG", "TSLA",
    "UNH", "HD", "ABT",
]


# ---------------------------------------------------------------------------
# Regime lookup (shared logic with backtest_scavenger)
# ---------------------------------------------------------------------------

def _build_regime_lookup(spy_df: pd.DataFrame | None,
                         vix_df: pd.DataFrame | None) -> tuple:
    overseer = Overseer()
    spy_ind, spy_date_idx, vix_by_date = None, {}, {}

    if spy_df is not None:
        spy_ind = compute_indicators(spy_df).dropna().reset_index(drop=True)
        spy_dates = pd.to_datetime(spy_ind["datetime"]).dt.date
        spy_date_idx = {d: i for i, d in enumerate(spy_dates)}

    if vix_df is not None:
        col = "datetime" if "datetime" in vix_df.columns else "date"
        dates = pd.to_datetime(vix_df[col]).dt.date
        vix_by_date = dict(zip(dates, vix_df["close"]))

    return overseer, spy_ind, spy_date_idx, vix_by_date


def _get_regime(overseer, cur_date, spy_ind, spy_date_idx, vix_by_date):
    vix = vix_by_date.get(cur_date, 20.0)
    if vix >= 30.0:
        return Overseer.NUKED_ZONE
    if spy_ind is None:
        return Overseer.WASTELAND
    spy_i = spy_date_idx.get(cur_date)
    if spy_i is None or spy_i < MIN_HISTORY:
        return Overseer.WASTELAND
    return overseer.classify(spy_ind.iloc[:spy_i + 1], vix)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _spread_mark(close: float, short_strike: float, long_strike: float,
                 bars_remaining: int, hv: float) -> float:
    """
    Current mark-to-market COST to close the credit spread (buy it back).
    short_strike > long_strike (we sold short, bought long for protection).
    credit spread cost = short_put_value - long_put_value
    """
    T = max(bars_remaining / 365, 1 / 365)
    short_val = black_scholes_put(close, short_strike, T, RISK_FREE, hv)
    long_val  = black_scholes_put(close, long_strike,  T, RISK_FREE, hv)
    return short_val - long_val   # cost to buy back = close the position


def walk_forward_chemist(df: pd.DataFrame, symbol: str,
                         spy_df: pd.DataFrame | None = None,
                         vix_df: pd.DataFrame | None = None) -> list[dict]:
    """
    Simulate The Chemist put-spread strategy.
    Returns list of closed trade events with P&L.
    """
    ind = compute_indicators(df).dropna().reset_index(drop=True)
    ind["_date"] = pd.to_datetime(ind["datetime"]).dt.date
    chem = Chemist()
    n    = len(ind)

    overseer, spy_ind, spy_date_idx, vix_by_date = _build_regime_lookup(spy_df, vix_df)

    events = []
    state  = FLAT
    pos: dict = {}

    i = MIN_HISTORY
    while i < n:
        row      = ind.iloc[i]
        snapshot = ind.iloc[:i + 1]
        cur_date = row["_date"]
        regime   = _get_regime(overseer, cur_date, spy_ind, spy_date_idx, vix_by_date)
        close    = float(row["close"])

        # ── FLAT: look for entry in NUKED_ZONE ───────────────────────────────
        if state == FLAT:
            if regime == Overseer.NUKED_ZONE:
                res = chem.scan(symbol, snapshot, regime=regime)
                if res["signal"] == "SELL_PUT_SPREAD":
                    pos = {
                        "symbol":       symbol,
                        "entry_i":      i,
                        "entry_date":   str(cur_date),
                        "short_strike": res["short_strike"],  # the one we sold
                        "long_strike":  res["long_strike"],   # the protection leg
                        "net_credit":   res["net_credit"],
                        "max_profit":   res["max_profit"],
                        "max_loss":     res["max_loss"],
                        "expiry_i":     min(i + DTE_BARS, n - 1),
                        "entry_close":  close,
                        "regime":       regime,
                    }
                    state = SPREAD_OPEN
            i += 1

        # ── SPREAD_OPEN: track and exit ───────────────────────────────────────
        elif state == SPREAD_OPEN:
            hv             = historical_vol(snapshot["close"])
            bars_remaining = pos["expiry_i"] - i
            # cost_to_close = current value of spread we're short
            cost_to_close  = _spread_mark(close, pos["short_strike"],
                                          pos["long_strike"], bars_remaining, hv)
            # unrealized P&L: we collected net_credit, now it costs cost_to_close to exit
            unrealized_pnl = (pos["net_credit"] - cost_to_close) * SHARES

            exit_reason = None

            if i >= pos["expiry_i"]:
                # Expiry: pay intrinsic value, keep the rest
                intrinsic_short = max(pos["short_strike"] - close, 0)
                intrinsic_long  = max(pos["long_strike"]  - close, 0)
                cost_at_expiry  = intrinsic_short - intrinsic_long
                pnl = (pos["net_credit"] - cost_at_expiry) * SHARES
                exit_reason = "spread_expired"

            elif unrealized_pnl >= PROFIT_TARGET * pos["max_profit"]:
                # Spread decayed — buy back cheap
                pnl = unrealized_pnl
                exit_reason = "spread_profit_target"

            elif unrealized_pnl <= -LOSS_LIMIT * pos["max_loss"]:
                # Spread expanded — cut loss
                pnl = unrealized_pnl
                exit_reason = "spread_loss_limit"

            if exit_reason:
                events.append({
                    **pos,
                    "exit_i":     i,
                    "exit_date":  str(cur_date),
                    "exit_close": close,
                    "event":      exit_reason,
                    "pnl":        round(pnl, 2),
                })
                state = FLAT
                pos   = {}

            i += 1

    return events


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_chemist_report(events: list[dict], symbol: str = "") -> float:
    expired_cnt = sum(1 for e in events if e["event"] == "spread_expired")
    profit_cnt  = sum(1 for e in events if e["event"] == "spread_profit_target")
    loss_cnt    = sum(1 for e in events if e["event"] == "spread_loss_limit")
    total_pnl   = sum(e["pnl"] for e in events)
    wins        = sum(1 for e in events if e["pnl"] > 0)
    total       = len(events)
    win_rate    = wins / total * 100 if total else 0.0

    label = f" [{symbol}]" if symbol else ""
    print(f"{'─'*55}")
    print(f"  THE CHEMIST — Credit Spread Backtest{label}")
    print(f"{'─'*55}")
    if total == 0:
        print(f"  No NUKED_ZONE trades entered.")
        print(f"{'─'*55}")
        return 0.0
    print(f"  Trades:         {total}  (win rate {win_rate:.0f}%)")
    print(f"  Profit targets: {profit_cnt}  |  Expired: {expired_cnt}  |  Stopped: {loss_cnt}")
    print(f"  Chemist P&L:    ${total_pnl:+.2f}")
    print(f"{'─'*55}")
    return total_pnl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    spy_path = os.path.join(DATA_DIR, "spy_history.csv")
    vix_path = os.path.join(DATA_DIR, "vix_history.csv")
    spy_df = pd.read_csv(spy_path, parse_dates=["datetime"]) if os.path.exists(spy_path) else None
    vix_df = pd.read_csv(vix_path, parse_dates=["datetime"]) if os.path.exists(vix_path) else None

    if vix_df is None:
        print("ERROR: data/vix_history.csv missing — NUKED_ZONE cannot be detected.")
        return

    results = []
    for symbol in WATCHLIST:
        path = os.path.join(DATA_DIR, f"{symbol.lower()}_history.csv")
        if not os.path.exists(path):
            print(f"  [{symbol}] no data file — skip")
            continue
        df = pd.read_csv(path, parse_dates=["datetime"])
        if len(df) < MIN_HISTORY + DTE_BARS + 10:
            continue

        events = walk_forward_chemist(df, symbol, spy_df=spy_df, vix_df=vix_df)
        total_pnl = print_chemist_report(events, symbol)
        results.append({"symbol": symbol, "trades": len(events), "pnl": total_pnl})

    if not results:
        return

    results.sort(key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'='*55}")
    print(f"  THE CHEMIST — NUKED_ZONE Summary (all symbols)")
    print(f"{'='*55}")
    print(f"  {'Symbol':<8} {'Trades':>6}   {'Chemist P&L':>12}")
    print(f"  {'─'*8} {'─'*6}   {'─'*12}")
    for r in results:
        print(f"  {r['symbol']:<8} {r['trades']:>6}   ${r['pnl']:>+11,.2f}")
    total = sum(r["pnl"] for r in results)
    print(f"  {'─'*8} {'─'*6}   {'─'*12}")
    print(f"  {'TOTAL':<8} {'':>6}   ${total:>+11,.2f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
