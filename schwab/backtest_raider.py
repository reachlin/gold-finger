"""
Backtest for The Raider role — pullback-in-trend long strategy.

Only enters when Overseer classifies RECLAMATION or WASTELAND. Blocked in NUKED_ZONE.

State machine per symbol:
  FLAT → scan for BUY signal
  LONG → hold 100 shares; exit on first of:
           (a) target hit  : close ≥ entry × 1.20 (+20%)
           (b) trend end   : EMA20 < EMA50          (primary — follow the trend)
           (c) stop hit    : close ≤ entry × 0.92   (-8%)
           (d) max hold    : 60 bars (~3 months)     (safety valve)

P&L: (exit_close − entry_close) × 100 shares per trade.
"""
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from trend_scanner import compute_indicators
from vault76.armory.raider import Raider
from backtest_scavenger import _build_regime_lookup, _get_regime, MIN_HISTORY

SHARES   = 100
MAX_HOLD = 60   # bars — safety valve

FLAT = "FLAT"
LONG = "LONG"

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
WATCHLIST = [
    "NVDA", "AMD", "AAPL", "AMZN", "META", "MSFT", "GOOGL",
    "IBM",  "INTC", "IONQ", "KO",  "MMM",  "XOM",  "PG", "TSLA",
    "UNH",  "HD",   "ABT",
]


def walk_forward_raider(df: pd.DataFrame, symbol: str,
                        spy_df: pd.DataFrame | None = None,
                        vix_df: pd.DataFrame | None = None) -> list[dict]:
    """Simulate The Raider pullback-in-trend strategy. Returns closed trade events."""
    ind = compute_indicators(df).dropna().reset_index(drop=True)
    ind["_date"] = pd.to_datetime(ind["datetime"]).dt.date
    raid = Raider()
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
        close    = float(row["close"])
        regime   = _get_regime(overseer, cur_date, spy_ind, spy_date_idx, vix_by_date)

        if state == FLAT:
            res = raid.scan(symbol, snapshot, regime=regime)
            if res["signal"] == "BUY":
                pos = {
                    "symbol":      symbol,
                    "entry_i":     i,
                    "entry_date":  str(cur_date),
                    "entry_price": close,
                    "target":      res["target"],
                    "stop":        res["stop"],
                    "max_exit_i":  min(i + MAX_HOLD, n - 1),
                    "regime":      regime,
                    "entry_rsi":   res.get("rsi"),
                    "entry_adx":   res.get("adx"),
                }
                state = LONG
            i += 1

        elif state == LONG:
            ema20 = float(row["ema20"])
            ema50 = float(row["ema50"])
            exit_reason = None

            if close >= pos["target"]:
                exit_reason = "raider_target_hit"
            elif ema20 < ema50:
                exit_reason = "raider_trend_end"
            elif close <= pos["stop"]:
                exit_reason = "raider_stop_hit"
            elif i >= pos["max_exit_i"]:
                exit_reason = "raider_max_hold"

            if exit_reason:
                pnl = (close - pos["entry_price"]) * SHARES
                events.append({
                    **pos,
                    "exit_i":     i,
                    "exit_date":  str(cur_date),
                    "exit_price": close,
                    "event":      exit_reason,
                    "pnl":        round(pnl, 2),
                })
                state = FLAT
                pos   = {}
            i += 1

    return events
