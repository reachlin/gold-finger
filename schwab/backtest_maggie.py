"""
Backtest for The Maggie role — Qullamaggie-style breakout strategy.

Only enters when Overseer classifies RECLAMATION (breakouts need a bull
regime — Maggie's optimal_regimes is RECLAMATION only) and her own scan()
confirms run-up + tight consolidation + range-expansion breakout.

State machine per symbol:
  FLAT -> scan for BUY signal
  LONG -> hold 100 shares; exit on first of:
            (a) stop hit  : close <= stop. Stop starts at the ATR/ADR-capped
                            breakout stop, then moves to breakeven the first
                            time the initial R-multiple target is reached —
                            approximates Qullamaggie's "sell 1/3-1/2, move
                            stop to breakeven, trail the rest"
            (b) trend end : close < EMA10, once trailing has begun (10-day
                            EMA close-below — Qullamaggie's beginner trail)
            (c) max hold  : 90 bars (~4.5 months) — safety valve

P&L: (exit_close - entry_close) * 100 shares per trade. This simplifies
Qullamaggie's partial profit-taking into a single full-position exit — it
does not separately account for selling a third of the position at the
first target and letting the remainder run.
"""
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from trend_scanner import compute_indicators
from vault76.armory.maggie import Maggie
from backtest_scavenger import _build_regime_lookup, _get_regime, MIN_HISTORY

SHARES   = 100
MAX_HOLD = 90   # bars — safety valve

FLAT = "FLAT"
LONG = "LONG"


def walk_forward_maggie(df: pd.DataFrame, symbol: str,
                        spy_df: pd.DataFrame | None = None,
                        vix_df: pd.DataFrame | None = None) -> list[dict]:
    """Simulate The Maggie breakout strategy. Returns closed trade events."""
    ind = compute_indicators(df).dropna().reset_index(drop=True)
    ind["_date"] = pd.to_datetime(ind["datetime"]).dt.date
    maggie = Maggie()
    n = len(ind)

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
            if "maggie" not in overseer.recommend_roles(regime, row):
                i += 1
                continue
            res = maggie.scan(symbol, snapshot, regime=regime)
            if res["signal"] == "BUY":
                pos = {
                    "symbol":      symbol,
                    "entry_i":     i,
                    "entry_date":  str(cur_date),
                    "entry_price": close,
                    "target":      res["target"],
                    "stop":        res["stop"],
                    "trailing":    False,
                    "max_exit_i":  min(i + MAX_HOLD, n - 1),
                    "regime":      regime,
                    "entry_rsi":   res.get("rsi"),
                    "entry_adx":   res.get("adx"),
                }
                state = LONG
            i += 1

        elif state == LONG:
            ema10 = float(row["ema10"])
            exit_reason = None

            if not pos["trailing"] and close >= pos["target"]:
                # First target reached: lock in breakeven, trail the rest.
                pos["stop"]     = pos["entry_price"]
                pos["trailing"] = True

            if pos["trailing"] and close < ema10:
                exit_reason = "maggie_trend_end"
            elif close <= pos["stop"]:
                exit_reason = "maggie_stop_hit"
            elif i >= pos["max_exit_i"]:
                exit_reason = "maggie_max_hold"

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
