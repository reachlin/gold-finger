"""
Backtest for The Medic — crisis accumulation of defensive ETFs.

Rule (mirrors vault76/armory/medic.py):
  BUY  100 shares at the close of the first NUKED_ZONE bar (VIX >= 30)
  HOLD through WASTELAND (calm VIX is not yet a recovery)
  SELL at the close of the first RECLAMATION bar (SPY above rising EMA50)

Candidate roster is empirical: quality dividend / low-vol / staples ETFs
should profit (bought cheap in panic, sold into recovery); flight-to-
safety assets (TLT, GLD) are expected to FAIL this specific entry — they
are expensive during panic — and are included to prove the point.

Usage:
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/backtest_medic.py
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/backtest_medic.py SCHD GLD
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from trend_scanner import compute_indicators
from vault76.overseer import Overseer
from backtest_scavenger import (MIN_HISTORY, _build_regime_lookup,
                                _get_regime)

SHARES     = 100
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
CANDIDATES = ["SCHD", "VIG", "VYM", "NOBL", "USMV", "XLP", "GLD", "TLT"]


def walk_forward_medic(df: pd.DataFrame, symbol: str,
                       spy_df=None, vix_df=None,
                       entry_mode: str = "nuke") -> list[dict]:
    """
    One buy / sell-at-reclamation episode at a time.
    entry_mode "nuke": buy the first NUKED_ZONE bar (panic entry).
    entry_mode "calm": buy the first bar AFTER a NUKED episode ends —
    waits for the knife to land instead of catching it.
    """
    ind = compute_indicators(df).dropna().reset_index(drop=True)
    ind["_date"] = pd.to_datetime(ind["datetime"]).dt.date
    overseer, spy_ind, spy_date_idx, vix_by_date = \
        _build_regime_lookup(spy_df, vix_df)

    events, holding, was_nuked = [], None, False
    for i in range(MIN_HISTORY, len(ind)):
        row    = ind.iloc[i]
        close  = float(row["close"])
        regime = _get_regime(overseer, row["_date"], spy_ind,
                             spy_date_idx, vix_by_date)

        entry_now = (regime == Overseer.NUKED_ZONE if entry_mode == "nuke"
                     else was_nuked and regime != Overseer.NUKED_ZONE)
        was_nuked = regime == Overseer.NUKED_ZONE

        if holding is None and entry_now:
            holding = {"entry_i": i, "entry_close": close,
                       "entry_date": str(row["_date"])}
        elif holding is not None and regime == Overseer.RECLAMATION:
            events.append({"symbol": symbol, "event": "medic_sell",
                           **holding, "exit_i": i, "exit_close": close,
                           "exit_date": str(row["_date"]),
                           "pnl": round((close - holding["entry_close"])
                                        * SHARES, 2)})
            holding = None

    if holding is not None:
        close = float(ind.iloc[-1]["close"])
        events.append({"symbol": symbol, "event": "end_liquidate",
                       **holding, "exit_i": len(ind) - 1,
                       "exit_close": close,
                       "exit_date": str(ind.iloc[-1]["_date"]),
                       "pnl": round((close - holding["entry_close"])
                                    * SHARES, 2)})

    # Tag buys for readability (entry recorded inside sell events too)
    buys = [{"symbol": symbol, "event": "medic_buy", "pnl": 0.0,
             "entry_i": e["entry_i"], "entry_close": e["entry_close"],
             "entry_date": e["entry_date"]} for e in events]
    return sorted(buys + events, key=lambda e: e["entry_i"])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--entry", choices=("nuke", "calm"), default="nuke")
    args = ap.parse_args()
    symbols = [s.upper() for s in args.symbols] or CANDIDATES

    spy_df = pd.read_csv(os.path.join(DATA_DIR, "spy_history.csv"),
                         parse_dates=["datetime"])
    vix_df = pd.read_csv(os.path.join(DATA_DIR, "vix_history.csv"),
                         parse_dates=["datetime"])

    t0 = time.time()
    print("═" * 92)
    print(f"  THE MEDIC — entry mode: {args.entry.upper()} "
          f"({'buy first panic bar' if args.entry == 'nuke' else 'buy first calm bar after panic'})"
          f", sell at RECLAMATION")
    print("  100 shares per episode, dividend-adjusted closes")
    print("═" * 92)
    print(f"  {'Symbol':7s}{'Episodes':>9s}{'Wins':>6s}{'Total P&L':>13s}"
          f"{'Avg/episode':>13s}{'Hold B&H':>13s}{'Edge':>13s}")
    print("  " + "─" * 86)
    total = 0.0
    for sym in symbols:
        path = os.path.join(DATA_DIR, f"{sym.lower()}_history.csv")
        if not os.path.exists(path):
            print(f"  {sym:7s} no data")
            continue
        df = pd.read_csv(path, parse_dates=["datetime"])
        events = walk_forward_medic(df, sym, spy_df, vix_df,
                                    entry_mode=args.entry)
        sells  = [e for e in events if e["event"] in ("medic_sell",
                                                      "end_liquidate")]
        pnl    = sum(e["pnl"] for e in sells)
        wins   = sum(1 for e in sells if e["pnl"] > 0)
        ind    = compute_indicators(df).dropna().reset_index(drop=True)
        bnh    = (float(ind.iloc[-1]["close"])
                  - float(ind.iloc[MIN_HISTORY]["close"])) * SHARES
        total += pnl
        avg = pnl / len(sells) if sells else 0.0
        print(f"  {sym:7s}{len(sells):>9d}{wins:>6d}{pnl:>+13,.0f}"
              f"{avg:>+13,.0f}{bnh:>+13,.0f}{pnl - bnh:>+13,.0f}")
        for e in sells:
            print(f"          {e['entry_date']} → {e['exit_date']}"
                  f"  ${e['entry_close']:.2f} → ${e['exit_close']:.2f}"
                  f"  {e['pnl']:+,.0f}")
    print("  " + "─" * 86)
    print(f"  TOTAL   {total:+,.0f}")
    print(f"  Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
