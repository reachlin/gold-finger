"""
Wheel-vs-hold router experiment — measures wheel_router.py mechanically.

For each symbol:
  1. Compute the walk-forward P(>8% rally in 30d) series once (quarterly
     LGBM refits, no lookahead — see wheel_router.py).
  2. Run the scavenger backtest as-is (baseline) and with the router at
     several thresholds. Router ON: predicted runners are held as shares
     (uncapped) instead of wheeled; covered calls are skipped while hot.
  3. Compare each variant's edge vs buy-and-hold.

Baseline reference: checkpoint 2026-07-04 — scavenger-only edge is what
this experiment moves; Raider/Chemist books are untouched.

Usage:
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/backtest_router.py
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/backtest_router.py AMD GOOGL
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from trend_scanner import compute_indicators
from backtest_scavenger import walk_forward_scavenger, MIN_HISTORY, DTE_BARS
import wheel_router
from run_backtest import WATCHLIST, _load_df, _bnh_pnl

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
THRESHOLDS = (0.50, 0.60, 0.70)


def run_symbol(symbol: str, spy_df, vix_df) -> "dict | None":
    df = _load_df(symbol)
    if df is None:
        return None

    # Same post-dropna frame the scavenger backtest iterates — probs align 1:1
    ind   = compute_indicators(df).dropna().reset_index(drop=True)
    probs = wheel_router.walk_forward_probs(ind)

    base = walk_forward_scavenger(df, symbol, spy_df, vix_df)
    row  = {
        "symbol":   symbol,
        "bnh":      _bnh_pnl(df),
        "base_pnl": sum(e["pnl"] for e in base),
    }
    for tau in THRESHOLDS:
        events = walk_forward_scavenger(df, symbol, spy_df, vix_df,
                                        router_probs=probs,
                                        router_threshold=tau)
        holds  = [e for e in events if e["event"] == "router_hold_exit"]
        row[f"pnl@{tau}"]   = sum(e["pnl"] for e in events)
        row[f"holds@{tau}"] = len(holds)
        row[f"hpnl@{tau}"]  = sum(e["pnl"] for e in holds)
    return row


def main():
    symbols = [s.upper() for s in sys.argv[1:]] or WATCHLIST

    spy_path = os.path.join(DATA_DIR, "spy_history.csv")
    vix_path = os.path.join(DATA_DIR, "vix_history.csv")
    spy_df = pd.read_csv(spy_path, parse_dates=["datetime"]) if os.path.exists(spy_path) else None
    vix_df = pd.read_csv(vix_path, parse_dates=["datetime"]) if os.path.exists(vix_path) else None

    t0   = time.time()
    rows = []
    for sym in symbols:
        print(f"  {sym}...", flush=True)
        row = run_symbol(sym, spy_df, vix_df)
        if row is not None:
            rows.append(row)

    # ── Report ────────────────────────────────────────────────────────────
    taus = THRESHOLDS
    print()
    print("═" * 100)
    print("  WHEEL-vs-HOLD ROUTER  —  Scavenger book only, edge vs B&H per variant")
    print("  Router: hold shares uncapped while walk-forward P(>8% rally in 30d) >= threshold")
    print("═" * 100)
    hdr = f"  {'Symbol':7s}{'B&H':>12s}{'base edge':>12s}"
    for tau in taus:
        hdr += f"{f'edge@{tau:.2f}':>12s}"
    hdr += f"{'holds@' + format(taus[1], '.2f'):>12s}"
    print(hdr)
    print("  " + "─" * 96)

    def edge(row, key):
        return row[key] - row["bnh"]

    rows.sort(key=lambda r: edge(r, "base_pnl"))
    tot = {"bnh": 0.0, "base_pnl": 0.0, **{f"pnl@{t}": 0.0 for t in taus}}
    beats = {"base_pnl": 0, **{f"pnl@{t}": 0 for t in taus}}
    for r in rows:
        line = f"  {r['symbol']:7s}{r['bnh']:>+12,.0f}{edge(r, 'base_pnl'):>+12,.0f}"
        for tau in taus:
            line += f"{edge(r, f'pnl@{tau}'):>+12,.0f}"
        line += f"{r[f'holds@{taus[1]}']:>12d}"
        print(line)
        for k in tot:
            tot[k] += r[k]
        for k in beats:
            if edge(r, k) > 0:
                beats[k] += 1

    print("  " + "─" * 96)
    line = f"  {'TOTAL':7s}{tot['bnh']:>+12,.0f}{tot['base_pnl'] - tot['bnh']:>+12,.0f}"
    for tau in taus:
        line += f"{tot[f'pnl@{tau}'] - tot['bnh']:>+12,.0f}"
    print(line)
    line = f"  {'beats':7s}{'':>12s}{beats['base_pnl']:>9d}/{len(rows)}"
    for tau in taus:
        line += f"{beats[f'pnl@{tau}']:>9d}/{len(rows)}"
    print(line)
    print("═" * 100)
    print(f"  Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
