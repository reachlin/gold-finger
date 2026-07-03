"""
Compare the new live-scanner wheel behavior against the previous version.

  previous — what the live scanner effectively did before wheel completion:
             sell puts, hold to expiry, liquidate assignments at the expiry
             close. No covered calls, no early exit.
  new      — full wheel as now implemented in options_ledger.py: adaptive
             profit-target early exits, assignment → covered calls →
             called away.

Both arms use identical put entries (same Scavenger scan, regime gates,
fast risk-off), so the delta is purely the position-management change.

Usage:
  /opt/miniconda3/envs/trader/bin/python schwab/compare_wheel_versions.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from backtest_scavenger import (
    walk_forward_scavenger, compute_indicators, MIN_HISTORY, DTE_BARS, SHARES,
)

WATCHLIST = [
    "NVDA", "AMD", "AAPL", "AMZN",
    "META", "MSFT", "GOOGL",
    "IBM", "INTC", "IONQ", "KO",
    "MMM", "XOM", "PG", "TSLA",
    "UNH", "HD", "ABT",
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    spy_path = os.path.join(DATA_DIR, "spy_history.csv")
    vix_path = os.path.join(DATA_DIR, "vix_history.csv")
    spy_df = pd.read_csv(spy_path, parse_dates=["datetime"]) if os.path.exists(spy_path) else None
    vix_df = pd.read_csv(vix_path, parse_dates=["datetime"]) if os.path.exists(vix_path) else None

    results = []
    for symbol in WATCHLIST:
        path = os.path.join(DATA_DIR, f"{symbol.lower()}_history.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["datetime"])
        if len(df) < MIN_HISTORY + DTE_BARS + 10:
            continue

        prev_events = walk_forward_scavenger(df, symbol, spy_df=spy_df,
                                             vix_df=vix_df,
                                             early_exit=False, sell_calls=False)
        new_events  = walk_forward_scavenger(df, symbol, spy_df=spy_df,
                                             vix_df=vix_df,
                                             early_exit=True, sell_calls=True)
        prev_pnl = sum(e["pnl"] for e in prev_events)
        new_pnl  = sum(e["pnl"] for e in new_events)
        results.append({
            "symbol":      symbol,
            "prev_events": len(prev_events),
            "prev_pnl":    prev_pnl,
            "new_events":  len(new_events),
            "new_pnl":     new_pnl,
            "delta":       new_pnl - prev_pnl,
        })

    if not results:
        print("No history data found under data/*_history.csv")
        return

    print(f"\n{'='*72}")
    print("  WHEEL BACKTEST — previous (puts-only, hold to expiry) vs new (full wheel)")
    print(f"{'='*72}")
    print(f"  {'Symbol':<8} {'prev ev':>7} {'prev P&L':>11} {'new ev':>7} "
          f"{'new P&L':>11} {'delta':>11}")
    print(f"  {'─'*8} {'─'*7} {'─'*11} {'─'*7} {'─'*11} {'─'*11}")
    for r in sorted(results, key=lambda x: x["delta"], reverse=True):
        print(f"  {r['symbol']:<8} {r['prev_events']:>7} "
              f"${r['prev_pnl']:>+9.0f} {r['new_events']:>7} "
              f"${r['new_pnl']:>+9.0f} ${r['delta']:>+9.0f}")
    tp = sum(r["prev_pnl"] for r in results)
    tn = sum(r["new_pnl"] for r in results)
    print(f"  {'─'*8} {'─'*7} {'─'*11} {'─'*7} {'─'*11} {'─'*11}")
    print(f"  {'TOTAL':<8} {'':>7} ${tp:>+9.0f} {'':>7} ${tn:>+9.0f} ${tn-tp:>+9.0f}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
