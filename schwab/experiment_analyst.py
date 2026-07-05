"""
Analyst-features experiment — do upgrade/downgrade and price-target
features lift the LGBM models above the DEPLOYED baseline (technical +
fundamentals, medians 0.546 down / 0.570 up as of 2026-07-05)?

Features (fundamentals.py, as-of joined, no lookahead): net_upgrades_90d,
pt_revisions_90d, pt_gap — from yfinance upgrades_downgrades, the free
backtestable alternative to Seeking Alpha's unofficial scrapers.

Usage:
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/experiment_analyst.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

import fundamentals
from assignment_risk import AssignmentRiskModel

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
WATCHLIST = ["NVDA", "AMD", "AAPL", "AMZN", "META", "MSFT", "GOOGL",
             "IBM", "INTC", "IONQ", "KO", "MMM", "XOM", "PG", "TSLA",
             "UNH", "HD", "ABT"]


def _auc(df, direction, earnings, analyst) -> "float | None":
    try:
        m = AssignmentRiskModel(direction=direction)
        m.fit(df, earnings=earnings, analyst=analyst)
        return m.holdout_auc
    except Exception:
        return None


def main():
    t0 = time.time()
    print("═" * 86)
    print("  ANALYST FEATURES EXPERIMENT — holdout AUC,"
          " deployed baseline (tech+fund) vs +analyst")
    print("═" * 86)
    print(f"  {'Symbol':7s}{'down base':>11s}{'down +ana':>11s}{'Δ':>8s}"
          f"{'up base':>11s}{'up +ana':>11s}{'Δ':>8s}")
    print("  " + "─" * 80)

    cols = {"down_base": [], "down_ana": [], "up_base": [], "up_ana": []}
    for sym in WATCHLIST:
        path = os.path.join(DATA_DIR, f"{sym.lower()}_history.csv")
        if not os.path.exists(path):
            continue
        df       = pd.read_csv(path)
        earnings = fundamentals.load_earnings(sym)
        analyst  = fundamentals.load_analyst(sym)
        row = {}
        for direction in ("down", "up"):
            row[f"{direction}_base"] = _auc(df, direction, earnings, None)
            row[f"{direction}_ana"]  = (_auc(df, direction, earnings, analyst)
                                        if analyst is not None else None)
        for k, v in row.items():
            if v is not None:
                cols[k].append(v)

        def f(v):
            return f"{v:.3f}" if v is not None else "  —  "

        def d(a, b):
            return (f"{b - a:+.3f}" if a is not None and b is not None
                    else "  —  ")

        print(f"  {sym:7s}{f(row['down_base']):>11s}{f(row['down_ana']):>11s}"
              f"{d(row['down_base'], row['down_ana']):>8s}"
              f"{f(row['up_base']):>11s}{f(row['up_ana']):>11s}"
              f"{d(row['up_base'], row['up_ana']):>8s}")

    print("  " + "─" * 80)
    med = {k: (float(np.median(v)) if v else float("nan"))
           for k, v in cols.items()}
    print(f"  {'MEDIAN':7s}{med['down_base']:>11.3f}{med['down_ana']:>11.3f}"
          f"{med['down_ana'] - med['down_base']:>+8.3f}"
          f"{med['up_base']:>11.3f}{med['up_ana']:>11.3f}"
          f"{med['up_ana'] - med['up_base']:>+8.3f}")
    print("═" * 86)
    print(f"  Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
