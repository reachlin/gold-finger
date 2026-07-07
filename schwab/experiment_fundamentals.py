"""
Fundamental-features experiment — does adding earnings-derived features
lift the LGBM models above their measured technical-only AUC ~0.5?

For each watchlist symbol and each direction (down = assignment risk,
up = called-away / router label), fit AssignmentRiskModel twice — technical
features only vs technical + fundamentals (fundamentals.py, no lookahead) —
and compare the chronological 80/20 holdout AUC that fit() already probes.

Borrowed hypothesis (WorldQuant alpha playbook, measured pass rates):
fundamental 40% > mixed 12.7% > pure technical 5.3%.

Usage:
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/experiment_fundamentals.py
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


def _auc(df, direction, earnings) -> "float | None":
    try:
        m = AssignmentRiskModel(direction=direction)
        m.fit(df, earnings=earnings)
        return m.holdout_auc
    except Exception:
        return None


def main():
    t0 = time.time()
    print("═" * 86)
    print("  FUNDAMENTAL FEATURES EXPERIMENT — holdout AUC, technical-only vs +fundamentals")
    print("═" * 86)
    print(f"  {'Symbol':7s}{'down base':>11s}{'down +fund':>12s}{'Δ':>7s}"
          f"{'up base':>11s}{'up +fund':>12s}{'Δ':>7s}")
    print("  " + "─" * 80)

    cols = {"down_base": [], "down_fund": [], "up_base": [], "up_fund": []}
    for sym in WATCHLIST:
        path = os.path.join(DATA_DIR, f"{sym.lower()}_history.csv")
        if not os.path.exists(path):
            continue
        df       = pd.read_csv(path)
        earnings = fundamentals.load_earnings(sym)
        row = {}
        for direction in ("down", "up"):
            row[f"{direction}_base"] = _auc(df, direction, None)
            # earnings=None would silently reproduce the baseline — show a
            # dash instead (this exact trap hid a wrong cache path once)
            row[f"{direction}_fund"] = (_auc(df, direction, earnings)
                                        if earnings is not None else None)
        for k, v in row.items():
            if v is not None:
                cols[k].append(v)

        def f(v):
            return f"{v:.3f}" if v is not None else "  —  "

        def d(a, b):
            return (f"{b - a:+.3f}" if a is not None and b is not None
                    else "  —  ")

        print(f"  {sym:7s}{f(row['down_base']):>11s}{f(row['down_fund']):>12s}"
              f"{d(row['down_base'], row['down_fund']):>7s}"
              f"{f(row['up_base']):>11s}{f(row['up_fund']):>12s}"
              f"{d(row['up_base'], row['up_fund']):>7s}")

    print("  " + "─" * 80)
    med = {k: (float(np.median(v)) if v else float("nan"))
           for k, v in cols.items()}
    print(f"  {'MEDIAN':7s}{med['down_base']:>11.3f}{med['down_fund']:>12.3f}"
          f"{med['down_fund'] - med['down_base']:>+7.3f}"
          f"{med['up_base']:>11.3f}{med['up_fund']:>12.3f}"
          f"{med['up_fund'] - med['up_base']:>+7.3f}")
    print("═" * 86)
    print(f"  Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
