"""
Earnings-derived fundamental features for the LGBM models.

Motivated by the WorldQuant alpha-mining playbook's measured pass rates —
fundamental 40% > mixed 12.7% > pure technical 5.3% — and by our own
measurement that the pure-technical assignment-risk models sit at holdout
AUC ~0.5. These features add the fundamental/analyst dimension for our 18
symbols from cached yfinance earnings history
(data/fundamentals/{sym}_earnings.csv, fetched 2026-07-05, 2001→2026).

Features (per daily bar, as-of joined with NO lookahead — a report dated D
becomes visible strictly after D):
  earnings_yield  EPS_ttm / close      (E/P — the operating_income/equity
                                        spirit, computable from EPS history)
  eps_growth      EPS_ttm vs EPS_ttm one year earlier − 1
  surprise_last   most recent quarterly surprise %
  surprise_mean4  mean surprise % of the last 4 quarters
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

FUND_FEATURES = ["earnings_yield", "eps_growth",
                 "surprise_last", "surprise_mean4"]

FUND_DIR = os.path.join(os.path.dirname(__file__), "..", "data",
                        "fundamentals")


def load_earnings(symbol: str, fund_dir: str = FUND_DIR) -> "pd.DataFrame | None":
    """Cached earnings history (date, eps_est, eps, surprise_pct) or None."""
    path = os.path.join(fund_dir, f"{symbol.lower()}_earnings.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_fundamental_features(bars: pd.DataFrame,
                               earnings: pd.DataFrame) -> pd.DataFrame:
    """
    One row per bar, aligned to bars.index. NaN until enough reports have
    been PUBLISHED (strictly before the bar date — no lookahead).
    """
    out = pd.DataFrame(index=bars.index, columns=FUND_FEATURES, dtype=float)
    if earnings is None or len(earnings) == 0:
        return out

    e = earnings.sort_values("date").reset_index(drop=True)
    eps      = e["eps"].to_numpy(dtype=float)
    surprise = e["surprise_pct"].to_numpy(dtype=float)
    # report becomes visible strictly AFTER its date
    visible_after = pd.to_datetime(e["date"]).dt.date.to_numpy()

    bar_dates = pd.to_datetime(bars["datetime"]).dt.date.to_numpy()
    closes    = bars["close"].to_numpy(dtype=float)

    # k[i] = number of reports visible at bar i (report date < bar date)
    k = np.searchsorted(visible_after, bar_dates, side="left")

    for i in range(len(bars)):
        n = k[i]
        if n >= 4:
            ttm = eps[n - 4:n].sum()
            if closes[i] > 0:
                out.iat[i, 0] = ttm / closes[i]                # earnings_yield
            if n >= 8:
                prev_ttm = eps[n - 8:n - 4].sum()
                if abs(prev_ttm) > 1e-9:
                    out.iat[i, 1] = ttm / prev_ttm - 1         # eps_growth
        if n >= 1:
            out.iat[i, 2] = surprise[n - 1]                    # surprise_last
        if n >= 4:
            out.iat[i, 3] = float(np.nanmean(surprise[n - 4:n]))  # surprise_mean4
    return out
