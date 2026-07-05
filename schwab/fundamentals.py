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

# Analyst features from yfinance upgrades_downgrades (dated actions with
# price targets, ~2012→now) — the free, backtestable alternative to
# Seeking Alpha's unofficial scrapers (assessed 2026-07-05)
ANALYST_FEATURES = ["net_upgrades_90d", "pt_revisions_90d", "pt_gap"]

FUND_DIR = os.path.join(os.path.dirname(__file__), "..", "data",
                        "fundamentals")


def load_earnings(symbol: str, fund_dir: str = FUND_DIR) -> "pd.DataFrame | None":
    """Cached earnings history (date, eps_est, eps, surprise_pct) or None."""
    path = os.path.join(fund_dir, f"{symbol.lower()}_earnings.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_analyst(symbol: str, fund_dir: str = FUND_DIR) -> "pd.DataFrame | None":
    """Cached analyst actions (gradedate, action, pricetargetaction,
    currentpricetarget, ...) or None."""
    path = os.path.join(fund_dir, f"{symbol.lower()}_analyst.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["gradedate"])
    return df.sort_values("gradedate").reset_index(drop=True)


ANALYST_MAX_AGE_DAYS = 90    # newest action must be this recent
ANALYST_MIN_ACTIONS  = 100   # and the feed must have real coverage


def analyst_is_fresh(analyst: "pd.DataFrame | None", bars: pd.DataFrame,
                     max_age_days: int = ANALYST_MAX_AGE_DAYS,
                     min_actions: int = ANALYST_MIN_ACTIONS) -> bool:
    """
    Data-quality gate (experiment 2026-07-05): stale/sparse analyst feeds
    actively poison the model — IONQ's feed (29 actions, dead since
    2024-08) collapsed its up-model AUC 0.694 → 0.308. Models retrain
    daily, so a tight gate drops a dying feed quickly.
    """
    if analyst is None or len(analyst) < min_actions:
        return False
    last_action = pd.to_datetime(analyst["gradedate"]).max()
    last_bar    = pd.to_datetime(bars["datetime"]).max()
    return (last_bar - last_action).days <= max_age_days


def build_analyst_features(bars: pd.DataFrame,
                           analyst: pd.DataFrame) -> pd.DataFrame:
    """
    Per-bar analyst features, as-of joined — an action dated D is visible
    strictly after D:
      net_upgrades_90d   #upgrades − #downgrades, trailing 90 calendar days
      pt_revisions_90d   #PT-raises − #PT-lowers, trailing 90 calendar days
      pt_gap             trailing-180d median price target / close − 1
    Counts are 0 with no visible actions; pt_gap is NaN until a PT exists.
    """
    out = pd.DataFrame(index=bars.index, columns=ANALYST_FEATURES,
                       dtype=float)
    out[["net_upgrades_90d", "pt_revisions_90d"]] = 0.0
    if analyst is None or len(analyst) == 0:
        out["pt_gap"] = np.nan
        return out

    a = analyst.sort_values("gradedate").reset_index(drop=True)
    act_dates = pd.to_datetime(a["gradedate"]).dt.normalize().to_numpy()
    action    = a.get("action", pd.Series([""] * len(a))).astype(str)
    pt_action = a.get("pricetargetaction",
                      pd.Series([""] * len(a))).astype(str)
    up   = np.cumsum(np.where(action.str.lower() == "up", 1, 0))
    down = np.cumsum(np.where(action.str.lower() == "down", 1, 0))
    rai  = np.cumsum(np.where(pt_action.str.lower() == "raises", 1, 0))
    low  = np.cumsum(np.where(pt_action.str.lower() == "lowers", 1, 0))
    pts  = a.get("currentpricetarget",
                 pd.Series([np.nan] * len(a))).to_numpy(dtype=float)

    def _csum(arr, hi, lo):
        """Sum of arr counts over action index range (lo, hi]."""
        return (arr[hi - 1] if hi > 0 else 0) - (arr[lo - 1] if lo > 0 else 0)

    bar_dates = pd.to_datetime(bars["datetime"]).dt.normalize().to_numpy()
    closes    = bars["close"].to_numpy(dtype=float)
    d90  = np.timedelta64(90, "D")
    d180 = np.timedelta64(180, "D")

    for i in range(len(bars)):
        hi   = int(np.searchsorted(act_dates, bar_dates[i], side="left"))
        lo90 = int(np.searchsorted(act_dates, bar_dates[i] - d90,
                                   side="left"))
        out.iat[i, 0] = _csum(up, hi, lo90) - _csum(down, hi, lo90)
        out.iat[i, 1] = _csum(rai, hi, lo90) - _csum(low, hi, lo90)
        lo180 = int(np.searchsorted(act_dates, bar_dates[i] - d180,
                                    side="left"))
        window_pts = pts[lo180:hi]
        window_pts = window_pts[~np.isnan(window_pts)]
        if len(window_pts) and closes[i] > 0:
            out.iat[i, 2] = float(np.median(window_pts)) / closes[i] - 1
    return out


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
