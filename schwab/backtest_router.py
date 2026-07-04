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

Predictors:
  lgbm    — walk-forward LGBM P(>8% rally in 30d); thresholds are probabilities
  timesfm — per-bar zero-shot TimesFM 30d SMA5 forecast; thresholds are %
            (forecast series cached in data/router_cache/, resumable)

Usage:
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/backtest_router.py
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/backtest_router.py --predictor timesfm
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/backtest_router.py AMD GOOGL
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from trend_scanner import compute_indicators
from backtest_scavenger import walk_forward_scavenger, MIN_HISTORY, DTE_BARS
import wheel_router
from run_backtest import WATCHLIST, _load_df, _bnh_pnl

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_DIR  = os.path.join(DATA_DIR, "router_cache")
THRESHOLDS = {
    "lgbm":    (0.50, 0.60, 0.70),   # probability
    "timesfm": (2.00, 4.00, 6.00),   # forecast % over 30 trading days
}

_tfm_forecast_fn = None   # TimesFM loaded once, on first uncached symbol


def _event_i(e: dict) -> int:
    """Bar index an event's P&L is realized at — the latest *_i it carries."""
    return max((v for k, v in e.items()
                if k.endswith("_i") and isinstance(v, (int, np.integer))),
               default=0)


def _filter_since(events: list, dates: "np.ndarray", since) -> list:
    """Keep events whose realization date is >= since."""
    last = len(dates) - 1
    return [e for e in events if dates[min(_event_i(e), last)] >= since]


def _predictor_series(symbol: str, ind: pd.DataFrame,
                      predictor: str) -> np.ndarray:
    if predictor == "lgbm":
        return wheel_router.walk_forward_probs(ind)

    # timesfm — disk-cached: ~2.6k zero-shot forecasts per symbol
    global _tfm_forecast_fn
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{symbol.lower()}_tfm_{len(ind)}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    if _tfm_forecast_fn is None:
        from timesfm_advisor import _load_forecast_fn
        _tfm_forecast_fn = _load_forecast_fn()
    series = wheel_router.walk_forward_timesfm(ind,
                                               forecast_fn=_tfm_forecast_fn)
    np.save(cache_path, series)
    return series


def run_symbol(symbol: str, spy_df, vix_df, predictor: str,
               thresholds: tuple, since=None) -> "dict | None":
    df = _load_df(symbol)
    if df is None:
        return None

    # Same post-dropna frame the scavenger backtest iterates — series align 1:1
    ind   = compute_indicators(df).dropna().reset_index(drop=True)
    t0    = time.time()
    probs = _predictor_series(symbol, ind, predictor)
    if time.time() - t0 > 5:
        print(f"    predictor series in {time.time() - t0:.0f}s", flush=True)

    # --since: simulate on full history (warm indicators/positions), but
    # attribute P&L only to events realized inside the window, and match
    # B&H to the same window.
    dates = pd.to_datetime(ind["datetime"]).dt.date.to_numpy()
    if since is not None:
        if dates[-1] < since:
            return None
        bnh_start = max(int(np.argmax(dates >= since)), MIN_HISTORY)
        bnh = (float(ind.iloc[-1]["close"])
               - float(ind.iloc[bnh_start]["close"])) * 100
    else:
        bnh = _bnh_pnl(df)

    def _pnl(events):
        if since is not None:
            events = _filter_since(events, dates, since)
        return events

    base = _pnl(walk_forward_scavenger(df, symbol, spy_df, vix_df))
    row  = {
        "symbol":   symbol,
        "bnh":      bnh,
        "base_pnl": sum(e["pnl"] for e in base),
    }
    for tau in thresholds:
        events = _pnl(walk_forward_scavenger(df, symbol, spy_df, vix_df,
                                             router_probs=probs,
                                             router_threshold=tau))
        holds  = [e for e in events if e["event"] == "router_hold_exit"]
        row[f"pnl@{tau}"]   = sum(e["pnl"] for e in events)
        row[f"holds@{tau}"] = len(holds)
        row[f"hpnl@{tau}"]  = sum(e["pnl"] for e in holds)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", help="default: full watchlist")
    ap.add_argument("--predictor", choices=("lgbm", "timesfm"),
                    default="lgbm")
    ap.add_argument("--thresholds", type=str, default=None,
                    help="comma-separated, e.g. 3.0,4.0,5.0")
    ap.add_argument("--since", type=str, default=None,
                    help="attribute P&L to events realized on/after this "
                         "date (YYYY-MM-DD); B&H measured over the same window")
    args = ap.parse_args()

    symbols    = [s.upper() for s in args.symbols] or WATCHLIST
    thresholds = (tuple(float(t) for t in args.thresholds.split(","))
                  if args.thresholds else THRESHOLDS[args.predictor])
    since      = (pd.to_datetime(args.since).date()
                  if args.since else None)

    spy_path = os.path.join(DATA_DIR, "spy_history.csv")
    vix_path = os.path.join(DATA_DIR, "vix_history.csv")
    spy_df = pd.read_csv(spy_path, parse_dates=["datetime"]) if os.path.exists(spy_path) else None
    vix_df = pd.read_csv(vix_path, parse_dates=["datetime"]) if os.path.exists(vix_path) else None

    t0   = time.time()
    rows = []
    for sym in symbols:
        print(f"  {sym}...", flush=True)
        row = run_symbol(sym, spy_df, vix_df, args.predictor, thresholds,
                         since=since)
        if row is not None:
            rows.append(row)

    # ── Report ────────────────────────────────────────────────────────────
    taus = thresholds
    unit = "P(>8% rally in 30d)" if args.predictor == "lgbm" \
        else "TimesFM 30d SMA5 forecast %"
    window = f"  —  window: {since} → end" if since else ""
    print()
    print("═" * 100)
    print(f"  WHEEL-vs-HOLD ROUTER  —  predictor: {args.predictor.upper()}  "
          f"—  Scavenger book only, edge vs B&H{window}")
    print(f"  Router: hold shares uncapped while walk-forward {unit} >= threshold")
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
