"""
Fetch and cache earnings history (EPS estimate / reported / surprise) for
the watchlist via yfinance → data/fundamentals/{sym}_earnings.csv.

The cached files feed the fundamental features in fundamentals.py, which
lift the LGBM advisory models' holdout AUC (experiment 2026-07-05:
data/experiment_fundamentals_2026-07-05.txt). Earnings are quarterly, so
re-run this every few weeks (or after earnings season) to stay current.

Usage:
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/fetch_fundamentals.py
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/fetch_fundamentals.py NVDA KO
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import yfinance as yf

FUND_DIR = os.path.join(os.path.dirname(__file__), "..", "data",
                        "fundamentals")
WATCHLIST = ["NVDA", "AMD", "AAPL", "AMZN", "META", "MSFT", "GOOGL",
             "IBM", "INTC", "IONQ", "KO", "MMM", "XOM", "PG", "TSLA",
             "UNH", "HD", "ABT"]


def fetch(symbol: str) -> "int | None":
    ed = yf.Ticker(symbol).get_earnings_dates(limit=100)
    if ed is None or ed.empty:
        return None
    ed = ed.reset_index()
    ed.columns = ["date", "eps_est", "eps", "surprise_pct"]
    ed["date"] = ed["date"].dt.tz_localize(None)
    ed = ed.dropna(subset=["eps"]).sort_values("date")
    ed.to_csv(os.path.join(FUND_DIR, f"{symbol.lower()}_earnings.csv"),
              index=False)
    return len(ed)


def main():
    symbols = [s.upper() for s in sys.argv[1:]] or WATCHLIST
    os.makedirs(FUND_DIR, exist_ok=True)
    for sym in symbols:
        try:
            n = fetch(sym)
            print(f"  {sym:6s} {'no data' if n is None else f'{n} reports'}")
        except Exception as exc:
            print(f"  {sym:6s} FAILED ({exc})")
        time.sleep(1)


if __name__ == "__main__":
    main()
