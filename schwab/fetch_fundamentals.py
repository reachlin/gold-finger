"""
Fetch and cache earnings history (EPS estimate / reported / surprise) for
the watchlist via yfinance → data/fundamentals/{sym}_earnings.csv.

The cached files feed the fundamental features in fundamentals.py, which
lift the LGBM advisory models' holdout AUC (experiment 2026-07-05:
data/experiment_fundamentals_2026-07-05.txt). Earnings are quarterly, so
re-run this every few weeks (or after earnings season) to stay current.

Usage:
  /Users/lincai/anaconda3/envs/gold-finger-py312/bin/python schwab/fetch_fundamentals.py
  /Users/lincai/anaconda3/envs/gold-finger-py312/bin/python schwab/fetch_fundamentals.py NVDA KO
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


def split_reports(ed: "pd.DataFrame", today=None) -> "tuple":
    """
    Split yfinance's earnings-date frame into (reported, upcoming).

    A row with no reported EPS is a scheduled future print. Those rows used
    to be dropped on the floor, which is why no signal ever carried an
    earnings date — see earnings_calendar.py. `reported` keeps the LGBM
    fundamental features unchanged; `upcoming` is the forward calendar the
    SELL_PUT gate reads.
    """
    import pandas as pd

    today = today or pd.Timestamp.now().normalize()
    ed = ed.reset_index()
    ed.columns = ["date", "eps_est", "eps", "surprise_pct"]
    ed["date"] = ed["date"].dt.tz_localize(None)
    ed = ed.sort_values("date")

    reported = ed.dropna(subset=["eps"])
    # Trust "no EPS yet" over the clock: a print lands in the evening and the
    # feed can lag, so a dated-today row with no number is still upcoming.
    upcoming = ed[ed["eps"].isna() & (ed["date"] >= today)]
    return reported, upcoming[["date", "eps_est"]]


def fetch(symbol: str) -> "int | None":
    import pandas as pd

    ed = yf.Ticker(symbol).get_earnings_dates(limit=100)
    if ed is None or ed.empty:
        return None
    reported, upcoming = split_reports(ed)

    reported.to_csv(os.path.join(FUND_DIR, f"{symbol.lower()}_earnings.csv"),
                    index=False)

    # Stamp the scrape date — earnings_calendar stops trusting a cache older
    # than CACHE_MAX_AGE_DAYS, because a scheduled date can be moved.
    upcoming = upcoming.copy()
    upcoming["fetched_at"] = str(pd.Timestamp.now().date())
    upcoming.to_csv(
        os.path.join(FUND_DIR, f"{symbol.lower()}_earnings_upcoming.csv"),
        index=False)

    return len(reported)


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
