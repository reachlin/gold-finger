"""
vault8/backtest_weekly_range.py

Theoretical-max weekly range backtest.

For each stock, group daily OHLCV into ISO calendar weeks, compute:
  - weekly_low  = min(low)  across the week
  - weekly_high = max(high) across the week

Then simulate the oracle strategy: buy at weekly_low, sell at weekly_high.
This gives the *upper bound* of what any weekly-range prediction model
could ever achieve — the target we're trying to approximate.

Outputs per stock (annualised over the data window):
  - Median weekly range %      (high-low)/low * 100
  - Win rate                   weeks where range > 1% (tradeable)
  - Oracle CAGR %              compound annual growth from perfect trades
  - Oracle total return %      raw compounded return over full period
  - Sharpe (weekly returns)    risk-adjusted quality
  - Data years                 length of history used

Run:
  /opt/miniconda3/envs/claude-sandbox/bin/python vault8/backtest_weekly_range.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Exclude pure volatility / bond indices — not tradeable the same way
EXCLUDE = {"vix", "tlt", "vym", "usmv", "vig", "nobl", "schd"}


def load_stock(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"{ticker}_history.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "date"})
    df = df.sort_values("date").reset_index(drop=True)
    # Drop rows with zero/missing OHLC
    df = df[(df["high"] > 0) & (df["low"] > 0)].copy()
    return df


def weekly_range_backtest(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["year_week"] = df["date"].dt.isocalendar().year.astype(str) + "-W" + \
                      df["date"].dt.isocalendar().week.astype(str).str.zfill(2)

    weekly = df.groupby("year_week").agg(
        week_start=("date", "min"),
        week_end=("date", "max"),
        weekly_low=("low", "min"),
        weekly_high=("high", "max"),
        open_price=("open", "first"),
        close_price=("close", "last"),
        n_days=("date", "count"),
    ).reset_index()

    # Drop partial weeks (< 3 trading days — e.g. holidays, data gaps)
    weekly = weekly[weekly["n_days"] >= 3].copy()
    weekly = weekly.sort_values("week_start").reset_index(drop=True)

    # Oracle return per week: buy at low, sell at high
    weekly["range_pct"] = (weekly["weekly_high"] - weekly["weekly_low"]) / weekly["weekly_low"] * 100
    weekly["oracle_return"] = weekly["weekly_high"] / weekly["weekly_low"]  # gross multiplier

    # Compound oracle return (assumes full reinvestment, 1 trade/week)
    total_return = weekly["oracle_return"].prod()
    n_weeks = len(weekly)
    n_years = n_weeks / 52.0
    cagr = (total_return ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    # Weekly log returns for Sharpe
    log_ret = np.log(weekly["oracle_return"])
    sharpe_weekly = (log_ret.mean() / log_ret.std() * np.sqrt(52)) if log_ret.std() > 0 else 0

    win_rate = (weekly["range_pct"] > 1.0).mean() * 100  # weeks with >1% spread

    data_start = weekly["week_start"].min()
    data_end   = weekly["week_end"].max()

    return {
        "n_weeks":          n_weeks,
        "data_years":       round(n_years, 1),
        "data_start":       str(data_start.date()),
        "data_end":         str(data_end.date()),
        "median_range_pct": round(weekly["range_pct"].median(), 2),
        "p25_range_pct":    round(weekly["range_pct"].quantile(0.25), 2),
        "p75_range_pct":    round(weekly["range_pct"].quantile(0.75), 2),
        "win_rate_pct":     round(win_rate, 1),
        "oracle_total_ret": round((total_return - 1) * 100, 0),
        "oracle_cagr_pct":  round(cagr, 1),
        "oracle_sharpe":    round(sharpe_weekly, 2),
        "weekly":           weekly,
    }


def main():
    files = glob.glob(os.path.join(DATA_DIR, "*_history.csv"))
    tickers = sorted(
        os.path.basename(f).replace("_history.csv", "") for f in files
    )
    tickers = [t for t in tickers if t not in EXCLUDE]

    results = []
    for ticker in tickers:
        df = load_stock(ticker)
        if df is None or len(df) < 52:
            continue
        stats = weekly_range_backtest(df)
        stats["ticker"] = ticker.upper()
        results.append(stats)
        print(f"  {ticker.upper():<6}  {stats['n_weeks']:>4}wk  "
              f"range {stats['median_range_pct']:>5.1f}%  "
              f"win {stats['win_rate_pct']:>4.0f}%  "
              f"CAGR {stats['oracle_cagr_pct']:>7,.0f}%  "
              f"Sharpe {stats['oracle_sharpe']:>5.2f}")

    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k != "weekly"}
        for r in results
    ])

    summary = summary.sort_values("oracle_cagr_pct", ascending=False)

    print("\n" + "=" * 95)
    print(f"{'VAULT 8 — WEEKLY RANGE ORACLE BACKTEST':^95}")
    print(f"{'(Perfect buy-at-low / sell-at-high each week)':^95}")
    print("=" * 95)
    print(f"{'Ticker':<8} {'Weeks':>6} {'Yrs':>4}  "
          f"{'Med Range':>10} {'P25':>6} {'P75':>6}  "
          f"{'Win%':>5}  {'CAGR%':>10}  {'Sharpe':>7}")
    print("-" * 95)
    for _, row in summary.iterrows():
        print(f"{row['ticker']:<8} {row['n_weeks']:>6} {row['data_years']:>4.1f}  "
              f"{row['median_range_pct']:>9.1f}%  "
              f"{row['p25_range_pct']:>5.1f}% {row['p75_range_pct']:>5.1f}%  "
              f"{row['win_rate_pct']:>4.0f}%  "
              f"{row['oracle_cagr_pct']:>9,.0f}%  "
              f"{row['oracle_sharpe']:>7.2f}")
    print("=" * 95)

    print("\nTop 10 by median weekly range % (most volatile = most opportunity):")
    top10 = summary.nlargest(10, "median_range_pct")
    for _, row in top10.iterrows():
        print(f"  {row['ticker']:<6}  med range {row['median_range_pct']:>5.1f}%  "
              f"win rate {row['win_rate_pct']:.0f}%  "
              f"CAGR {row['oracle_cagr_pct']:,.0f}%")

    print("\nTop 10 by Sharpe (best risk-adjusted weekly range capture):")
    top10s = summary.nlargest(10, "oracle_sharpe")
    for _, row in top10s.iterrows():
        print(f"  {row['ticker']:<6}  Sharpe {row['oracle_sharpe']:>5.2f}  "
              f"med range {row['median_range_pct']:>5.1f}%  "
              f"CAGR {row['oracle_cagr_pct']:,.0f}%")

    out_path = os.path.join(os.path.dirname(__file__), "weekly_range_backtest.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nFull results saved → {out_path}")


if __name__ == "__main__":
    main()
