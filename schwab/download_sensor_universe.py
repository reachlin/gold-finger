"""
download_sensor_universe.py

Download max-history daily OHLCV for the RMT sensor universe — a broad,
sector-balanced set of large-cap US names used purely to MEASURE cross-
sectional correlation (the "propagation of chaos" / herding index in
rmt_chaos.py). This is a market-state sensor, not a tradable watchlist.

~10 names per GICS sector so no single sector dominates the correlation
bulk. Saves to data/<ticker>_history.csv in the same format as
vault8/download_bluechips.py. Skips names already on disk (the blue chips
were refreshed separately); pass --force to re-download everything.

Run:
    /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/download_sensor_universe.py
"""
import os
import sys
import time
import argparse
import pandas as pd
import yfinance as yf

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# GICS-sector-balanced. Kept as a dict for auditability; order irrelevant.
SENSOR_UNIVERSE = {
    "Info Tech":     ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE",
                      "ACN", "CSCO", "AMD", "QCOM", "TXN"],
    "Comm Svcs":     ["GOOGL", "META", "NFLX", "DIS", "VZ", "T", "CMCSA", "TMUS"],
    "Cons Disc":     ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX",
                      "BKNG", "TJX"],
    "Cons Staples":  ["PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL", "MO",
                      "PM", "KMB"],
    "Health Care":   ["UNH", "LLY", "JNJ", "ABBV", "MRK", "TMO", "ABT", "PFE",
                      "DHR", "BMY", "GILD"],
    "Financials":    ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK",
                      "SPGI", "SCHW", "CB"],
    "Industrials":   ["CAT", "HON", "BA", "GE", "UPS", "RTX", "UNP", "DE",
                      "LMT", "EMR", "ETN"],
    "Energy":        ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "OXY",
                      "WMB"],
    "Materials":     ["LIN", "SHW", "APD", "FCX", "NEM", "NUE", "ECL", "DOW"],
    "Utilities":     ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL"],
    "Real Estate":   ["AMT", "PLD", "EQIX", "CCI", "PSA", "O", "SPG", "WELL"],
}


def all_tickers() -> list[str]:
    seen, out = set(), []
    for names in SENSOR_UNIVERSE.values():
        for t in names:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def _csv_path(ticker: str) -> str:
    return os.path.join(_DATA_DIR, f"{ticker.lower().replace('-', '')}_history.csv")


def download_ticker(ticker: str) -> bool:
    out_path = _csv_path(ticker)
    try:
        hist = yf.Ticker(ticker).history(period="max", auto_adjust=True)
        if hist.empty or len(hist) < 100:
            print(f"  ✗ {ticker:<6} — no data or too short")
            return False
        hist = hist.reset_index()
        hist["datetime"] = pd.to_datetime(hist["Date"]).dt.date
        hist = hist.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })[["datetime", "open", "high", "low", "close", "volume"]]
        hist = hist[hist["close"] > 0].dropna()
        hist.to_csv(out_path, index=False)
        print(f"  ✓ {ticker:<6} {hist['datetime'].iloc[0]} → "
              f"{hist['datetime'].iloc[-1]}  ({len(hist)} days)")
        return True
    except Exception as e:
        print(f"  ✗ {ticker:<6} — {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-download even if a CSV already exists")
    args = ap.parse_args()

    os.makedirs(_DATA_DIR, exist_ok=True)
    tickers = all_tickers()
    todo = [t for t in tickers
            if args.force or not os.path.exists(_csv_path(t))]
    have = len(tickers) - len(todo)
    print(f"Sensor universe: {len(tickers)} names across "
          f"{len(SENSOR_UNIVERSE)} sectors. "
          f"{have} already on disk, downloading {len(todo)}.\n")

    ok, fail = 0, 0
    for t in todo:
        ok += download_ticker(t)
        fail += 0 if os.path.exists(_csv_path(t)) else 1
        time.sleep(0.4)   # polite rate limit
    print(f"\nDone: {ok} downloaded, {len(todo) - ok} failed, "
          f"{len(tickers)} total in universe.")


if __name__ == "__main__":
    main()
