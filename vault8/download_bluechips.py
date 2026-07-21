"""
vault8/download_bluechips.py

Download max-history daily OHLCV for blue-chip US stocks via yfinance.
Saves to data/<ticker>_history.csv in the same format as existing files.

Blue chip universe:
  - Dow Jones 30 components
  - S&P 500 mega-caps (FAANG/MAMAA + semiconductors)
  - A few liquid sector leaders

Run:
  /Users/lincai/anaconda3/envs/gold-finger/bin/python vault8/download_bluechips.py
"""

import os
import time
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

BLUE_CHIPS = {
    # --- Dow Jones 30 ---
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "UNH":   "UnitedHealth",
    "GS":    "Goldman Sachs",
    "HD":    "Home Depot",
    "MCD":   "McDonald's",
    "CAT":   "Caterpillar",
    "CRM":   "Salesforce",
    "V":     "Visa",
    "AMGN":  "Amgen",
    "HON":   "Honeywell",
    "AXP":   "AmEx",
    "TRV":   "Travelers",
    "JPM":   "JPMorgan",
    "IBM":   "IBM",
    "JNJ":   "J&J",
    "WMT":   "Walmart",
    "PG":    "P&G",
    "CVX":   "Chevron",
    "MRK":   "Merck",
    "DIS":   "Disney",
    "NKE":   "Nike",
    "MMM":   "3M",
    "KO":    "Coca-Cola",
    "BA":    "Boeing",
    "CSCO":  "Cisco",
    "VZ":    "Verizon",
    "INTC":  "Intel",
    "DOW":   "Dow Inc",

    # --- Mega-cap tech / MAMAA ---
    "NVDA":  "NVIDIA",
    "GOOGL": "Alphabet",
    "AMZN":  "Amazon",
    "META":  "Meta",
    "TSLA":  "Tesla",
    "AMD":   "AMD",
    "AVGO":  "Broadcom",
    "QCOM":  "Qualcomm",
    "TXN":   "Texas Instruments",

    # --- Healthcare blue chips ---
    "LLY":   "Eli Lilly",
    "ABT":   "Abbott",
    "TMO":   "Thermo Fisher",
    "ABBV":  "AbbVie",

    # --- Financials ---
    "BRK-B": "Berkshire B",
    "BAC":   "Bank of America",
    "WFC":   "Wells Fargo",

    # --- Consumer / retail ---
    "COST":  "Costco",
    "XOM":   "ExxonMobil",

    # --- Sector ETFs (liquid, long history) ---
    "SPY":   "S&P 500 ETF",
    "QQQ":   "Nasdaq 100 ETF",
    "GLD":   "Gold ETF",
}


def download_ticker(ticker: str, name: str) -> bool:
    out_path = os.path.join(DATA_DIR, f"{ticker.lower().replace('-','')}_history.csv")
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="max", auto_adjust=True)
        if hist.empty or len(hist) < 100:
            print(f"  ✗ {ticker:<6} ({name}) — no data or too short")
            return False

        hist = hist.reset_index()
        hist["datetime"] = pd.to_datetime(hist["Date"]).dt.date
        hist = hist.rename(columns={
            "Open": "open", "High": "high",
            "Low":  "low",  "Close": "close",
            "Volume": "volume",
        })[["datetime", "open", "high", "low", "close", "volume"]]
        hist = hist[hist["close"] > 0].dropna()

        hist.to_csv(out_path, index=False)
        start = hist["datetime"].iloc[0]
        end   = hist["datetime"].iloc[-1]
        print(f"  ✓ {ticker:<6} ({name:<20}) {start} → {end}  ({len(hist)} days)")
        return True

    except Exception as e:
        print(f"  ✗ {ticker:<6} ({name}) — {e}")
        return False


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading {len(BLUE_CHIPS)} blue-chip US stocks (max history)...\n")

    ok, fail = 0, 0
    for ticker, name in BLUE_CHIPS.items():
        success = download_ticker(ticker, name)
        if success:
            ok += 1
        else:
            fail += 1
        time.sleep(0.4)   # polite rate limit

    print(f"\nDone: {ok} downloaded, {fail} failed.")


if __name__ == "__main__":
    main()
