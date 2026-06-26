"""
Daily pullback-in-trend signal scanner.

Fetches data for a watchlist from Schwab, runs trend_scanner on each,
prints a summary table, and sends a Slack notification for any BUY signals.

Usage:
    python schwab/daily_signal.py
"""
import os
import sys
import json
import urllib.request
from datetime import datetime, timedelta
from dotenv import load_dotenv
import schwab
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

CLIENT_ID     = os.environ["SCHWAB_CLIENT_ID"]
CLIENT_SECRET = os.environ["SCHWAB_CLIENT_SECRET"]
REDIRECT_URI  = "https://127.0.0.1"
TOKEN_PATH    = os.path.join(os.path.dirname(__file__), "schwab_token.json")
SLACK_WEBHOOK = os.environ.get("SLACK_WEBHOOK_URL", os.environ.get("SLACK_WEB_HOOK", ""))

WATCHLIST = [
    "NVDA", "AMD", "TSLA", "AAPL", "AMZN",
    "META", "MSFT", "GOOGL", "TQQQ", "SOXL",
]


def get_client():
    return schwab.auth.client_from_token_file(TOKEN_PATH, CLIENT_ID, CLIENT_SECRET)


def fetch_history(client, symbol, days=200):
    end   = datetime.now()
    start = end - timedelta(days=days)
    resp  = client.get_price_history_every_day(symbol, start_datetime=start, end_datetime=end)
    resp.raise_for_status()
    candles = resp.json().get("candles", [])
    if not candles:
        return None
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    df["volume"]   = df["volume"].astype(float)
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def send_slack(signals):
    if not SLACK_WEBHOOK:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"*Pullback Scanner — {today}*\n"]
    for s in signals:
        lines.append(
            f":green_circle: *{s['symbol']}* @ ${s['entry']}\n"
            f"  Target: ${s['target']} (+20%)  |  Stop: ${s['stop']} (-8%)  |  R/R: {s['risk_reward']}\n"
            f"  RSI: {s['rsi']}  ADX: {s['adx']}  |  {s['reason']}"
        )
    if not signals:
        lines.append("No BUY signals today.")
    payload = json.dumps({"text": "\n".join(lines)}).encode()
    req = urllib.request.Request(
        SLACK_WEBHOOK, data=payload,
        headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req)
    print("Slack notification sent.")


def main():
    from trend_scanner import compute_indicators, scan_symbol

    client = get_client()
    today  = datetime.now().strftime("%Y-%m-%d")
    print(f"\nPullback-in-Trend Scanner  [{today}]")
    print("=" * 68)

    results = []
    for symbol in WATCHLIST:
        try:
            df = fetch_history(client, symbol)
            if df is None or len(df) < 60:
                print(f"  {symbol:<6} — insufficient data")
                continue
            df_ind = compute_indicators(df).dropna().reset_index(drop=True)
            result = scan_symbol(symbol, df_ind)
            results.append(result)
            signal = result["signal"]
            if signal == "BUY":
                print(
                    f"  {symbol:<6} BUY  entry=${result['entry']}  "
                    f"target=${result['target']}  stop=${result['stop']}  "
                    f"RSI={result['rsi']}  ADX={result['adx']}"
                )
            else:
                print(f"  {symbol:<6} —    ({result['reason']})")
        except Exception as e:
            print(f"  {symbol:<6} ERROR: {e}")

    buy_signals = [r for r in results if r["signal"] == "BUY"]
    print("=" * 68)
    print(f"BUY signals: {len(buy_signals)} / {len(results)} scanned")

    if buy_signals:
        print("\n--- Action Plan ---")
        for s in buy_signals:
            risk_per_share = s["entry"] - s["stop"]
            shares = max(1, int(600 / s["entry"]))   # ~$600 budget per trade
            max_loss = round(shares * risk_per_share, 2)
            print(f"  {s['symbol']}: buy {shares} shares @ ${s['entry']}")
            print(f"    Target ${s['target']}  |  Stop ${s['stop']}  |  Max loss ${max_loss}")

    send_slack(buy_signals)


if __name__ == "__main__":
    main()
