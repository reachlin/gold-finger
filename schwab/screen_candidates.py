"""
Screen US stocks for daily ML-driven trading suitability.
Ranks by: ATR%, volume, options liquidity, affordability on a ~$2400 account.
"""
import os
import sys
import schwab
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

CLIENT_ID = os.environ["SCHWAB_CLIENT_ID"]
CLIENT_SECRET = os.environ["SCHWAB_CLIENT_SECRET"]
REDIRECT_URI = "https://127.0.0.1"
TOKEN_PATH = os.path.join(os.path.dirname(__file__), "schwab_token.json")

ACCOUNT_SIZE = 2400

CANDIDATES = [
    "NVDA", "AMD", "TSLA", "GOOGL", "AMZN",
    "AAPL", "META", "MSFT", "TQQQ", "SOXL",
]


def get_client():
    return schwab.auth.client_from_token_file(TOKEN_PATH, CLIENT_ID, CLIENT_SECRET)


def fetch_quotes(client, symbols):
    resp = client.get_quotes(symbols)
    resp.raise_for_status()
    return resp.json()


def fetch_price_history(client, symbol):
    from datetime import datetime, timedelta
    end = datetime.now()
    start = end - timedelta(days=365)
    resp = client.get_price_history_every_day(symbol, start_datetime=start, end_datetime=end)
    resp.raise_for_status()
    data = resp.json()
    candles = data.get("candles", [])
    if not candles:
        return None
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    return df


def fetch_option_chain(client, symbol):
    resp = client.get_option_chain(symbol)
    resp.raise_for_status()
    return resp.json()


def score_stock(symbol, quote, history_df, option_data):
    price = quote.get("quote", {}).get("lastPrice", 0)
    volume = quote.get("quote", {}).get("totalVolume", 0)

    if price <= 0:
        return None

    shares_affordable = int(ACCOUNT_SIZE / price)

    # ATR% over last 20 days
    atr_pct = 0
    if history_df is not None and len(history_df) >= 20:
        recent = history_df.tail(20).copy()
        recent["range"] = recent["high"] - recent["low"]
        atr_pct = (recent["range"] / recent["close"]).mean() * 100

    # Options liquidity: total open interest across all strikes
    options_oi = 0
    if option_data:
        for side in ("callExpDateMap", "putExpDateMap"):
            for exp in option_data.get(side, {}).values():
                for strikes in exp.values():
                    for opt in strikes:
                        options_oi += opt.get("openInterest", 0)

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "shares_affordable": shares_affordable,
        "volume_1M": round(volume / 1_000_000, 2),
        "atr_pct_20d": round(atr_pct, 2),
        "options_oi_k": round(options_oi / 1000, 1),
    }


def main():
    client = get_client()
    print(f"Screening {len(CANDIDATES)} candidates...\n")

    quotes = fetch_quotes(client, CANDIDATES)

    rows = []
    for symbol in CANDIDATES:
        print(f"  {symbol}...", end=" ", flush=True)
        try:
            history = fetch_price_history(client, symbol)
            options = fetch_option_chain(client, symbol)
            row = score_stock(symbol, quotes.get(symbol, {}), history, options)
            if row:
                rows.append(row)
                print(f"${row['price']}  ATR={row['atr_pct_20d']}%  vol={row['volume_1M']}M  OI={row['options_oi_k']}k")
        except Exception as e:
            print(f"ERROR: {e}")

    if not rows:
        print("No data retrieved.")
        return

    df = pd.DataFrame(rows)

    # Composite score: weight ATR, volume, OI — normalize each
    for col in ("atr_pct_20d", "volume_1M", "options_oi_k"):
        mx = df[col].max()
        df[f"{col}_score"] = df[col] / mx if mx > 0 else 0

    # Penalize stocks where you can afford fewer than 5 shares (options still ok)
    df["afford_score"] = (df["shares_affordable"] >= 5).astype(float)

    df["total_score"] = (
        df["atr_pct_20d_score"] * 0.4
        + df["volume_1M_score"] * 0.3
        + df["options_oi_k_score"] * 0.2
        + df["afford_score"] * 0.1
    )

    df = df.sort_values("total_score", ascending=False)

    print("\n--- Ranking ---")
    display_cols = ["symbol", "price", "shares_affordable", "volume_1M", "atr_pct_20d", "options_oi_k", "total_score"]
    print(df[display_cols].to_string(index=False))

    top = df.iloc[0]
    print(f"\nRecommendation: {top['symbol']} — ${top['price']}, ATR {top['atr_pct_20d']}%/day, "
          f"vol {top['volume_1M']}M, options OI {top['options_oi_k']}k")


if __name__ == "__main__":
    main()
