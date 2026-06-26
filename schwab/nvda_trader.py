"""
NVDA daily trader: fetch data from Schwab → LGBM signal → place order.
"""
import os
import sys
import schwab
import pandas as pd
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trading_bot import compute_indicators, FEATURE_COLS
from lgbm_trading_bot import LGBMTradingBot

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

CLIENT_ID = os.environ["SCHWAB_CLIENT_ID"]
CLIENT_SECRET = os.environ["SCHWAB_CLIENT_SECRET"]
REDIRECT_URI = "https://127.0.0.1"
TOKEN_PATH = os.path.join(os.path.dirname(__file__), "schwab_token.json")

SYMBOL = "NVDA"
TRADE_BUDGET = 600          # USD per trade
SIGNAL_NAMES = ["strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"]


def get_client():
    return schwab.auth.client_from_token_file(TOKEN_PATH, CLIENT_ID, CLIENT_SECRET)


def fetch_nvda_history(client, days=400):
    from datetime import datetime, timedelta
    end = datetime.now()
    start = end - timedelta(days=days)
    resp = client.get_price_history_every_day(SYMBOL, start_datetime=start, end_datetime=end)
    resp.raise_for_status()
    return resp.json().get("candles", [])


def candles_to_df(candles):
    df = pd.DataFrame(candles, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def compute_trade_signal(df):
    """Train LGBM on full history and predict signal for the last row."""
    df = compute_indicators(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    bot = LGBMTradingBot()
    bot.fit(df)

    last = df[FEATURE_COLS].iloc[[-1]]
    last_scaled = pd.DataFrame(bot.scaler.transform(last), columns=FEATURE_COLS)
    pred = bot.model.predict(last_scaled)[0]
    return SIGNAL_NAMES[pred]


def get_position_shares(client, symbol):
    resp = client.get_accounts()
    resp.raise_for_status()
    for entry in resp.json():
        for pos in entry["securitiesAccount"].get("positions", []):
            if pos["instrument"]["symbol"] == symbol:
                return int(pos["longQuantity"] - pos["shortQuantity"])
    return 0


def build_order(symbol, instruction, quantity, price):
    if quantity <= 0:
        raise ValueError(f"Quantity must be > 0, got {quantity}")
    return {
        "orderType": "LIMIT",
        "session": "NORMAL",
        "duration": "DAY",
        "price": round(price, 2),
        "orderLegCollection": [{
            "instruction": instruction,
            "quantity": quantity,
            "instrument": {"symbol": symbol, "assetType": "EQUITY"},
        }],
    }


def get_account_hash(client):
    resp = client.get_account_numbers()
    resp.raise_for_status()
    return resp.json()[0]["hashValue"]


def main():
    client = get_client()

    print("Fetching NVDA history...")
    candles = fetch_nvda_history(client)
    df = candles_to_df(candles)
    print(f"  {len(df)} trading days loaded, latest: {df['datetime'].iloc[-1].date()}")

    signal = compute_trade_signal(df)
    print(f"Signal: {signal}")

    last_price = df["close"].iloc[-1]
    shares_held = get_position_shares(client, SYMBOL)
    print(f"Current position: {shares_held} shares @ last close ${last_price:.2f}")

    action = None
    qty = 0

    if signal in ("strong_buy", "mild_buy") and shares_held == 0:
        qty = max(1, int(TRADE_BUDGET / last_price))
        action = "BUY"
    elif signal in ("strong_sell", "mild_sell") and shares_held > 0:
        qty = shares_held
        action = "SELL"

    if action and qty > 0:
        order = build_order(SYMBOL, action, qty, last_price)
        account_hash = get_account_hash(client)
        print(f"Placing {action} {qty} shares of {SYMBOL} @ ${last_price:.2f}...")
        resp = client.place_order(account_hash, order)
        resp.raise_for_status()
        print("Order placed.")
    else:
        print("No trade today.")


if __name__ == "__main__":
    main()
