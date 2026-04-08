#!/usr/bin/env python3
"""Load saved models and output today's trading signal for a given stock.

Usage:
    python predict.py --symbol 601933
    python predict.py --symbol 601933 --csv data/601933_10yr.csv
"""

import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd

from trading_bot import TradingBot, compute_indicators, FEATURE_COLS
from dnn_trading_bot import DNNTradingBot
from lgbm_trading_bot import LGBMTradingBot
from ppo_trading_bot import PPOTradingBot


def classify(signal: str) -> str:
    if signal in ("strong_buy", "mild_buy"):
        return "BUY"
    elif signal in ("strong_sell", "mild_sell"):
        return "SELL"
    return "HOLD"


def load_models(model_dir: str) -> dict:
    """Load all available models from a directory."""
    models = {}

    km_path = os.path.join(model_dir, "kmeans.joblib")
    if os.path.exists(km_path):
        models["K-Means"] = TradingBot.load(km_path)

    lstm_path = os.path.join(model_dir, "lstm.pt")
    if os.path.exists(lstm_path):
        models["LSTM"] = DNNTradingBot.load(lstm_path)

    lgbm_path = os.path.join(model_dir, "lgbm.joblib")
    if os.path.exists(lgbm_path):
        models["LightGBM"] = LGBMTradingBot.load(lgbm_path)

    ppo_path = os.path.join(model_dir, "ppo_model.zip")
    if os.path.exists(ppo_path):
        models["PPO"] = PPOTradingBot.load(model_dir)

    return models


def predict(symbol: str, csv_path: str | None = None):
    """Load models and predict today's signal."""
    model_dir = os.path.join("models", symbol)
    if not os.path.isdir(model_dir):
        print(f"No saved models found at {model_dir}/")
        print("Run daily_pipeline.py first to train and save models.")
        sys.exit(1)

    # Load metadata
    meta_path = os.path.join(model_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
        print(f"Model: {metadata.get('label', symbol)}")
        print(f"Trained: {metadata.get('train_date', 'unknown')}")
        print(f"Data range: {metadata.get('data_range', {}).get('start', '?')} "
              f"to {metadata.get('data_range', {}).get('end', '?')}")
    else:
        print(f"Model: {symbol}")

    # Load price data
    if csv_path is None:
        # Try common patterns
        for pattern in [
            f"data/{symbol}_10yr.csv",
            f"data/{symbol}_20yr.csv",
            f"data/{symbol}.csv",
        ]:
            if os.path.exists(pattern):
                csv_path = pattern
                break
    if csv_path is None or not os.path.exists(csv_path):
        print(f"\nNo price data found for {symbol}. Provide --csv path.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df = compute_indicators(df).dropna(subset=FEATURE_COLS).reset_index(drop=True)
    last_row = df.iloc[-1]
    print(f"\nLatest data: {last_row['date']}, close={last_row['close']:.2f}")

    # Load models and predict
    models = load_models(model_dir)
    if not models:
        print("No models could be loaded.")
        sys.exit(1)

    print(f"\n  {'Model':<14s} {'Signal':<16s} {'Direction'}")
    print("  " + "-" * 42)

    signals = {}
    for name, bot in models.items():
        if name == "LSTM":
            sig = bot.predict_single(df)
        else:
            sig = bot.predict_single(last_row)
        direction = classify(sig)
        signals[name] = direction
        print(f"  {name:<14s} {sig:<16s} {direction}")

    # Majority vote
    counts = Counter(signals.values())
    majority = next((d for d, c in counts.most_common() if c >= 3), None)
    verdict = majority if majority else "NO CONSENSUS"
    print(f"\n  Majority vote (>= 3/{len(signals)}): {verdict}")


def main():
    parser = argparse.ArgumentParser(description="Load saved models and predict today's signal")
    parser.add_argument("--symbol", required=True, help="Stock symbol (e.g. 601933)")
    parser.add_argument("--csv", default=None, help="Path to price CSV (auto-detected if omitted)")
    args = parser.parse_args()
    predict(args.symbol, args.csv)


if __name__ == "__main__":
    main()
