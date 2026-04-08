#!/usr/bin/env python3
"""Load saved models, predict signals, and evaluate model drift.

Usage:
    python predict.py --symbol 601933                    # predict only
    python predict.py --symbol 601933 --evaluate         # predict + drift check
    python predict.py --symbol 601933 --evaluate --notify # also Slack alert if drift
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

from trading_bot import TradingBot, compute_indicators, FEATURE_COLS
from dnn_trading_bot import DNNTradingBot
from lgbm_trading_bot import LGBMTradingBot
from ppo_trading_bot import PPOTradingBot

DRIFT_THRESHOLD = 0.45  # rolling accuracy below this triggers retrain alert


def classify(signal: str) -> str:
    if signal in ("strong_buy", "mild_buy"):
        return "BUY"
    elif signal in ("strong_sell", "mild_sell"):
        return "SELL"
    return "HOLD"


def evaluate_signals(directions: list[str], actual_returns: list[float]) -> float:
    """Compute accuracy: did signal direction match actual return direction?

    BUY + positive return = correct
    SELL + negative return = correct
    HOLD = always correct (neutral stance)
    Returns accuracy as a float in [0, 1]. Returns 0.0 for empty input.
    """
    if not directions:
        return 0.0
    correct = 0
    for direction, ret in zip(directions, actual_returns):
        if direction == "HOLD":
            correct += 1
        elif direction == "BUY" and ret > 0:
            correct += 1
        elif direction == "SELL" and ret < 0:
            correct += 1
    return correct / len(directions)


def rolling_accuracy(directions: list[str], actual_returns: list[float],
                     window: int = 20) -> float:
    """Compute accuracy over the last `window` days."""
    n = min(window, len(directions))
    if n == 0:
        return 0.0
    return evaluate_signals(directions[-n:], actual_returns[-n:])


def needs_retrain(rolling_acc: float) -> bool:
    """Return True if model accuracy has drifted below threshold."""
    return rolling_acc < DRIFT_THRESHOLD


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


def _predict_row(models: dict, df: pd.DataFrame, row_idx: int) -> str:
    """Predict majority vote direction for a single row."""
    row = df.iloc[row_idx]
    signals = {}
    for name, bot in models.items():
        if name == "LSTM":
            # LSTM needs a window of rows ending at row_idx
            if row_idx < bot.window_size:
                signals[name] = "HOLD"
                continue
            sig = bot.predict_single(df.iloc[:row_idx + 1])
        else:
            sig = bot.predict_single(row)
        signals[name] = classify(sig)

    counts = Counter(signals.values())
    majority = next((d for d, c in counts.most_common() if c >= 3), None)
    return majority if majority else "HOLD"


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


def evaluate(symbol: str, csv_path: str | None = None, notify: bool = False) -> dict:
    """Evaluate model accuracy on post-training data and predict today's signal.

    Returns dict with: today_signal, accuracy, rolling_20d_accuracy,
    num_eval_days, needs_retrain, last_date, last_close.
    """
    model_dir = os.path.join("models", symbol)
    if not os.path.isdir(model_dir):
        print(f"No saved models found at {model_dir}/")
        sys.exit(1)

    # Load metadata for training cutoff
    meta_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"No metadata.json in {model_dir}/")
        sys.exit(1)
    with open(meta_path) as f:
        metadata = json.load(f)
    cutoff_date = metadata["data_range"]["end"]
    print(f"Model: {metadata.get('label', symbol)}")
    print(f"Trained: {metadata.get('train_date', 'unknown')}")
    print(f"Training cutoff: {cutoff_date}")

    # Load price data
    if csv_path is None:
        for pattern in [
            f"data/{symbol}_10yr.csv",
            f"data/{symbol}_20yr.csv",
            f"data/{symbol}.csv",
        ]:
            if os.path.exists(pattern):
                csv_path = pattern
                break
    if csv_path is None or not os.path.exists(csv_path):
        print(f"\nNo price data found for {symbol}.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df = compute_indicators(df).dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"Data: {len(df)} rows ({df['date'].iloc[0]} to {df['date'].iloc[-1]})")

    # Split at training cutoff
    cutoff_mask = df["date"] > cutoff_date
    new_indices = df.index[cutoff_mask].tolist()

    # Load models
    models = load_models(model_dir)
    if not models:
        print("No models could be loaded.")
        sys.exit(1)

    # Evaluate on post-training data
    directions = []
    actual_returns = []
    for i in new_indices:
        if i + 1 >= len(df):
            break  # can't compute next-day return for last row
        direction = _predict_row(models, df, i)
        next_day_return = (df.loc[i + 1, "close"] - df.loc[i, "close"]) / df.loc[i, "close"]
        directions.append(direction)
        actual_returns.append(next_day_return)

    num_eval = len(directions)
    acc = evaluate_signals(directions, actual_returns)
    roll_acc = rolling_accuracy(directions, actual_returns, window=20)
    retrain = needs_retrain(roll_acc) if num_eval >= 5 else False

    # Today's signal (last row)
    today_direction = _predict_row(models, df, len(df) - 1)
    last_row = df.iloc[-1]

    # Print results
    print(f"\nLatest data: {last_row['date']}, close={last_row['close']:.2f}")
    print(f"\n  Evaluation ({num_eval} days since training cutoff):")
    print(f"    Overall accuracy:     {acc:.1%}")
    print(f"    Rolling 20d accuracy: {roll_acc:.1%}")
    print(f"    Drift threshold:      {DRIFT_THRESHOLD:.0%}")
    print(f"    Needs retrain:        {'YES' if retrain else 'No'}")
    print(f"\n  Today's signal: {today_direction}")

    # Slack alert if needed
    if notify and retrain:
        try:
            from notify_slack import send
            msg = (
                f":warning: *{symbol}* model drift detected!\n"
                f"Rolling 20d accuracy: *{roll_acc:.1%}* (threshold: {DRIFT_THRESHOLD:.0%})\n"
                f"Days since training: *{num_eval}*\n"
                f"Please retrain models."
            )
            send(msg)
            print("\n  Slack retrain alert sent.")
        except Exception as e:
            print(f"\n  Failed to send Slack alert: {e}")

    return {
        "today_signal": today_direction,
        "last_date": str(last_row["date"]),
        "last_close": float(last_row["close"]),
        "accuracy": acc,
        "rolling_20d_accuracy": roll_acc,
        "num_eval_days": num_eval,
        "needs_retrain": retrain,
    }


def main():
    parser = argparse.ArgumentParser(description="Load saved models, predict, and evaluate drift")
    parser.add_argument("--symbol", required=True, help="Stock symbol (e.g. 601933)")
    parser.add_argument("--csv", default=None, help="Path to price CSV (auto-detected if omitted)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model accuracy on new data")
    parser.add_argument("--notify", action="store_true", help="Send Slack alert if retrain needed")
    args = parser.parse_args()

    if args.evaluate:
        evaluate(args.symbol, args.csv, notify=args.notify)
    else:
        predict(args.symbol, args.csv)


if __name__ == "__main__":
    main()
