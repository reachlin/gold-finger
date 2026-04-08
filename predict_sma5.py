#!/usr/bin/env python3
"""Predict the next SMA5 value for a stock using TimesFM.

Usage:
    python predict_sma5.py --csv data/601933_10yr.csv
    python predict_sma5.py --csv data/601933_10yr.csv --horizon 5
"""

import argparse
import sys

import numpy as np
import pandas as pd

_TIMESFM_SRC = "/Users/lincai/dev/3rd-party/timesfm/src"


def predict_next_sma5(csv_path: str, horizon: int = 1, context_len: int = 512):
    # --- Load and compute SMA5 ---
    df = pd.read_csv(csv_path)
    df["sma5"] = df["close"].rolling(5).mean()
    df = df.dropna(subset=["sma5"]).reset_index(drop=True)

    sma5 = df["sma5"].values.astype(np.float32)
    context = sma5[-context_len:]

    # --- Load model ---
    if _TIMESFM_SRC not in sys.path:
        sys.path.insert(0, _TIMESFM_SRC)
    import timesfm

    print("Loading TimesFM 2.5 (200M)…")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=context_len,
            max_horizon=horizon,
            normalize_inputs=True,
            per_core_batch_size=32,
        )
    )
    print("Model ready.\n")

    # --- Forecast ---
    point_forecast, quantile_forecast = model.forecast(
        horizon=horizon,
        inputs=[context],
    )

    # --- Print results ---
    last_date = df["date"].iloc[-1]
    last_sma5 = float(sma5[-1])
    last_close = float(df["close"].iloc[-1])

    print(f"Stock:       {csv_path}")
    print(f"Last date:   {last_date}")
    print(f"Last close:  {last_close:.2f}")
    print(f"Last SMA5:   {last_sma5:.2f}")
    print()

    forecasts = point_forecast[0]       # shape (horizon,)
    q10 = quantile_forecast[0, :, 0]   # 10th percentile
    q90 = quantile_forecast[0, :, 8]   # 90th percentile

    print(f"{'Step':<6} {'Predicted SMA5':>16} {'Change':>10} {'10th pct':>10} {'90th pct':>10}")
    print("-" * 56)
    prev = last_sma5
    for i in range(horizon):
        pred = float(forecasts[i])
        chg = (pred - prev) / prev * 100
        print(f"  +{i+1:<4} {pred:>16.4f} {chg:>+9.2f}% {float(q10[i]):>10.4f} {float(q90[i]):>10.4f}")
        prev = pred

    return forecasts, q10, q90


def main():
    parser = argparse.ArgumentParser(description="Predict next SMA5 with TimesFM")
    parser.add_argument("--csv", default="data/601933_10yr.csv")
    parser.add_argument("--horizon", type=int, default=1,
                        help="How many SMA5 steps to forecast (default: 1)")
    parser.add_argument("--context-len", type=int, default=512,
                        help="History length fed to model (default: 512)")
    args = parser.parse_args()

    predict_next_sma5(args.csv, horizon=args.horizon, context_len=args.context_len)


if __name__ == "__main__":
    main()
