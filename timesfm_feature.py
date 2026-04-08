#!/usr/bin/env python3
"""Compute TimesFM SMA5 forecast as a walk-forward feature column.

Adds `tfm_sma5_ret` to a dataframe:
    tfm_sma5_ret[i] = (predicted_sma5[i+1] - sma5[i]) / sma5[i]

Walk-forward: for row i, only SMA5 values up to (but not including) i are
used as context, so there is no look-ahead leakage.

All N contexts are batched into a single model.forecast() call.
"""

import sys

import numpy as np
import pandas as pd

_TIMESFM_SRC = "/Users/lincai/dev/3rd-party/timesfm/src"


def add_timesfm_feature(
    df: pd.DataFrame,
    context_len: int = 512,
    _model=None,          # injectable for tests
) -> pd.DataFrame:
    """Return a copy of df with tfm_sma5_ret column added.

    Requires df to have a 'close' column. SMA5 is computed internally.
    Rows without sufficient SMA5 history (first 4 rows) are set to 0.
    """
    df = df.copy()

    # Compute SMA5 if not already present
    if "sma5" not in df.columns:
        df["sma5"] = df["close"].rolling(5).mean()

    sma5_vals = df["sma5"].values.astype(np.float64)
    n = len(df)
    tfm_ret = np.zeros(n, dtype=np.float64)

    # Find rows with valid SMA5
    valid_mask = ~np.isnan(sma5_vals)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        df["tfm_sma5_ret"] = tfm_ret
        return df

    # Load model once if not injected
    model = _model
    if model is None:
        model = _load_timesfm(context_len)

    # Build walk-forward contexts: for valid row at position p,
    # context = sma5[max(0, p-context_len) : p]
    inputs = []
    for p in valid_indices:
        start = max(0, p - context_len)
        ctx = sma5_vals[start:p]
        ctx = ctx[~np.isnan(ctx)]   # strip NaN warmup values at start of series
        if len(ctx) == 0:
            ctx = np.array([sma5_vals[p]], dtype=np.float32)
        inputs.append(ctx.astype(np.float32))

    point_forecast, _ = model.forecast(horizon=1, inputs=inputs)

    for k, p in enumerate(valid_indices):
        current = sma5_vals[p]
        predicted = float(point_forecast[k, 0])
        if current > 0:
            tfm_ret[p] = (predicted - current) / current

    df["tfm_sma5_ret"] = tfm_ret
    return df


def _load_timesfm(context_len: int):
    """Load and compile TimesFM 2.5."""
    if _TIMESFM_SRC not in sys.path:
        sys.path.insert(0, _TIMESFM_SRC)
    import timesfm  # noqa: PLC0415
    print("Loading TimesFM 2.5 for feature computation…")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=context_len,
            max_horizon=1,
            normalize_inputs=True,
            per_core_batch_size=32,
        )
    )
    print("TimesFM ready.")
    return model
