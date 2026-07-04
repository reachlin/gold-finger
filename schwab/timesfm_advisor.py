"""
TimesFM second-opinion advisory — zero-shot 30-day SMA5 direction forecast.

At scanner startup, Google's TimesFM 2.5 (200M, zero-shot — no training)
forecasts the next PRED_LEN trading days of each watchlist symbol's SMA5
series. The percent change of the forecast endpoint vs today's SMA5 is
attached to signals as `timesfm_30d_pct` — ADVISORY ONLY, never gates.

This gives the AutoOverseer LLM an independent foundation-model view next
to Kronos: both bearish on a SELL_PUT → stronger skip signal than either
alone; strongly positive on a SELL_CALL → the call likely caps a run.

SMA5 (not raw closes) is forecast because the smoothed series is far less
noisy day-to-day — the same choice that made timesfm_sma5_bot.py the best
TimesFM variant in the A-share experiments. Model load + compile takes
~30s and ~1GB download on first run; forecasting all 18 symbols is one
batched call. If TimesFM is unavailable the scanner runs without it.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

TIMESFM_SRC = os.environ.get("TIMESFM_SRC",
                             "/Users/lincai/dev/3rd-party/timesfm/src")
CHECKPOINT  = "google/timesfm-2.5-200m-pytorch"
SMA_WINDOW  = 5      # smoothing window — matches timesfm_sma5_bot.py
PRED_LEN    = 30     # trading days — matches SELL_DTE / Kronos PRED_LEN
CONTEXT_LEN = 512    # bars of SMA5 history fed to the model
MIN_ROWS    = 100    # minimum usable SMA5 points to forecast


def build_context(closes: pd.Series, window: int = SMA_WINDOW,
                  context_len: int = CONTEXT_LEN) -> "np.ndarray | None":
    """Trailing SMA window of the close series, trimmed to context_len."""
    sma = closes.rolling(window).mean().dropna().to_numpy(dtype=np.float32)
    if len(sma) < MIN_ROWS:
        return None
    return sma[-context_len:]


def _load_forecast_fn():
    """
    Load and compile TimesFM 2.5. Returns forecast_fn(contexts, horizon)
    -> np.ndarray of shape (len(contexts), horizon).
    """
    if TIMESFM_SRC not in sys.path:
        sys.path.insert(0, TIMESFM_SRC)
    import timesfm
    print("Loading TimesFM 2.5 (200M, zero-shot)...", flush=True)
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(CHECKPOINT)
    model.compile(timesfm.ForecastConfig(
        max_context=CONTEXT_LEN,
        max_horizon=PRED_LEN,
        normalize_inputs=True,
        per_core_batch_size=32,
    ))
    print("  TimesFM ready", flush=True)

    def forecast_fn(contexts, horizon):
        point, _ = model.forecast(
            horizon=horizon,
            inputs=[np.asarray(c, dtype=np.float32) for c in contexts],
        )
        return np.asarray(point)

    return forecast_fn


def load_cache(symbols: list[str], data_dir: str,
               forecast_fn=None) -> dict:
    """
    One batched forecast for all symbols with local history. Returns
    {sym: {"sma_now", "sma_pred", "pct"}}, or {} if TimesFM is unavailable —
    the advisory is optional by design.
    """
    try:
        fn = forecast_fn or _load_forecast_fn()
    except Exception as exc:
        print(f"  [TimesFM] unavailable — scanner runs without it ({exc})")
        return {}

    syms, contexts = [], []
    for sym in symbols:
        path = os.path.join(data_dir, f"{sym.lower()}_history.csv")
        if not os.path.exists(path):
            continue
        try:
            ctx = build_context(pd.read_csv(path)["close"])
        except Exception as exc:
            print(f"  [TimesFM] {sym}: skipped ({exc})")
            continue
        if ctx is not None:
            syms.append(sym)
            contexts.append(ctx)

    if not contexts:
        return {}

    cache: dict = {}
    try:
        preds = fn(contexts, PRED_LEN)
        for sym, ctx, pred in zip(syms, contexts, preds):
            sma_now, sma_pred = float(ctx[-1]), float(pred[-1])
            if sma_now <= 0:
                continue
            cache[sym] = {
                "sma_now":  round(sma_now, 2),
                "sma_pred": round(sma_pred, 2),
                "pct":      round((sma_pred / sma_now - 1) * 100, 1),
            }
        print(f"  [TimesFM] cache ready: {len(cache)}/{len(symbols)} symbols",
              flush=True)
    except Exception as exc:
        print(f"  [TimesFM] forecast failed ({exc})")
    return cache


def advise(cache: dict, symbol: str) -> "float | None":
    """Forecast SMA5 change over the next 30 trading days as a %, or None."""
    entry = cache.get(symbol)
    return None if entry is None else entry["pct"]
