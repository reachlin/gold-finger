"""
Wheel-vs-hold strategy router — walk-forward P(>8% rally within 30 days).

The wheel's negative edge vs buy-and-hold is concentrated in runners
(AMD -$22K, GOOGL -$13K, HD -$11K, MSFT -$9K on the 2026-07-04 checkpoint):
covered calls cap every rally at strike+premium, and cash-secured put
premium (~1.5%/mo) can't compete with a trending stock's drift. The router
asks the question that actually matters — "should this symbol be wheeled
at all right now?" — using the same label as the called-away model
(assignment_risk direction="up", 8% matching the covered-call strike):

    P high → hold mode: own shares uncapped (buy instead of selling a put;
             skip the covered call after assignment)
    P low  → wheel mode: collect premium as usual

Unlike the LLM advisories, this is a mechanical rule, so it goes straight
into the walk-forward backtest and gets measured against the baseline.

No lookahead: `walk_forward_probs` refits a LightGBM per symbol every
`refit_every` bars using only data up to that bar (make_labels_up NaNs the
trailing HORIZON rows, so no label peeks past the fit bar). Features are
computed once over the full series — all indicators are causal. Raw
(uncalibrated) probabilities are used for speed; thresholds are swept in
backtest_router.py so absolute calibration matters less than ranking.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from assignment_risk import (
    make_features,
    make_labels_up,
    HORIZON,
    RISE_PCT,
)

REFIT_EVERY = 63     # trading days between refits (~quarterly)
MIN_TRAIN   = 300    # bars of history before the first fit
MODEL_KW    = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                   num_leaves=15, random_state=42, verbose=-1)


def walk_forward_probs(df: pd.DataFrame, refit_every: int = REFIT_EVERY,
                       min_train: int = MIN_TRAIN,
                       rise_pct: float = RISE_PCT) -> np.ndarray:
    """
    Per-bar P(close rises >rise_pct within the next HORIZON bars), refit
    every `refit_every` bars on data up to that bar only. Bars before the
    first successful fit (or where no model is trainable) are NaN.
    """
    from lightgbm import LGBMClassifier

    n      = len(df)
    feats  = make_features(df)                       # causal — computed once
    labels = pd.Series(make_labels_up(df["close"], HORIZON, rise_pct),
                       index=feats.index)
    probs  = np.full(n, np.nan)
    model  = None

    for start in range(min_train, n, refit_every):
        # Train on rows [0, start - HORIZON): row j's label looks at closes
        # up to j+HORIZON, so rows past start-HORIZON would peek beyond the
        # fit bar — exactly the lookahead this walk-forward must not have.
        f_tr = feats.iloc[:start - HORIZON]
        l_tr = labels.iloc[:start - HORIZON]
        mask = ~(f_tr.isna().any(axis=1) | l_tr.isna())
        X, y = f_tr[mask], l_tr[mask]
        if len(X) >= min_train // 2 and y.nunique() > 1:
            m = LGBMClassifier(**MODEL_KW)
            try:
                m.fit(X, y)
                model = m
            except Exception:
                pass                                  # keep previous model
        if model is None:
            continue

        end   = min(start + refit_every, n)
        chunk = feats.iloc[start:end]
        ok    = ~chunk.isna().any(axis=1)
        if ok.any():
            probs[start:end][ok.to_numpy()] = \
                model.predict_proba(chunk[ok])[:, 1]
    return probs


def load_probs(symbol: str, data_dir: str) -> "np.ndarray | None":
    """Walk-forward probability series from {data_dir}/{sym}_history.csv."""
    path = os.path.join(data_dir, f"{symbol.lower()}_history.csv")
    if not os.path.exists(path):
        return None
    try:
        return walk_forward_probs(pd.read_csv(path))
    except Exception as exc:
        print(f"  [Router] {symbol}: skipped ({exc})")
        return None


# ---------------------------------------------------------------------------
# Live scanner side — mechanical proposes, LLM disposes.
#
# The scanner calls live_route() with the cached TimesFM forecast; the
# resulting HOLD_SHARES / RESUME_WHEEL signal goes through the AutoOverseer
# like any other signal (default APPROVE — the mechanical policy is the
# backtested one; the LLM vetoes only on context the models can't see).
# Approved holds persist in a JSON file so hold mode survives restarts.
# ---------------------------------------------------------------------------

ROUTER_TAU = 3.5   # % — mid-plateau of the 2026-07-04 experiments
                   # (full history and 2024+ validation both hold at 3.0-4.0)


def live_route(forecast_pct: "float | None", in_hold: bool) -> "str | None":
    """HOLD_SHARES to enter hold mode, RESUME_WHEEL to exit, None to stay."""
    if forecast_pct is None:
        return None
    if not in_hold and forecast_pct >= ROUTER_TAU:
        return "HOLD_SHARES"
    if in_hold and forecast_pct < ROUTER_TAU:
        return "RESUME_WHEEL"
    return None


def load_holds(path: str) -> dict:
    """{symbol: {"shares", "entry", "date"}} — {} if no file."""
    import json
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def save_holds(path: str, holds: dict) -> None:
    import json
    with open(path, "w") as f:
        json.dump(holds, f, indent=2)


def enter_hold(path: str, symbol: str, shares: int, entry_price: float,
               date: str) -> None:
    holds = load_holds(path)
    holds[symbol] = {"shares": shares, "entry": entry_price, "date": date}
    save_holds(path, holds)


def exit_hold(path: str, symbol: str, exit_price: float) -> "float | None":
    """Remove the hold and return realized P&L, or None if not held."""
    holds = load_holds(path)
    pos = holds.pop(symbol, None)
    if pos is None:
        return None
    save_holds(path, holds)
    return (exit_price - pos["entry"]) * pos["shares"]


# ---------------------------------------------------------------------------
# TimesFM as the predictor — zero-shot, so walk-forward is free of training
# leakage by construction: the context at bar i is the trailing SMA5 window
# ending at bar i, nothing else.
# ---------------------------------------------------------------------------

MIN_CTX    = 100    # minimum SMA5 points before forecasting a bar
BATCH_SIZE = 256    # contexts per model call


def walk_forward_timesfm(df: pd.DataFrame, forecast_fn=None,
                         batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    Per-bar zero-shot forecast of the SMA5 change over the next PRED_LEN
    bars, in % (the same quantity timesfm_advisor caches for the live
    scanner, computed at every bar). NaN during SMA/context warm-up.

    The output plugs into walk_forward_scavenger(router_probs=...) with
    router_threshold expressed in % (e.g. 4.0 = forecast >= +4%).
    """
    from timesfm_advisor import (SMA_WINDOW, PRED_LEN, CONTEXT_LEN,
                                 _load_forecast_fn)

    fn  = forecast_fn or _load_forecast_fn()
    sma = df["close"].rolling(SMA_WINDOW).mean().to_numpy(dtype=np.float32)
    n   = len(df)
    out = np.full(n, np.nan)

    contexts, idxs = [], []
    for i in range(n):
        if np.isnan(sma[i]):
            continue
        start = max(SMA_WINDOW - 1, i - CONTEXT_LEN + 1)
        ctx   = sma[start:i + 1]
        if len(ctx) < MIN_CTX:
            continue
        contexts.append(ctx)
        idxs.append(i)

    for b in range(0, len(contexts), batch_size):
        batch = contexts[b:b + batch_size]
        preds = np.asarray(fn(batch, PRED_LEN))
        for k, i in enumerate(idxs[b:b + batch_size]):
            now, pred = float(batch[k][-1]), float(preds[k][-1])
            if now > 0:
                out[i] = (pred / now - 1) * 100
    return out
