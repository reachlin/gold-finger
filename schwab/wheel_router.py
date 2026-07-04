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
