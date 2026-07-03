"""
LGBM assignment-risk advisory for the Scavenger.

Answers the exact question a cash-secured put asks: how likely is this stock
to fall below the strike before expiry? A LightGBM classifier is trained per
symbol on daily history with the binary label

    "did the close fall below 95% of today's close within the next
     30 trading days?"

which mirrors the Scavenger's 5% OTM / 30 DTE put. The predicted probability
is attached to SELL_PUT signals as `assign_risk_pct` — ADVISORY ONLY. It
never gates a trade; the AutoOverseer LLM sees it next to the Kronos support
estimate and weighs both (Kronos gives a predicted floor, this gives a
calibrated probability).

Training takes seconds per symbol (LightGBM on ~2k tabular rows), so models
are retrained from data/*_history.csv at every scanner startup. This reuses
the labeling/percentile approach of the root lgbm_trading_bot.py but is
standalone — no dependency on the China A-share stack.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

HORIZON   = 30      # trading days — matches SELL_DTE
DROP_PCT  = 0.05    # 5% below close — matches OTM_PUT_PCT
MIN_ROWS  = 200     # minimum usable rows to train
FEATURES  = ["rsi", "adx", "vol_ratio", "dist_ema20", "dist_ema50",
             "atr_pct", "hv20"]


def make_labels(closes: pd.Series, horizon: int = HORIZON,
                drop_pct: float = DROP_PCT) -> np.ndarray:
    """
    1.0 if the close drops below close*(1-drop_pct) within the next `horizon`
    bars, else 0.0. The trailing `horizon` bars have no full future window
    and are NaN (excluded from training).
    """
    values = closes.to_numpy(dtype=float)
    n      = len(values)
    labels = np.full(n, np.nan)
    # future_min[i] = min(values[i+1 : i+1+horizon])
    for i in range(n - horizon):
        future_min = values[i + 1: i + 1 + horizon].min()
        labels[i]  = 1.0 if future_min < values[i] * (1 - drop_pct) else 0.0
    return labels


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabular features per row from raw OHLCV. Uses the same indicator engine
    as the scanner (trend_scanner.compute_indicators) plus normalized
    distances and a 20-day HV. First ~60 rows are NaN (indicator warm-up).
    """
    from trend_scanner import compute_indicators
    ind = compute_indicators(df)

    out = pd.DataFrame(index=ind.index)
    out["rsi"]        = ind["rsi"]
    out["adx"]        = ind["adx"]
    out["vol_ratio"]  = ind["vol_ratio"]
    out["dist_ema20"] = ind["close"] / ind["ema20"] - 1
    out["dist_ema50"] = ind["close"] / ind["ema50"] - 1
    out["atr_pct"]    = ind["atr"] / ind["close"]
    log_ret           = np.log(ind["close"] / ind["close"].shift(1))
    out["hv20"]       = log_ret.rolling(20).std() * np.sqrt(252)
    return out


class AssignmentRiskModel:
    """Per-symbol binary classifier: P(>5% drawdown within 30 trading days)."""

    def __init__(self):
        self.model       = None
        self.holdout_auc = None

    def fit(self, df: pd.DataFrame):
        from lightgbm import LGBMClassifier

        feats  = make_features(df)
        labels = pd.Series(make_labels(df["close"]), index=feats.index)
        mask   = ~(feats.isna().any(axis=1) | labels.isna())
        X, y   = feats[mask], labels[mask]

        if len(X) < MIN_ROWS:
            raise ValueError(f"need >= {MIN_ROWS} usable rows, got {len(X)}")

        # Chronological holdout AUC — sanity signal, not a gate
        split = int(len(X) * 0.8)
        model_kwargs = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                            num_leaves=15, random_state=42, verbose=-1)
        if 0 < y.iloc[split:].sum() < len(y) - split and y.iloc[:split].nunique() > 1:
            probe = LGBMClassifier(**model_kwargs)
            probe.fit(X.iloc[:split], y.iloc[:split])
            try:
                from sklearn.metrics import roc_auc_score
                pred = probe.predict_proba(X.iloc[split:])[:, 1]
                self.holdout_auc = float(roc_auc_score(y.iloc[split:], pred))
            except Exception:
                self.holdout_auc = None

        self.model = LGBMClassifier(**model_kwargs)
        self.model.fit(X, y)
        return self

    def predict_prob(self, df: pd.DataFrame) -> float:
        """Probability for the most recent row of df (raw OHLCV)."""
        feats = make_features(df).iloc[[-1]]
        return float(self.model.predict_proba(feats)[:, 1][0])


def load_models(symbols: list[str], data_dir: str) -> dict:
    """
    Train one model per symbol from {data_dir}/{sym}_history.csv.
    Symbols without data or with training failures are skipped — the
    advisory is optional by design.
    """
    models: dict[str, AssignmentRiskModel] = {}
    for sym in symbols:
        path = os.path.join(data_dir, f"{sym.lower()}_history.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            models[sym] = AssignmentRiskModel().fit(df)
        except Exception as exc:
            print(f"  [AssignRisk] {sym}: training skipped ({exc})")
    return models


def advise(models: dict, symbol: str, df: pd.DataFrame) -> float | None:
    """P(assignment-level drop) as a percentage, or None if no model."""
    model = models.get(symbol)
    if model is None:
        return None
    try:
        return round(model.predict_prob(df) * 100, 1)
    except Exception:
        return None
