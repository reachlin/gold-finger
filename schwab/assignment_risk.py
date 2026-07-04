"""
LGBM move-risk advisories for the Scavenger and Raider.

Two per-symbol binary classifiers, one per direction:

  direction="down" (the original assignment-risk model)
    "did the close fall below 95% of today's close within the next
     30 trading days?"  — mirrors the 5% OTM / 30 DTE cash-secured put.
    Attached to SELL_PUT signals as `assign_risk_pct` and to Raider BUY
    signals as `drop_risk_pct` (a pullback with high drop risk may be a
    breakdown).

  direction="up" (the call-side mirror)
    "did the close rise above 108% of today's close within the next
     30 trading days?"  — mirrors the 8% OTM / 30 DTE covered call.
    Attached to SELL_CALL signals as `called_away_pct`.

All fields are ADVISORY ONLY — they never gate a trade; the AutoOverseer
LLM sees them next to the Kronos range estimate and weighs both. Predicted
probabilities are isotonic-calibrated on chronological folds, and each
model's holdout AUC is attached as `model_auc` so the LLM can discount
weak per-symbol models.

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
RISE_PCT  = 0.08    # 8% above close — matches OTM_CALL_PCT (base, sideways)
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


def make_labels_up(closes: pd.Series, horizon: int = HORIZON,
                   rise_pct: float = RISE_PCT) -> np.ndarray:
    """
    Call-side mirror of make_labels: 1.0 if the close rises above
    close*(1+rise_pct) within the next `horizon` bars, else 0.0. The trailing
    `horizon` bars have no full future window and are NaN.
    """
    values = closes.to_numpy(dtype=float)
    n      = len(values)
    labels = np.full(n, np.nan)
    for i in range(n - horizon):
        future_max = values[i + 1: i + 1 + horizon].max()
        labels[i]  = 1.0 if future_max > values[i] * (1 + rise_pct) else 0.0
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
    """
    Per-symbol binary classifier.
      direction="down": P(>move_pct drawdown within 30 trading days)
      direction="up"  : P(>move_pct rally    within 30 trading days)
    """

    def __init__(self, direction: str = "down", move_pct: float | None = None):
        if direction not in ("down", "up"):
            raise ValueError(f"direction must be 'down' or 'up', got {direction!r}")
        self.direction   = direction
        self.move_pct    = move_pct if move_pct is not None else (
            DROP_PCT if direction == "down" else RISE_PCT)
        self.model       = None
        self.holdout_auc = None
        self.calibrated  = False

    def fit(self, df: pd.DataFrame):
        from lightgbm import LGBMClassifier

        feats   = make_features(df)
        labeler = make_labels if self.direction == "down" else make_labels_up
        labels  = pd.Series(labeler(df["close"], HORIZON, self.move_pct),
                            index=feats.index)
        mask    = ~(feats.isna().any(axis=1) | labels.isna())
        X, y    = feats[mask], labels[mask]

        if len(X) < MIN_ROWS:
            raise ValueError(f"need >= {MIN_ROWS} usable rows, got {len(X)}")

        # Chronological holdout AUC — sanity signal, not a gate. AUC is
        # invariant to monotonic calibration, so the raw probe is enough.
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

        # Isotonic calibration on chronological folds — raw LGBM probabilities
        # are typically over/under-confident. Falls back to the uncalibrated
        # classifier when a fold lacks both classes.
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.model_selection import TimeSeriesSplit
            calib = CalibratedClassifierCV(LGBMClassifier(**model_kwargs),
                                           method="isotonic",
                                           cv=TimeSeriesSplit(n_splits=3))
            calib.fit(X, y)
            self.model      = calib
            self.calibrated = True
        except Exception:
            self.model = LGBMClassifier(**model_kwargs)
            self.model.fit(X, y)
            self.calibrated = False
        return self

    def predict_prob(self, df: pd.DataFrame) -> float:
        """Probability for the most recent row of df (raw OHLCV)."""
        feats = make_features(df).iloc[[-1]]
        return float(self.model.predict_proba(feats)[:, 1][0])


def load_models(symbols: list[str], data_dir: str,
                direction: str = "down") -> dict:
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
            models[sym] = AssignmentRiskModel(direction=direction).fit(df)
        except Exception as exc:
            print(f"  [AssignRisk] {sym}: training skipped ({exc})")
    return models


def advise(models: dict, symbol: str, df: pd.DataFrame) -> float | None:
    """P(threshold move in the model's direction) as a %, or None if no model."""
    model = models.get(symbol)
    if model is None:
        return None
    try:
        return round(model.predict_prob(df) * 100, 1)
    except Exception:
        return None


def model_auc(models: dict, symbol: str) -> float | None:
    """Chronological holdout AUC of the symbol's model, or None."""
    model = models.get(symbol)
    if model is None or model.holdout_auc is None:
        return None
    return round(model.holdout_auc, 2)
