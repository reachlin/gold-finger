"""
The Responder — Vault 8 Role #001

Weekly swing trader. Uses a BiLSTM range predictor to forecast next week's
(low, high) price band, then places limit orders: BUY near the predicted low,
SELL near the predicted high.

Strategy: weekly range capture → BUY_WEEK_LOW / SELL_WEEK_HIGH
  - Predict next week's (low, high) using a trained BiLSTM model
  - Blend with a HV-based 1-sigma range as a sanity check
  - Gate on market regime (skip NUKED_ZONE; reduce confidence in WASTELAND)
  - Quality filter: only act when the recent base is compressed
    (_tightest_base_range ≤ BASE_RANGE_MAX)
  - Confidence = blend of model capture score, HV alignment, and base quality

Exit rules (within the same week):
  - Take-profit: SELL limit at pred_high
  - Stop-loss:   stop at pred_low × (1 - STOP_PCT)  (below the entry)
  - Time stop:   close by Friday if neither target nor stop hit

Optimal regimes: RECLAMATION (full size), WASTELAND (reduced confidence)
Avoid: NUKED_ZONE

Borrowed from vault76:
  - Role ABC (vault76/armory/base.py)
  - Overseer.classify_row (vault76/overseer.py)
  - historical_vol (schwab/options_pricer.py)
  - _tightest_base_range (vault76/armory/hunter.py)
  - compute_indicators (schwab/trend_scanner.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import torch

from vault76.armory.base import Role
from vault76.overseer import Overseer
from schwab.options_pricer import historical_vol
from schwab.trend_scanner import compute_indicators

# ── Responder parameters ──────────────────────────────────────────────────────

WINDOW          = 8      # weeks of history fed to BiLSTM
BASE_RANGE_MAX  = 12.0   # skip if tightest 5-week range > 12% (too volatile to predict)
HV_BLEND        = 0.3    # weight of HV-baseline range in final prediction (0=BiLSTM only)
STOP_PCT        = 0.015  # stop-loss 1.5% below entry (pred_low)
MIN_RANGE_PCT   = 1.0    # skip if predicted range < 1% (not worth the trade)
WASTELAND_CONF  = 0.6    # multiply confidence by this in WASTELAND regime

# Feature columns must match weekly_range_model.py
FEATURE_COLS = [
    "rsi14", "macd_hist", "boll_pctb", "vol_ratio",
    "roc5", "atr_ratio", "hl_ratio", "ret1w",
]

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "vault8_weekly_range.pt")


# ── Lazy model loader ─────────────────────────────────────────────────────────

_model_cache = None
_device_cache = None


def _load_model():
    global _model_cache, _device_cache
    if _model_cache is not None:
        return _model_cache, _device_cache

    # Import here to avoid circular deps and slow startup
    from vault8.weekly_range_model import WeeklyRangeModel
    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else "cpu"
    )
    model = WeeklyRangeModel()
    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            f"Vault8 model not found at {_MODEL_PATH}. "
            "Run: python vault8/weekly_range_model.py --train"
        )
    model.load_state_dict(torch.load(_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    _model_cache  = model
    _device_cache = device
    return model, device


# ── Weekly feature engineering ────────────────────────────────────────────────

def _resample_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily OHLCV → weekly bars (Friday-anchored)."""
    df = daily_df.copy()
    if "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    weekly = df.resample("W-FRI").agg({
        "open": "first", "high": "max",
        "low":  "min",   "close": "last",
        "volume": "sum",
    }).dropna()
    counts = df["close"].resample("W-FRI").count()
    weekly = weekly[counts >= 3].reset_index()
    return weekly


def _add_weekly_features(wdf: pd.DataFrame) -> pd.DataFrame:
    df = wdf.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd - sig) / c

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["boll_pctb"] = (c - (sma20 - 2 * std20)) / (4 * std20 + 1e-9)

    vol_ma = v.rolling(20).mean()
    df["vol_ratio"] = v / (vol_ma + 1e-9)
    df["roc5"]      = c.pct_change(5)

    tr = pd.concat([h - l,
                    (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr_ratio"] = tr.rolling(14).mean() / c
    df["hl_ratio"]  = (h - l) / c
    df["ret1w"]     = c.pct_change(1)
    return df


def _tightest_base_range(wdf: pd.DataFrame, lookback: int = 8) -> float:
    """5-week rolling range compression in the last `lookback` weekly bars."""
    window = wdf.iloc[-lookback:]
    ranges = []
    for i in range(len(window) - 4):
        chunk = window.iloc[i:i + 5]
        mid   = float(chunk["close"].mean())
        if mid > 0:
            r = (float(chunk["high"].max()) - float(chunk["low"].min())) / mid * 100
            ranges.append(r)
    return round(min(ranges), 1) if ranges else 999.0


# ── HV baseline range ─────────────────────────────────────────────────────────

def _hv_weekly_range(daily_df: pd.DataFrame, close: float) -> tuple[float, float]:
    """
    Estimate next-week (low, high) from 22-day annualised HV.
    Weekly 1-sigma move = HV / sqrt(52) × close.
    Returns (pred_low, pred_high) as absolute prices.
    """
    hv = historical_vol(daily_df["close"], window=22)
    if hv <= 0:
        return close * 0.98, close * 1.02
    weekly_sigma = hv / np.sqrt(52) * close
    return close - weekly_sigma, close + weekly_sigma


# ── Main Role ─────────────────────────────────────────────────────────────────

class Responder(Role):
    codename        = "responder"
    name            = "The Responder"
    optimal_regimes = [Overseer.RECLAMATION, Overseer.WASTELAND]

    def scan(self, symbol: str, df: pd.DataFrame,
             regime: str | None = None) -> dict:
        """
        df: daily OHLCV dataframe (datetime|date, open, high, low, close, volume).
        Returns signal dict.
        """
        none = {"symbol": symbol, "signal": "NONE",
                "role": self.codename, "reason": ""}

        # ── Regime gate ───────────────────────────────────────────────────────
        if regime == Overseer.NUKED_ZONE:
            return {**none, "reason": "NUKED_ZONE — Responder benched"}

        # ── Weekly feature engineering ────────────────────────────────────────
        wdf = _resample_to_weekly(df)
        wdf = _add_weekly_features(wdf)
        clean = wdf.dropna(subset=FEATURE_COLS)

        if len(clean) < WINDOW + 2:
            return {**none, "reason": f"insufficient weekly data ({len(clean)} bars)"}

        last_close = float(clean["close"].iloc[-1])
        if last_close <= 0:
            return {**none, "reason": "invalid close price"}

        # ── Quality filter: base compression ──────────────────────────────────
        base_rng = _tightest_base_range(clean)
        if base_rng > BASE_RANGE_MAX:
            return {**none,
                    "reason": f"base too wide ({base_rng:.1f}% > {BASE_RANGE_MAX}%) — low predictability"}

        # ── BiLSTM prediction ─────────────────────────────────────────────────
        try:
            model, device = _load_model()
        except FileNotFoundError as e:
            return {**none, "reason": str(e)}

        feat = np.clip(clean[FEATURE_COLS].values[-WINDOW:].astype(np.float32),
                       -10, 10)
        X = torch.tensor(feat).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(X).cpu().numpy()[0]

        # pred = [rel_low, rel_high] relative to next week's open ≈ last close
        bilstm_low  = last_close * (1 + float(pred[0]))
        bilstm_high = last_close * (1 + float(pred[1]))

        # Ensure low < high
        if bilstm_low >= bilstm_high:
            bilstm_low, bilstm_high = bilstm_high, bilstm_low

        # ── HV baseline blend ─────────────────────────────────────────────────
        hv_low, hv_high = _hv_weekly_range(df, last_close)
        pred_low  = bilstm_low  * (1 - HV_BLEND) + hv_low  * HV_BLEND
        pred_high = bilstm_high * (1 - HV_BLEND) + hv_high * HV_BLEND

        # Take the tighter (more conservative) range
        pred_low  = max(pred_low,  min(bilstm_low,  hv_low))
        pred_high = min(pred_high, max(bilstm_high, hv_high))

        pred_range_pct = (pred_high - pred_low) / pred_low * 100

        if pred_range_pct < MIN_RANGE_PCT:
            return {**none,
                    "reason": f"predicted range too narrow ({pred_range_pct:.2f}%) — not tradeable"}

        # ── Confidence score ──────────────────────────────────────────────────
        # Base score from predicted range width (more range = more opportunity)
        conf = min(int(pred_range_pct * 10), 80)

        # Boost for tight base (compressed = higher model accuracy)
        if base_rng < 5.0:
            conf = min(conf + 15, 95)
        elif base_rng < 8.0:
            conf = min(conf + 8, 95)

        # HV alignment: if BiLSTM and HV agree on direction, boost confidence
        bilstm_dir = bilstm_high + bilstm_low - 2 * last_close  # net bias
        hv_center  = (hv_high + hv_low) / 2
        if abs(bilstm_dir) < 0.01 * last_close or abs(hv_center - last_close) < 0.005 * last_close:
            conf = min(conf + 5, 95)

        # Wasteland penalty
        if regime == Overseer.WASTELAND:
            conf = int(conf * WASTELAND_CONF)

        stop = round(pred_low * (1 - STOP_PCT), 2)

        return {
            "symbol":       symbol,
            "signal":       "BUY_WEEK_LOW",
            "role":         self.codename,
            "entry":        round(pred_low,  2),
            "target":       round(pred_high, 2),
            "stop":         stop,
            "pred_range_pct": round(pred_range_pct, 2),
            "base_range":   base_rng,
            "last_close":   round(last_close, 2),
            "confidence":   conf,
            "regime":       regime or "UNKNOWN",
            "reason": (
                f"Responder: buy @ ${pred_low:.2f} → sell @ ${pred_high:.2f} "
                f"({pred_range_pct:.1f}% range)  base {base_rng:.1f}%"
            ),
        }
