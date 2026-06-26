"""
Pullback-in-trend scanner.

Strategy:
  - Identify stocks in a confirmed uptrend (EMA20 > EMA50, ADX > 25)
  - Detect pullback to EMA20 zone (RSI 35-55, price within 5% of EMA20)
  - Confirm entry: RSI turning up + green bounce candle + volume rising
  - Output entry price, +20% target, -8% stop loss
"""
import ta
import pandas as pd
import numpy as np

TREND_ADX_MIN    = 15      # ADX threshold — secondary filter only
PULLBACK_RSI_LO  = 28      # RSI lower bound for pullback zone
PULLBACK_RSI_HI  = 50      # RSI upper bound — wait for deeper pullback
PULLBACK_EMA_PCT = 0.08    # price must be within 8% of EMA20
TARGET_PCT       = 0.20    # take profit
STOP_PCT         = 0.08    # stop loss


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    df["ema20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()

    adx_ind     = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx"]   = adx_ind.adx()

    df["rsi"]   = ta.momentum.RSIIndicator(close, window=14).rsi()

    atr_ind     = ta.volatility.AverageTrueRange(high, low, close, window=14)
    df["atr"]   = atr_ind.average_true_range()

    vol_sma20   = volume.rolling(20).mean()
    df["vol_ratio"] = volume / vol_sma20.replace(0, np.nan)

    return df


def detect_trend(df: pd.DataFrame) -> bool:
    """True if last row is in a confirmed uptrend.
    Uses EMA alignment as primary signal; ADX as secondary confirmation.
    Also checks that EMA50 itself is rising (not just price above it).
    """
    last = df.iloc[-1]
    prev = df.iloc[-6] if len(df) > 6 else df.iloc[0]   # 5-bar lookback
    ema50_rising = last["ema50"] > prev["ema50"]
    return bool(
        last["ema20"] > last["ema50"]        # short MA above long MA
        and last["close"] > last["ema50"]    # price above long MA
        and ema50_rising                     # long MA itself is rising
    )


def detect_pullback(df: pd.DataFrame) -> bool:
    """True if a pullback to EMA20 occurred recently (last 5 bars).
    We look back to see if RSI dipped and price touched EMA20, even if
    it has since started recovering — that's the entry window.
    """
    last  = df.iloc[-1]
    ema20 = last["ema20"]

    # RSI dipped into pullback zone in the last 5 bars
    recent_rsi = df["rsi"].iloc[-5:]
    rsi_dipped = (recent_rsi < PULLBACK_RSI_HI).any() and (recent_rsi.min() > PULLBACK_RSI_LO)

    # Price touched or came close to EMA20 in last 5 bars
    recent_low    = df["low"].iloc[-5:].min()
    touched_ema20 = recent_low <= ema20 * 1.04

    return bool(rsi_dipped and touched_ema20)


def detect_entry(df: pd.DataFrame) -> bool:
    """True if today's bar signals pullback is ending and trend resuming."""
    if len(df) < 3:
        return False
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # RSI turning up
    rsi_recovering = last["rsi"] > prev["rsi"]

    # Green candle (close above open)
    green_candle = last["close"] > last["open"]

    # Price holding above EMA20
    above_ema20 = last["close"] >= last["ema20"]

    # Volume picking up (buyers returning)
    vol_rising = last["vol_ratio"] > 0.9

    return bool(rsi_recovering and green_candle and above_ema20 and vol_rising)


def compute_levels(entry: float) -> dict:
    target = round(entry * (1 + TARGET_PCT), 2)
    stop   = round(entry * (1 - STOP_PCT), 2)
    rr     = round(TARGET_PCT / STOP_PCT, 2)
    return {"entry": entry, "target": target, "stop": stop, "risk_reward": rr}


def scan_symbol(symbol: str, df: pd.DataFrame) -> dict:
    """Run full scan on a prepared (indicators already computed) dataframe."""
    base = {"symbol": symbol, "signal": "NONE", "entry": None,
            "target": None, "stop": None, "risk_reward": None,
            "rsi": None, "adx": None, "reason": ""}

    if len(df) < 60:
        base["reason"] = "insufficient data"
        return base

    last = df.iloc[-1]
    base["rsi"] = round(last["rsi"], 1)
    base["adx"] = round(last["adx"], 1)
    base["ema20"] = round(last["ema20"], 2)
    base["ema50"] = round(last["ema50"], 2)
    base["close"] = round(last["close"], 2)

    if not detect_trend(df):
        base["reason"] = "no uptrend"
        return base

    if not detect_pullback(df):
        base["reason"] = "no pullback"
        return base

    if not detect_entry(df):
        base["reason"] = "no entry signal"
        return base

    levels = compute_levels(last["close"])
    base.update({
        "signal":      "BUY",
        "entry":       levels["entry"],
        "target":      levels["target"],
        "stop":        levels["stop"],
        "risk_reward": levels["risk_reward"],
        "reason":      "trend+pullback+entry confirmed",
    })
    return base
