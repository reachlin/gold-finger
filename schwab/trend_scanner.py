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

TREND_ADX_MIN    = 25      # ADX threshold for confirmed trend
PULLBACK_RSI_LO  = 33      # RSI lower bound for pullback zone
PULLBACK_RSI_HI  = 57      # RSI upper bound for pullback zone
PULLBACK_EMA_PCT = 0.05    # price must be within 5% of EMA20
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
    """True if last row is in a confirmed uptrend."""
    last = df.iloc[-1]
    return bool(
        last["ema20"] > last["ema50"]        # short MA above long MA
        and last["adx"] > TREND_ADX_MIN      # trend has strength
        and last["close"] > last["ema50"]    # price above long MA
    )


def detect_pullback(df: pd.DataFrame) -> bool:
    """True if price has pulled back to EMA20 zone with RSI cooling off."""
    last = df.iloc[-1]
    price      = last["close"]
    ema20      = last["ema20"]
    rsi        = last["rsi"]
    pct_from_ema20 = abs(price - ema20) / ema20

    # RSI dipped into pullback zone
    rsi_in_zone = PULLBACK_RSI_LO < rsi < PULLBACK_RSI_HI

    # Price near or touched EMA20 recently (last 3 bars)
    recent_low = df["low"].iloc[-3:].min()
    touched_ema20 = recent_low <= ema20 * 1.02

    # Pullback on declining volume (sellers losing steam)
    vol_declining = last["vol_ratio"] < 1.1

    return bool(rsi_in_zone and touched_ema20 and vol_declining)


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
