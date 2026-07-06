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

# ── Breakout / consolidation params (Qullamaggie-style setups) ──────────────
ADR_WINDOW             = 14      # average daily range lookback
BREAKOUT_LOOKBACK       = 20      # N-day high a breakout must clear
RUNUP_LOOKBACK          = 90      # bars searched for the qualifying prior move
RUNUP_MIN_PCT           = 0.25    # min run-up % before a consolidation counts
CONSOLIDATION_LOOKBACK  = 15      # bars checked for tightening range / higher lows
CONSOLIDATION_EMA_PCT   = 0.08    # price must stay within 8% of EMA20 while surfing it
BREAKOUT_VOL_RATIO_MIN  = 1.4     # volume surge required on the breakout bar


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    df["ema10"] = ta.trend.EMAIndicator(close, window=10).ema_indicator()
    df["ema20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()

    adx_ind     = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx"]   = adx_ind.adx()

    df["rsi"]   = ta.momentum.RSIIndicator(close, window=14).rsi()

    atr_ind     = ta.volatility.AverageTrueRange(high, low, close, window=14)
    df["atr"]   = atr_ind.average_true_range()

    vol_sma20   = volume.rolling(20).mean()
    df["vol_ratio"] = volume / vol_sma20.replace(0, np.nan)

    daily_range_pct = (high - low) / close
    df["adr_pct"] = daily_range_pct.rolling(ADR_WINDOW).mean()

    # Prior N-day high, excluding today's own bar — the level a breakout must clear.
    df["high_20"] = high.shift(1).rolling(BREAKOUT_LOOKBACK).max()

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


def detect_market_regime(index_df: pd.DataFrame) -> bool:
    """True if the broad market (SPY/QQQ) is in a healthy uptrend.
    Requires close > EMA50 AND EMA50 is rising — both must hold.
    Pass a df with indicators already computed via compute_indicators().
    """
    last = index_df.iloc[-1]
    prev = index_df.iloc[-6] if len(index_df) > 6 else index_df.iloc[0]
    ema50_rising = last["ema50"] > prev["ema50"]
    return bool(last["close"] > last["ema50"] and ema50_rising)


def compute_levels(entry: float) -> dict:
    target = round(entry * (1 + TARGET_PCT), 2)
    stop   = round(entry * (1 - STOP_PCT), 2)
    rr     = round(TARGET_PCT / STOP_PCT, 2)
    return {"entry": entry, "target": target, "stop": stop, "risk_reward": rr}


def detect_prior_runup(df: pd.DataFrame, lookback: int = RUNUP_LOOKBACK,
                        min_pct: float = RUNUP_MIN_PCT) -> bool:
    """True if the stock ran up at least `min_pct` at some point in the
    `lookback` window before the current consolidation zone (Qullamaggie:
    "stock up 30-100%+ over 1-3 months" ahead of a breakout base).
    """
    start = max(0, len(df) - lookback)
    end   = len(df) - CONSOLIDATION_LOOKBACK
    if end - start < 10:
        return False
    window = df["close"].iloc[start:end]
    run_up = window.max() / window.min() - 1
    return bool(run_up >= min_pct)


def detect_tight_consolidation(df: pd.DataFrame,
                                lookback: int = CONSOLIDATION_LOOKBACK) -> bool:
    """True if the last `lookback` bars show a tightening base: higher lows,
    a contracting daily range (ADR%), and price surfing a rising EMA20 —
    Qullamaggie's "orderly pullback with higher lows and tightening range."
    """
    if len(df) < lookback + 1:
        return False
    window = df.iloc[-lookback:]

    half = lookback // 2
    early_low = window["low"].iloc[:half].min()
    late_low  = window["low"].iloc[half:].min()
    higher_lows = late_low >= early_low * 0.98

    contracting = window["adr_pct"].iloc[-1] < window["adr_pct"].iloc[0]

    ema20 = window["ema20"]
    surfing_ema20 = bool(
        (window["close"] >= ema20 * (1 - CONSOLIDATION_EMA_PCT)).all()
        and ema20.iloc[-1] > ema20.iloc[0]
    )

    return bool(higher_lows and contracting and surfing_ema20)


def detect_breakout_trigger(df: pd.DataFrame,
                             vol_ratio_min: float = BREAKOUT_VOL_RATIO_MIN) -> bool:
    """True if today's bar is a range-expansion breakout: close above the
    prior N-day high on a volume surge.
    """
    last = df.iloc[-1]
    if pd.isna(last["high_20"]) or pd.isna(last["vol_ratio"]):
        return False
    breaks_high  = last["close"] > last["high_20"]
    volume_surge = last["vol_ratio"] > vol_ratio_min
    return bool(breaks_high and volume_surge)


def compute_breakout_levels(entry: float, atr: float, adr_pct: float,
                             r_multiple: float = 3.0) -> dict:
    """Stop distance is capped at the tighter of ATR or ADR% of entry —
    "stop should not be wider than the ATR or ADR of the stock."
    Target is a simple R-multiple; live trailing (10/20-EMA close-below)
    takes over once the position is running.
    """
    stop_distance = min(atr, entry * adr_pct)
    stop_distance = max(stop_distance, entry * 0.01)   # floor — avoid a zero-width stop
    stop   = round(entry - stop_distance, 2)
    target = round(entry + stop_distance * r_multiple, 2)
    return {"entry": round(entry, 2), "target": target, "stop": stop,
            "risk_reward": round(r_multiple, 2)}


def position_size_by_risk(equity: float, risk_pct: float,
                           entry: float, stop: float) -> int:
    """Shares to buy so a stop-out loses exactly `risk_pct` of `equity` —
    Qullamaggie's 0.25-1% account risk per trade discipline.
    """
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0
    dollar_risk = equity * risk_pct
    return max(int(dollar_risk // risk_per_share), 0)


def scan_symbol(symbol: str, df: pd.DataFrame,
                regime_ok: bool = True) -> dict:
    """Run full scan on a prepared (indicators already computed) dataframe.

    regime_ok: pass detect_market_regime(spy_df) result; False blocks all signals.
    """
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

    if not regime_ok:
        base["reason"] = "market regime bearish"
        return base

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
