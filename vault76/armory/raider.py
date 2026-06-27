"""
The Raider — Vault 76 Perk Card #001

Aggressive opportunist. Attacks when the market shows weakness —
enters on pullbacks in strong uptrends, rides the bounce hard.

Raiders in Fallout 76 are fierce, fast, and opportunistic. They strike
when their target is vulnerable and retreat when the trend breaks.

Strategy: pullback-in-trend
  - Uptrend confirmed (EMA20 > EMA50, EMA50 rising)
  - RSI dips below 47 in last 5 bars (real pullback, not just noise)
  - ADX > 20 (confirmed trend strength — Raiders need a clear target)
  - Entry on bounce: RSI turning up + green candle + above EMA20 + volume returning
  - Exit: trend-end (EMA20 < EMA50) or +5×ATR target — no fixed stop

Optimal regimes: RECLAMATION (primary), WASTELAND (opportunistic)
Avoid: NUKED_ZONE (blast radius — even Raiders know when to run)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from vault76.armory.base import PerkCard
from vault76.overseer import Overseer
from schwab.trend_scanner import (
    detect_trend, compute_levels,
)

# ── Raider parameters (tuned from backtest sweep) ────────────────────────────
RSI_PULLBACK_HI  = 47    # RSI must dip below this (stricter than 50 — real dip only)
RSI_PULLBACK_LO  = 28    # RSI floor — don't attack a stock in freefall
ADX_MIN          = 20    # confirmed trend strength — Raiders pick strong targets
USE_FIXED_STOP   = False # exit on trend-end only — ride the wave


class Raider(PerkCard):
    codename        = "raider"
    name            = "The Raider"
    optimal_regimes = [Overseer.RECLAMATION, Overseer.WASTELAND]

    rsi_pullback_hi = RSI_PULLBACK_HI
    adx_min         = ADX_MIN
    use_fixed_stop  = USE_FIXED_STOP

    def _detect_pullback(self, df: pd.DataFrame) -> bool:
        last       = df.iloc[-1]
        recent_rsi = df["rsi"].iloc[-5:]
        rsi_dipped = ((recent_rsi < self.rsi_pullback_hi).any()
                      and recent_rsi.min() > RSI_PULLBACK_LO)
        touched_ema20 = df["low"].iloc[-5:].min() <= last["ema20"] * 1.04
        return bool(rsi_dipped and touched_ema20)

    def _detect_entry(self, df: pd.DataFrame) -> bool:
        if len(df) < 3:
            return False
        last = df.iloc[-1]
        prev = df.iloc[-2]
        return bool(
            last["rsi"]   > prev["rsi"]
            and last["close"] > last["open"]
            and last["close"] >= last["ema20"]
            and last["vol_ratio"] > 0.9
            and last["adx"] > self.adx_min
        )

    def scan(self, symbol: str, df: pd.DataFrame,
             regime: str | None = None) -> dict:
        """
        Run The Raider signal detection.

        df     : indicators already computed via compute_indicators()
        regime : current Overseer regime — NUKED_ZONE blocks all signals
        """
        base = {"symbol": symbol, "signal": "NONE", "entry": None,
                "target": None, "stop": None, "rsi": None,
                "adx": None, "reason": "", "card": self.codename}

        if len(df) < 60:
            base["reason"] = "insufficient data"
            return base

        last = df.iloc[-1]
        base["rsi"]   = round(last["rsi"], 1)
        base["adx"]   = round(last["adx"], 1)
        base["ema20"] = round(last["ema20"], 2)
        base["ema50"] = round(last["ema50"], 2)
        base["close"] = round(last["close"], 2)

        if regime is not None and not self.should_deploy(regime):
            base["reason"] = f"perk card benched in {regime}"
            return base

        if not detect_trend(df):
            base["reason"] = "no uptrend"
            return base

        if not self._detect_pullback(df):
            base["reason"] = "no pullback (RSI not low enough or price not near EMA20)"
            return base

        if not self._detect_entry(df):
            base["reason"] = "no entry signal"
            return base

        levels = compute_levels(last["close"])
        base.update({
            "signal":      "BUY",
            "entry":       levels["entry"],
            "target":      levels["target"],
            "stop":        levels["stop"],
            "risk_reward": levels["risk_reward"],
            "reason":      "trend+pullback+bounce confirmed (The Raider)",
        })
        return base
