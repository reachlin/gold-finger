"""
The Maggie — Vault 76 Role #003

Named for Qullamaggie, the swing trader whose breakout methodology this
role is built on: https://qullamaggie.com/

Wasteland runners don't wander aimlessly — the strongest ones make a hard
sprint, pause to catch their breath in a tight defensive huddle, then break
for open ground the instant the coast is clear. The Maggie waits for that
exact moment.

Strategy: Breakout
  - Prior run-up: stock already up >= 25% within the last ~90 bars, before
    the current consolidation began (Qullamaggie: "up 30-100%+ over 1-3
    months")
  - Tight consolidation: last 15 bars show higher lows, a contracting daily
    range (ADR% shrinking), and price surfing the rising EMA20
  - Breakout trigger: close breaks above the prior 20-day high on a volume
    surge (vol_ratio > 1.4) — range expansion after the base
  - Stop: capped at the tighter of ATR or ADR% of entry — "stop should not
    be wider than the ATR or ADR of the stock"
  - Target: a simple R-multiple off that stop distance; trail the runner on
    a 10/20-day EMA close-below once it's working (handled by the backtest/
    live position manager, not by scan() itself)

Optimal regimes: RECLAMATION only — Qullamaggie: setups "work best in
bullish markets"; sit out corrections and bear markets.
Avoid: WASTELAND, NUKED_ZONE
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from vault76.armory.base import Role
from vault76.overseer import Overseer
from schwab.trend_scanner import (
    detect_prior_runup, detect_tight_consolidation, detect_breakout_trigger,
    compute_breakout_levels,
)

# ── Maggie parameters ─────────────────────────────────────────────────────
R_MULTIPLE = 3.0   # initial target as a multiple of the stop distance


class Maggie(Role):
    codename        = "maggie"
    name            = "The Maggie"
    optimal_regimes = [Overseer.RECLAMATION]

    r_multiple = R_MULTIPLE

    def scan(self, symbol: str, df: pd.DataFrame,
             regime: str | None = None) -> dict:
        """
        Run The Maggie breakout signal detection.

        df     : indicators already computed via compute_indicators()
        regime : current Overseer regime — only RECLAMATION deploys
        """
        base = {"symbol": symbol, "signal": "NONE", "entry": None,
                "target": None, "stop": None, "rsi": None,
                "adx": None, "reason": "", "card": self.codename}

        if len(df) < 90:
            base["reason"] = "insufficient data"
            return base

        last = df.iloc[-1]
        base["rsi"]     = round(last["rsi"], 1)
        base["adx"]     = round(last["adx"], 1)
        base["close"]   = round(last["close"], 2)
        base["adr_pct"] = round(last["adr_pct"] * 100, 2)

        if regime is not None and not self.should_deploy(regime):
            base["reason"] = f"role benched in {regime}"
            return base

        if not detect_prior_runup(df):
            base["reason"] = "no qualifying prior run-up"
            return base

        if not detect_tight_consolidation(df):
            base["reason"] = "no tight consolidation (higher lows + contraction + EMA20 surf)"
            return base

        if not detect_breakout_trigger(df):
            base["reason"] = "no breakout (price/volume) yet"
            return base

        levels = compute_breakout_levels(
            float(last["close"]), float(last["atr"]), float(last["adr_pct"]),
            self.r_multiple,
        )
        base.update({
            "signal":      "BUY",
            "entry":       levels["entry"],
            "target":      levels["target"],
            "stop":        levels["stop"],
            "risk_reward": levels["risk_reward"],
            "reason":      "run-up+consolidation+breakout confirmed (The Maggie)",
        })
        return base
