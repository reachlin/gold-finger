"""
The Hunter — Vault 76 Role #005

Patient stalker. Waits for the perfect breakout setup, then strikes with
call options to ride the momentum wave.

Inspired by Tito Adhikary's momentum framework:
  "Right market + right sector + right stock"
  Then buy calls on the breakout, scale out at +50% and +100%.

Strategy: momentum breakout → BUY_CALL
  - Prior run-up: stock up ≥25% in the last 90 days (earned the base)
  - Tight consolidation: price contracts, higher lows, surfing EMA20
  - Breakout trigger: close above 20-day high on volume surge (≥1.4× avg)
  - Market regime: RECLAMATION only (SPY in uptrend — Tito's "right market")
  - ADX ≥ 20 confirms trend strength entering the base

Options structure:
  - ATM call (nearest $5 strike at or above close), ~35 DTE
  - Premium estimated via Black-Scholes on 22-day historical vol
  - Max cost: BUY_CALL_BUDGET per contract ($500 default)

Exit rules (Tito's scaling):
  - +50% gain  → sell 25% of position
  - +100% gain → sell 50% of remaining
  - Trail the rest
  - -50% loss  → full stop (premium halved)

Optimal regimes: RECLAMATION (uptrending market — momentum works)
Avoid: NUKED_ZONE, WASTELAND
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from vault76.armory.base import Role
from vault76.overseer import Overseer
from schwab.trend_scanner import (
    detect_prior_runup,
    detect_tight_consolidation,
    detect_breakout_trigger,
)
from schwab.options_pricer import (
    black_scholes_call,
    historical_vol,
    RISK_FREE_RATE,
)

# ── Hunter parameters ─────────────────────────────────────────────────────────
CALL_DTE          = 35      # target days to expiration
ADX_MIN           = 20      # minimum ADX — trend must exist before the base
RSI_LO            = 35      # avoid falling-knife entries
RSI_HI            = 80      # avoid chasing overbought breakouts (VCP often fires 65-80)
MAX_PREMIUM_PCT   = 0.12    # skip if call costs > 12% of stock price (too expensive)
BUY_CALL_BUDGET   = 500     # max premium per contract in dollars ($5.00/sh × 100)

# Sector ETF map — used by backtest for three-pillar sector check.
# Hunter.scan() does NOT fetch sector data (live_scanner handles that);
# the map is exported so callers can look up the right ETF.
SECTOR_MAP = {
    "NVDA":  "XLK",  "AMD":   "XLK",  "AAPL":  "XLK",  "MSFT":  "XLK",
    "GOOGL": "XLC",  "META":  "XLC",
    "IBM":   "XLK",  "INTC":  "XLK",  "IONQ":  "XLK",
    "AMZN":  "XLY",
    "KO":    "XLP",  "PG":    "XLP",
    "XOM":   "SPY",  "MMM":   "SPY",  "UNH":   "SPY",
}


class Hunter(Role):
    codename        = "hunter"
    name            = "The Hunter"
    optimal_regimes = [Overseer.RECLAMATION]

    adx_min       = ADX_MIN
    call_dte      = CALL_DTE
    max_prem_pct  = MAX_PREMIUM_PCT

    def scan(self, symbol: str, df: pd.DataFrame,
             regime: str | None = None) -> dict:
        """
        Run The Hunter signal detection.

        df     : indicators already computed via compute_indicators()
        regime : current Overseer regime — only RECLAMATION fires signals

        Returns a signal dict.  signal="BUY_CALL" on a valid setup,
        signal="NONE" otherwise.
        """
        base = {
            "symbol":  symbol,
            "signal":  "NONE",
            "strike":  None,
            "premium": None,
            "premium_pct": None,
            "dte":     CALL_DTE,
            "close":   None,
            "hv":      None,
            "adx":     None,
            "rsi":     None,
            "reason":  "",
            "card":    self.codename,
        }

        if len(df) < 100:
            base["reason"] = "insufficient data (need ≥100 bars)"
            return base

        if regime is not None and not self.should_deploy(regime):
            base["reason"] = f"role benched in {regime} — Hunter needs RECLAMATION"
            return base

        last = df.iloc[-1]
        close = float(last["close"])
        adx   = float(last["adx"])
        rsi   = float(last["rsi"])
        hv    = historical_vol(df["close"].iloc[-22:])

        base["close"] = round(close, 2)
        base["adx"]   = round(adx, 1)
        base["rsi"]   = round(rsi, 1)
        base["hv"]    = round(hv * 100, 1)

        # ── Filter: ADX must confirm trend before the base ────────────────────
        if adx < ADX_MIN:
            base["reason"] = f"ADX {adx:.1f} < {ADX_MIN} — no trend to break out from"
            return base

        # ── Filter: RSI guard ─────────────────────────────────────────────────
        if not (RSI_LO <= rsi <= RSI_HI):
            base["reason"] = f"RSI {rsi:.1f} outside [{RSI_LO}–{RSI_HI}]"
            return base

        # ── Three-stage VCP check ─────────────────────────────────────────────
        if not detect_prior_runup(df):
            base["reason"] = "no prior run-up (stock needs ≥25% move in 90 days first)"
            return base

        if not detect_tight_consolidation(df):
            base["reason"] = "no tight consolidation (range not contracting)"
            return base

        if not detect_breakout_trigger(df):
            base["reason"] = "no breakout trigger (price/volume not expanding)"
            return base

        # ── Options pricing ───────────────────────────────────────────────────
        strike  = _atm_strike(close)
        T       = CALL_DTE / 365
        premium = black_scholes_call(close, strike, T, RISK_FREE_RATE, hv)

        if premium <= 0:
            base["reason"] = "Black-Scholes returned zero premium"
            return base

        premium_pct = premium / close * 100
        if premium_pct > MAX_PREMIUM_PCT * 100:
            base["reason"] = (f"premium ${premium:.2f} ({premium_pct:.1f}% of close) "
                              f"too expensive — IV too high for a call buy")
            return base

        cost_per_contract = premium * 100
        if cost_per_contract > BUY_CALL_BUDGET:
            base["reason"] = (f"contract cost ${cost_per_contract:.0f} exceeds "
                              f"budget ${BUY_CALL_BUDGET}")
            return base

        # ── VCP quality metadata ──────────────────────────────────────────────
        tightest_range = _tightest_base_range(df)
        breakout_vol   = round(float(last["vol_ratio"]), 2)

        base.update({
            "signal":       "BUY_CALL",
            "strike":       strike,
            "premium":      round(premium, 2),
            "premium_pct":  round(premium_pct, 2),
            "cost_per_ct":  round(cost_per_contract, 0),
            "vcp_tight_pct": tightest_range,
            "breakout_vol":  breakout_vol,
            # Tito's exit rules
            "exit_25pct":   round(premium * 1.5, 2),   # sell 25% at +50% gain
            "exit_50pct":   round(premium * 2.0, 2),   # sell 50% at +100% gain
            "stop_premium": round(premium * 0.5, 2),   # stop at -50% loss
            "reason": (
                f"VCP breakout: tightest base {tightest_range:.1f}%, "
                f"vol ×{breakout_vol:.1f}, ADX {adx:.0f} — momentum call entry"
            ),
        })
        return base


# ── Helpers ───────────────────────────────────────────────────────────────────

def _atm_strike(close: float) -> float:
    """Round to nearest standard strike: $1 increments ≤$20, $5 above."""
    if close <= 20:
        return round(close)
    return round(close / 5) * 5


def _tightest_base_range(df: pd.DataFrame,
                          lookback: int = 15) -> float:
    """
    Tightest 5-bar rolling price range (%) in the last `lookback` bars.
    Lower = more compressed base = higher quality VCP.
    """
    window = df.iloc[-lookback:]
    ranges = []
    for i in range(len(window) - 4):
        chunk = window.iloc[i:i+5]
        mid   = float(chunk["close"].mean())
        if mid > 0:
            r = (float(chunk["high"].max()) - float(chunk["low"].min())) / mid * 100
            ranges.append(r)
    return round(min(ranges), 1) if ranges else 0.0
