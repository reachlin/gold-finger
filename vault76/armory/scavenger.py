"""
The Scavenger — Vault 76 Role #002

Patient survivor. Finds value in quiet corners of the Wasteland —
generates income by selling options when nothing else is moving.

Scavengers in Fallout 76 don't rush in; they wait, they forage,
they extract yield from situations others overlook.

Strategy: wheel (cash-secured put → covered call)

  Phase 1 — SELL_PUT (cash-secured put):
    - Stock NOT in a strong trend (ADX < 20) — if it were, The Raider takes over
    - RSI neutral (35–65): price is stable, not breaking down
    - Sufficient IV (HV ≥ 20%): premium is worth collecting
    - Sell a put 5% OTM, 30 DTE
    - If assigned: own shares at (strike − premium) effective cost

  Phase 2 — SELL_CALL (covered call, after assignment):
    - Already own shares at cost_basis
    - Stock not in a strong uptrend (don't cap a runner)
    - Price not too far underwater (> 10% below cost basis → wait, don't cap)
    - Sell a call 8% OTM above max(close, cost_basis), 30 DTE

Income goal: ~1–2% per month per position in sideways conditions.

Optimal regimes: WASTELAND (primary), RECLAMATION (on sideways individual stocks)
Avoid: NUKED_ZONE — IV is deceptively high, assignment risk is catastrophic
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from vault76.armory.base import Role
from vault76.overseer import Overseer
from schwab.options_pricer import (
    black_scholes_put, black_scholes_call,
    historical_vol, RISK_FREE_RATE,
)

# ── Scavenger parameters ──────────────────────────────────────────────────────
OTM_PUT_PCT      = 0.05   # sell put 5% below current price
OTM_CALL_PCT     = 0.08   # sell call 8% above cost basis / current (whichever higher)
SELL_DTE         = 30     # days to expiration for both legs
MIN_HV           = 0.20   # minimum historical vol — below this, premium is too thin
MIN_PREMIUM_PCT  = 0.005  # minimum premium as % of stock price (0.5%)
ADX_TREND_MAX    = 20     # don't sell puts if stock is trending (Raider's turf)
RSI_NEUTRAL_LO   = 35     # RSI floor for put-selling (below = falling knife)
RSI_NEUTRAL_HI   = 65     # RSI ceiling for put-selling (above = overbought)
UNDERWATER_MAX   = 0.10   # don't sell call if price is > 10% below cost basis


class Scavenger(Role):
    codename        = "scavenger"
    name            = "The Scavenger"
    optimal_regimes = [Overseer.WASTELAND, Overseer.RECLAMATION]

    otm_put_pct  = OTM_PUT_PCT
    otm_call_pct = OTM_CALL_PCT
    sell_dte     = SELL_DTE

    def scan(self, symbol: str, df: pd.DataFrame,
             regime: str | None = None,
             cost_basis: float | None = None) -> dict:
        """
        Run The Scavenger signal detection.

        df         : indicators already computed via compute_indicators()
        regime     : current Overseer regime — NUKED_ZONE blocks all signals
        cost_basis : if provided, we hold assigned shares at this price
                     → evaluate SELL_CALL opportunity
                     if None → evaluate SELL_PUT opportunity
        """
        base = {
            "symbol":   symbol,
            "signal":   "NONE",
            "strike":   None,
            "premium":  None,
            "premium_pct": None,
            "dte":      SELL_DTE,
            "close":    None,
            "hv":       None,
            "reason":   "",
            "card":     self.codename,
        }

        if len(df) < 60:
            base["reason"] = "insufficient data"
            return base

        if regime is not None and not self.should_deploy(regime):
            base["reason"] = f"role benched in {regime}"
            return base

        last     = df.iloc[-1]
        close    = float(last["close"])
        adx      = float(last["adx"])
        rsi      = float(last["rsi"])
        hv       = historical_vol(df["close"])

        base["close"] = round(close, 2)
        base["hv"]    = round(hv * 100, 1)
        base["rsi"]   = round(rsi, 1)
        base["adx"]   = round(adx, 1)

        if cost_basis is not None:
            return self._covered_call_signal(base, close, adx, hv, cost_basis)
        return self._cash_secured_put_signal(base, close, adx, rsi, hv)

    def _cash_secured_put_signal(self, base: dict, close: float,
                                  adx: float, rsi: float, hv: float) -> dict:
        if adx >= ADX_TREND_MAX:
            base["reason"] = f"ADX {adx:.1f} ≥ {ADX_TREND_MAX} — trending stock, Raider's territory"
            return base

        if not (RSI_NEUTRAL_LO <= rsi <= RSI_NEUTRAL_HI):
            base["reason"] = f"RSI {rsi:.1f} outside neutral range {RSI_NEUTRAL_LO}–{RSI_NEUTRAL_HI}"
            return base

        if hv < MIN_HV:
            base["reason"] = f"HV {hv*100:.1f}% below minimum {MIN_HV*100:.0f}% — premium too thin"
            return base

        strike  = round(close * (1 - OTM_PUT_PCT), 0)
        T       = SELL_DTE / 365
        premium = black_scholes_put(close, strike, T, RISK_FREE_RATE, hv)

        if premium / close < MIN_PREMIUM_PCT:
            base["reason"] = f"premium ${premium:.2f} ({premium/close*100:.2f}%) below minimum"
            return base

        base.update({
            "signal":      "SELL_PUT",
            "strike":      strike,
            "premium":     round(premium, 2),
            "premium_pct": round(premium / close * 100, 2),
            "max_loss":    round((strike - premium) * 100, 2),   # per contract
            "reason":      f"sideways stock — sell {OTM_PUT_PCT*100:.0f}% OTM put for income (The Scavenger)",
        })
        return base

    def _covered_call_signal(self, base: dict, close: float,
                              adx: float, hv: float, cost_basis: float) -> dict:
        underwater_pct = (cost_basis - close) / cost_basis
        if underwater_pct > UNDERWATER_MAX:
            base["reason"] = (f"price ${close:.2f} is {underwater_pct*100:.1f}% below cost "
                              f"${cost_basis:.2f} — wait for recovery")
            return base

        if adx >= ADX_TREND_MAX + 5:   # a little looser — let mild trends run
            base["reason"] = f"ADX {adx:.1f} — strong uptrend, hold shares, don't cap upside"
            return base

        if hv < MIN_HV:
            base["reason"] = f"HV {hv*100:.1f}% too low — call premium not worth selling"
            return base

        reference = max(close, cost_basis)
        strike    = round(reference * (1 + OTM_CALL_PCT), 0)
        T         = SELL_DTE / 365
        premium   = black_scholes_call(close, strike, T, RISK_FREE_RATE, hv)

        if premium / close < MIN_PREMIUM_PCT:
            base["reason"] = f"call premium ${premium:.2f} too thin"
            return base

        base.update({
            "signal":      "SELL_CALL",
            "strike":      strike,
            "premium":     round(premium, 2),
            "premium_pct": round(premium / close * 100, 2),
            "cost_basis":  round(cost_basis, 2),
            "max_gain":    round((strike - cost_basis + premium) * 100, 2),
            "reason":      f"assigned shares — sell {OTM_CALL_PCT*100:.0f}% OTM call for income (The Scavenger)",
        })
        return base
