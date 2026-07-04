"""
The Chemist — Vault 76 Role #003

In Fallout 76, Chemists synthesize chems from dangerous radioactive materials
found in Nuked Zones. High radiation = high-value ingredients. Everyone else
flees; the Chemist gears up and walks in.

Strategy: credit put spread (bull put spread) during NUKED_ZONE
  - Only activates when VIX ≥ 30 (market in panic, IV at extreme levels)
  - SELL a put 8% OTM, BUY a put 18% OTM for protection (10-point wide spread)
  - Net credit = short premium - long premium (we collect, not pay)
  - Profits from: (a) IV crush as VIX reverts, (b) stock staying above short put

Why sell premium in a crash?
  When VIX hits 30-50, implied vol is 2-3× normal. A put that normally
  costs $2 might be priced at $6-8. The Chemist SELLS that inflated premium
  with a defined-risk cap (the long leg). After the panic subsides (VIX drops
  from 40 → 20), that same put is worth $1-2 — we buy it back cheap. This is
  the "IV crush" trade. The Scavenger avoids selling in NUKED_ZONE due to
  naked assignment risk; the Chemist manages that risk with the long leg.

Entry conditions:
  - HV ≥ 20% (enough premium to make the spread worth selling)
  - RSI not in extreme oversold (< 20) — don't sell above a free-fall
  - Stock not already > 30% below EMA50 — don't sell above a crashed stock

Exit conditions (backtest state machine):
  - Profit target: spread value decays to 30% of credit (70% max profit captured)
  - Loss limit: spread value reaches 200% of credit received (cap the bleed)
  - Expiry: pay intrinsic value, keep remainder of credit

Optimal regimes: NUKED_ZONE only
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from vault76.armory.base import Role
from vault76.overseer import Overseer
from schwab.options_pricer import (
    black_scholes_put, historical_vol, RISK_FREE_RATE,
)

SHORT_PUT_OTM = 0.08   # sell put 8% OTM (collect fat premium, high delta)
LONG_PUT_OTM  = 0.18   # buy put 18% OTM (cap risk; 10% wide spread)
SELL_DTE      = 30     # trading days to expiration
MIN_HV        = 0.20   # floor — NUKED_ZONE HV usually 35%+
RSI_FREEFALL  = 20     # don't sell above a stock below RSI 20 (skip freefalls)
UNDERWATER_MAX = 0.30  # don't sell if stock already > 30% below EMA50


class Chemist(Role):
    codename        = "chemist"
    name            = "The Chemist"
    optimal_regimes = [Overseer.NUKED_ZONE]

    def scan(self, symbol: str, df: pd.DataFrame,
             regime: str | None = None) -> dict:
        """
        Run The Chemist signal detection.

        regime : must be NUKED_ZONE; any other regime returns NONE
        Returns dict with signal="BUY_PUT_SPREAD" or "NONE".
        """
        base = {
            "symbol":       symbol,
            "signal":       "NONE",
            "short_strike": None,   # the one we SELL (higher, closer to ATM)
            "long_strike":  None,   # the one we BUY (lower, protection)
            "net_credit":   None,
            "max_profit":   None,
            "max_loss":     None,
            "hv":           None,
            "rsi":          None,
            "close":        None,
            "dte":          SELL_DTE,
            "reason":       "",
            "card":         self.codename,
        }

        if len(df) < 60:
            base["reason"] = "insufficient data"
            return base

        if regime is not None and not self.should_deploy(regime):
            base["reason"] = f"role only active in NUKED_ZONE, current: {regime}"
            return base

        last  = df.iloc[-1]
        close = float(last["close"])
        rsi   = float(last["rsi"])
        ema50 = float(last["ema50"])
        hv    = historical_vol(df["close"].iloc[-22:])

        base["close"] = round(close, 2)
        base["rsi"]   = round(rsi, 1)
        base["hv"]    = round(hv * 100, 1)

        # Skip stocks in freefall — short put too dangerous even with protection
        if rsi < RSI_FREEFALL:
            base["reason"] = f"RSI {rsi:.1f} < {RSI_FREEFALL} — freefall, skip"
            return base

        below_pct = max((ema50 - close) / ema50, 0.0)
        if below_pct > UNDERWATER_MAX:
            base["reason"] = (f"stock {below_pct*100:.0f}% below EMA50 "
                              f"— too deep in crash, skip")
            return base

        if hv < MIN_HV:
            base["reason"] = f"HV {hv*100:.1f}% < {MIN_HV*100:.0f}% — premium too thin"
            return base

        short_strike = round(close * (1 - SHORT_PUT_OTM), 0)  # sell this (higher)
        long_strike  = round(close * (1 - LONG_PUT_OTM),  0)  # buy this (lower)
        T            = SELL_DTE / 365

        short_premium = black_scholes_put(close, short_strike, T, RISK_FREE_RATE, hv)
        long_premium  = black_scholes_put(close, long_strike,  T, RISK_FREE_RATE, hv)
        net_credit    = short_premium - long_premium   # we COLLECT this

        if net_credit <= 0:
            base["reason"] = "net credit ≤ 0 — spread not worth selling"
            return base

        spread_width = short_strike - long_strike
        max_profit   = net_credit * 100                          # if both expire worthless
        max_loss     = (spread_width - net_credit) * 100        # if stock below long_strike

        base.update({
            "signal":       "SELL_PUT_SPREAD",
            "short_strike": short_strike,
            "long_strike":  long_strike,
            "net_credit":   round(net_credit, 2),
            "max_profit":   round(max_profit, 2),
            "max_loss":     round(max_loss, 2),
            "reason":       (f"NUKED_ZONE credit spread: sell {short_strike} put / "
                             f"buy {long_strike} put, credit ${net_credit:.2f} (The Chemist)"),
        })
        return base
