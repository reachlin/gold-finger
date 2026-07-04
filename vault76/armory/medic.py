"""
The Medic — Vault 76 Role #004

When the blast radius is active, the Medic heals the portfolio: quality
dividend ETFs (SCHD, VIG) get bought at panic prices while every other
role is benched or shorting. Panic entries into diversified dividend
payers are contrarian mean-reversion buys — the Medic patches the wound
the crash opened and releases the capital once the market is walking
again.

Strategy: crisis accumulation
  - Entry: Overseer declares NUKED_ZONE (VIX >= 30) and the ETF is not
    already held — buy at the panic close
  - Hold: through WASTELAND — a calmer VIX is not yet a recovery
  - Exit: Overseer declares RECLAMATION (SPY above a rising EMA50, VIX
    calm) — recovery confirmed, capital goes back to the wheel

Optimal regime: NUKED_ZONE (the only role that WANTS the blast radius)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from vault76.armory.base import Role
from vault76.overseer import Overseer

# Quality dividend ETFs the Medic accumulates during panics.
# Roster measured by backtest_medic.py (2026-07-04, 19 episodes 2015-2026):
# VIG +$6,411 (16/19 wins), VYM +$3,580 (16/19), SCHD +$508 (15/19, small
# share price). TLT tested -$2,595 (6/19) — flight-to-safety assets are
# expensive at panic, excluded. GLD positive but gold-bull-dependent.
MEDIC_ETFS = ["SCHD", "VIG", "VYM"]


class Medic(Role):
    codename        = "medic"
    name            = "The Medic"
    optimal_regimes = [Overseer.NUKED_ZONE]

    def scan(self, symbol: str, df: pd.DataFrame,
             regime: str | None = None, holding: bool = False) -> dict:
        """
        df      : daily OHLCV (indicators optional — only close is used)
        regime  : current Overseer regime
        holding : True if this ETF is already in the medic book
        """
        base = {
            "symbol": symbol,
            "signal": "NONE",
            "close":  None,
            "reason": "",
            "card":   self.codename,
        }
        if len(df) < 60:
            base["reason"] = "insufficient data"
            return base

        close = float(df.iloc[-1]["close"])
        base["close"] = round(close, 2)

        if not holding and regime == Overseer.NUKED_ZONE:
            base["signal"] = "BUY_ETF"
            base["reason"] = (f"blast radius active — buy quality dividend "
                              f"ETF {symbol} at panic prices (The Medic)")
        elif holding and regime == Overseer.RECLAMATION:
            base["signal"] = "SELL_ETF"
            base["reason"] = (f"recovery confirmed — release {symbol} "
                              f"capital back to the wheel (The Medic)")
        elif holding:
            base["reason"] = "holding through the wasteland — not recovered yet"
        else:
            base["reason"] = f"no blast radius in {regime} — Medic stands by"
        return base
