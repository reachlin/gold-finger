"""
Tests for medic.py — The Medic buys quality dividend ETFs (SCHD, VIG) when
the Overseer declares NUKED_ZONE, and releases them once RECLAMATION
confirms the recovery.
"""
import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from vault76.overseer import Overseer


def _etf(n=120, base=80.0, seed=2):
    rng = np.random.default_rng(seed)
    closes = base * np.cumprod(1 + rng.normal(0, 0.008, n))
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open": closes * 0.999, "high": closes * 1.006,
        "low": closes * 0.994, "close": closes,
        "volume": np.full(n, 2e6),
    })


class TestMedic:
    def _medic(self):
        from vault76.armory.medic import Medic
        return Medic()

    def test_buys_in_nuked_zone_when_flat(self):
        res = self._medic().scan("SCHD", _etf(),
                                 regime=Overseer.NUKED_ZONE, holding=False)
        assert res["signal"] == "BUY_ETF"
        assert res["close"] > 0

    def test_no_rebuy_while_holding(self):
        res = self._medic().scan("SCHD", _etf(),
                                 regime=Overseer.NUKED_ZONE, holding=True)
        assert res["signal"] == "NONE"

    def test_sells_on_reclamation_when_holding(self):
        res = self._medic().scan("SCHD", _etf(),
                                 regime=Overseer.RECLAMATION, holding=True)
        assert res["signal"] == "SELL_ETF"

    def test_holds_through_wasteland(self):
        """WASTELAND is not recovery — keep the position until RECLAMATION."""
        res = self._medic().scan("SCHD", _etf(),
                                 regime=Overseer.WASTELAND, holding=True)
        assert res["signal"] == "NONE"

    def test_no_buy_outside_nuked_zone(self):
        for regime in (Overseer.WASTELAND, Overseer.RECLAMATION):
            res = self._medic().scan("SCHD", _etf(),
                                     regime=regime, holding=False)
            assert res["signal"] == "NONE"

    def test_insufficient_data(self):
        res = self._medic().scan("SCHD", _etf(n=10),
                                 regime=Overseer.NUKED_ZONE, holding=False)
        assert res["signal"] == "NONE"

    def test_etf_watchlist(self):
        from vault76.armory.medic import MEDIC_ETFS
        assert "SCHD" in MEDIC_ETFS and "VIG" in MEDIC_ETFS


class TestOverseerDeploysMedic:
    def test_medic_active_in_nuked_zone(self):
        o = Overseer()
        assert "medic" in o.recommend_roles(Overseer.NUKED_ZONE)
        assert "medic" in o.recommend_roles(Overseer.NUKED_ZONE, {"adx": 50})

    def test_medic_benched_elsewhere(self):
        o = Overseer()
        for regime in (Overseer.RECLAMATION, Overseer.WASTELAND):
            assert "medic" not in o.recommend_roles(regime)
