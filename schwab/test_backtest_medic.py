"""
Tests for backtest_medic.py — buy at NUKED_ZONE entry, hold through
WASTELAND, sell on the first RECLAMATION (recovery) bar.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import backtest_medic as bm


def _frames(n=400):
    """ETF dips then recovers; VIX spikes >=30 in the middle; SPY crashes
    then rebuilds an uptrend so RECLAMATION confirms late in the series."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    t = np.arange(n)
    etf_close = 80 - 10 * np.exp(-((t - 200) / 40.0) ** 2) + t * 0.01
    spy_close = np.concatenate([
        np.linspace(500, 520, 180),          # calm uptrend
        np.linspace(520, 430, 40),           # crash
        np.linspace(430, 560, n - 220),      # strong recovery
    ])
    vix = np.full(n, 15.0)
    vix[180:230] = 45.0                      # blast radius
    mk = lambda c: pd.DataFrame({
        "datetime": dates, "open": c, "high": c * 1.005,
        "low": c * 0.995, "close": c, "volume": np.full(n, 1e6)})
    return mk(etf_close), mk(spy_close), mk(vix)


class TestWalkForwardMedic:
    def test_buys_in_nuke_sells_on_recovery(self):
        etf, spy, vix = _frames()
        events = bm.walk_forward_medic(etf, "SCHD", spy_df=spy, vix_df=vix)
        kinds = [e["event"] for e in events]
        assert "medic_buy" in kinds
        assert "medic_sell" in kinds or "end_liquidate" in kinds
        buy = next(e for e in events if e["event"] == "medic_buy")
        sell = next(e for e in events if e["event"] in ("medic_sell",
                                                        "end_liquidate"))
        assert sell["exit_i"] > buy["entry_i"]

    def test_no_events_without_nuke(self):
        etf, spy, vix = _frames()
        vix["close"] = 15.0                  # never spikes
        events = bm.walk_forward_medic(etf, "SCHD", spy_df=spy, vix_df=vix)
        assert events == []

    def test_pnl_is_share_move(self):
        etf, spy, vix = _frames()
        events = bm.walk_forward_medic(etf, "SCHD", spy_df=spy, vix_df=vix)
        for e in events:
            if e["event"] in ("medic_sell", "end_liquidate"):
                expected = (e["exit_close"] - e["entry_close"]) * bm.SHARES
                assert e["pnl"] == pytest.approx(expected, abs=0.02)
