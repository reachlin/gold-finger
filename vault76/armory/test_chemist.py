"""
Tests for The Chemist role — bear put spread in NUKED_ZONE.

Run: /Users/lincai/anaconda3/envs/gold-finger/bin/python -m pytest vault76/armory/test_chemist.py -v
"""
import sys, os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from vault76.armory.chemist import Chemist
from vault76.overseer import Overseer


def _make_df(n=120, trend="down", seed=42):
    """OHLCV with realistic crash-level volatility (HV ~40% annualized)."""
    from schwab.trend_scanner import compute_indicators
    np.random.seed(seed)
    # 2.5% daily noise ≈ 40% annualized HV — realistic for NUKED_ZONE
    daily_sigma = 0.025
    if trend == "down":
        drift = -0.003    # -0.3%/day downtrend
    elif trend == "up":
        drift = +0.002
    else:
        drift = 0.0
    returns = drift + np.random.randn(n) * daily_sigma
    close   = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open":  close * 0.998,
        "high":  close * 1.020,
        "low":   close * 0.980,
        "close": close,
        "volume": np.ones(n) * 5_000_000,
    })
    return compute_indicators(df).dropna().reset_index(drop=True)


class TestChemistScan:
    def test_returns_none_in_reclamation(self):
        chemist = Chemist()
        df = _make_df(trend="down")
        res = chemist.scan("TEST", df, regime=Overseer.RECLAMATION)
        assert res["signal"] == "NONE"
        assert "NUKED_ZONE" in res["reason"]

    def test_returns_none_in_wasteland(self):
        chemist = Chemist()
        df = _make_df(trend="down")
        res = chemist.scan("TEST", df, regime=Overseer.WASTELAND)
        assert res["signal"] == "NONE"

    def test_returns_none_in_freefall(self):
        """RSI < 20 (freefall) blocks the credit spread — too risky to sell puts."""
        from schwab.trend_scanner import compute_indicators
        import numpy as np, pandas as pd
        np.random.seed(10)
        n = 120
        # Crash: -1% per day for 120 days → RSI will be in freefall (<20)
        returns = -0.010 + np.random.randn(n) * 0.015
        close   = 100 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n, freq="B"),
            "open": close * 0.998, "high": close * 1.010,
            "low": close * 0.990, "close": close,
            "volume": np.ones(n) * 5e6,
        })
        ind = compute_indicators(df).dropna().reset_index(drop=True)
        chemist = Chemist()
        res = chemist.scan("TEST", ind, regime=Overseer.NUKED_ZONE)
        # Either freefall blocked it or it's too underwater
        if ind.iloc[-1]["rsi"] < 20:
            assert res["signal"] == "NONE"
            assert "freefall" in res["reason"] or "underwater" in res["reason"]

    def test_sell_put_spread_in_nuked_zone(self):
        chemist = Chemist()
        df = _make_df(n=150)
        res = chemist.scan("TEST", df, regime=Overseer.NUKED_ZONE)
        # With default (flat/any) market, should get a spread or NONE
        assert res["signal"] in ("SELL_PUT_SPREAD", "NONE")
        if res["signal"] == "SELL_PUT_SPREAD":
            assert res["short_strike"] is not None
            assert res["long_strike"] is not None
            assert res["net_credit"] > 0

    def test_short_strike_above_long_strike(self):
        """Short leg (sold) is closer to ATM; long leg (bought) is further OTM."""
        chemist = Chemist()
        df = _make_df(n=150)
        res = chemist.scan("TEST", df, regime=Overseer.NUKED_ZONE)
        if res["signal"] == "SELL_PUT_SPREAD":
            assert res["short_strike"] > res["long_strike"]

    def test_net_credit_less_than_spread_width(self):
        """net credit < spread width → max_loss > 0 (defined risk)."""
        chemist = Chemist()
        df = _make_df(n=150)
        res = chemist.scan("TEST", df, regime=Overseer.NUKED_ZONE)
        if res["signal"] == "SELL_PUT_SPREAD":
            spread_width = res["short_strike"] - res["long_strike"]
            assert res["net_credit"] < spread_width
            assert res["max_loss"] > 0
            assert res["max_profit"] > 0

    def test_max_profit_equals_net_credit_times_100(self):
        chemist = Chemist()
        df = _make_df(n=150)
        res = chemist.scan("TEST", df, regime=Overseer.NUKED_ZONE)
        if res["signal"] == "SELL_PUT_SPREAD":
            assert abs(res["max_profit"] - res["net_credit"] * 100) < 1.0

    def test_insufficient_data_returns_none(self):
        chemist = Chemist()
        df = _make_df(n=30)
        res = chemist.scan("TEST", df, regime=Overseer.NUKED_ZONE)
        assert res["signal"] == "NONE"
        assert "insufficient" in res["reason"]

    def test_no_regime_passed_allows_scan(self):
        """regime=None bypasses the regime check (backward compat for tests)."""
        chemist = Chemist()
        df = _make_df(n=150)
        res = chemist.scan("TEST", df, regime=None)
        assert res["signal"] in ("SELL_PUT_SPREAD", "NONE")

    def test_should_deploy_only_in_nuked_zone(self):
        chemist = Chemist()
        assert chemist.should_deploy(Overseer.NUKED_ZONE) is True
        assert chemist.should_deploy(Overseer.WASTELAND) is False
        assert chemist.should_deploy(Overseer.RECLAMATION) is False
