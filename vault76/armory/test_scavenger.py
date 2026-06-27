"""
Tests for armory/scavenger.py — The Scavenger perk card (wheel strategy).

Run: /Users/lincai/anaconda3/envs/gold-finger/bin/python -m pytest vault76/armory/test_scavenger.py -v
"""
import sys, os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _make_sideways_df(n=120, seed=5):
    """Stock going nowhere — low ADX, RSI near 50, decent IV."""
    np.random.seed(seed)
    close = 150 + np.random.randn(n) * 3.0
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open":  close * 0.999,
        "high":  close * 1.015,
        "low":   close * 0.985,
        "close": close,
        "volume": np.ones(n) * 8_000_000,
    })


def _make_trending_df(n=120, drift=0.003, seed=1):
    """Stock in a strong uptrend — Raiders should take this."""
    np.random.seed(seed)
    close = 100 * np.cumprod(1 + drift + np.random.randn(n) * 0.01)
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open":  close * 0.995,
        "high":  close * 1.015,
        "low":   close * 0.985,
        "close": close,
        "volume": np.random.default_rng(seed).integers(10_000_000, 20_000_000, n).astype(float),
    })


class TestScavengerIdentity:
    def test_codename(self):
        from vault76.armory.scavenger import Scavenger
        assert Scavenger().codename == "scavenger"

    def test_name_contains_scavenger(self):
        from vault76.armory.scavenger import Scavenger
        assert "Scavenger" in Scavenger().name

    def test_optimal_regimes_include_wasteland(self):
        from vault76.armory.scavenger import Scavenger
        from vault76.overseer import Overseer
        assert Overseer.WASTELAND in Scavenger().optimal_regimes

    def test_should_deploy_in_wasteland(self):
        from vault76.armory.scavenger import Scavenger
        from vault76.overseer import Overseer
        assert Scavenger().should_deploy(Overseer.WASTELAND) is True

    def test_should_deploy_in_reclamation(self):
        from vault76.armory.scavenger import Scavenger
        from vault76.overseer import Overseer
        assert Scavenger().should_deploy(Overseer.RECLAMATION) is True

    def test_should_not_deploy_in_nuked_zone(self):
        from vault76.armory.scavenger import Scavenger
        from vault76.overseer import Overseer
        assert Scavenger().should_deploy(Overseer.NUKED_ZONE) is False


class TestScavengerSellPut:
    def test_scan_returns_dict(self):
        from vault76.armory.scavenger import Scavenger
        from schwab.trend_scanner import compute_indicators
        df  = compute_indicators(_make_sideways_df()).dropna().reset_index(drop=True)
        res = Scavenger().scan("TEST", df)
        assert isinstance(res, dict)
        assert "signal" in res

    def test_no_signal_in_nuked_zone(self):
        from vault76.armory.scavenger import Scavenger
        from vault76.overseer import Overseer
        from schwab.trend_scanner import compute_indicators
        df  = compute_indicators(_make_sideways_df()).dropna().reset_index(drop=True)
        res = Scavenger().scan("TEST", df, regime=Overseer.NUKED_ZONE)
        assert res["signal"] == "NONE"

    def test_sell_put_signal_has_required_fields(self):
        from vault76.armory.scavenger import Scavenger
        from schwab.trend_scanner import compute_indicators
        df  = compute_indicators(_make_sideways_df()).dropna().reset_index(drop=True)
        res = Scavenger().scan("TEST", df)
        if res["signal"] == "SELL_PUT":
            assert res["strike"] is not None
            assert res["premium"] > 0
            assert res["dte"] > 0
            assert res["max_loss"] is not None
            assert res["premium_pct"] > 0

    def test_put_strike_is_below_current_price(self):
        from vault76.armory.scavenger import Scavenger
        from schwab.trend_scanner import compute_indicators
        df  = compute_indicators(_make_sideways_df()).dropna().reset_index(drop=True)
        res = Scavenger().scan("TEST", df)
        if res["signal"] == "SELL_PUT":
            assert res["strike"] < res["close"]

    def test_result_includes_card_name(self):
        from vault76.armory.scavenger import Scavenger
        from schwab.trend_scanner import compute_indicators
        df  = compute_indicators(_make_sideways_df()).dropna().reset_index(drop=True)
        res = Scavenger().scan("TEST", df)
        assert res.get("card") == "scavenger"


class TestScavengerSellCall:
    def test_sell_call_when_cost_basis_provided(self):
        from vault76.armory.scavenger import Scavenger
        from schwab.trend_scanner import compute_indicators
        df    = compute_indicators(_make_sideways_df()).dropna().reset_index(drop=True)
        close = float(df.iloc[-1]["close"])
        res   = Scavenger().scan("TEST", df, cost_basis=close * 0.95)
        assert res["signal"] in ("SELL_CALL", "NONE")

    def test_sell_call_strike_above_cost_basis(self):
        from vault76.armory.scavenger import Scavenger
        from schwab.trend_scanner import compute_indicators
        df    = compute_indicators(_make_sideways_df()).dropna().reset_index(drop=True)
        close = float(df.iloc[-1]["close"])
        cost  = close * 0.95
        res   = Scavenger().scan("TEST", df, cost_basis=cost)
        if res["signal"] == "SELL_CALL":
            assert res["strike"] > cost

    def test_sell_call_premium_positive(self):
        from vault76.armory.scavenger import Scavenger
        from schwab.trend_scanner import compute_indicators
        df    = compute_indicators(_make_sideways_df()).dropna().reset_index(drop=True)
        close = float(df.iloc[-1]["close"])
        res   = Scavenger().scan("TEST", df, cost_basis=close * 0.95)
        if res["signal"] == "SELL_CALL":
            assert res["premium"] > 0

    def test_no_sell_call_when_deeply_underwater(self):
        """Don't sell a call when position is too far underwater — let it recover."""
        from vault76.armory.scavenger import Scavenger
        from schwab.trend_scanner import compute_indicators
        df    = compute_indicators(_make_sideways_df()).dropna().reset_index(drop=True)
        close = float(df.iloc[-1]["close"])
        res   = Scavenger().scan("TEST", df, cost_basis=close * 1.30)
        assert res["signal"] == "NONE"

    def test_no_sell_call_in_nuked_zone(self):
        from vault76.armory.scavenger import Scavenger
        from vault76.overseer import Overseer
        from schwab.trend_scanner import compute_indicators
        df    = compute_indicators(_make_sideways_df()).dropna().reset_index(drop=True)
        close = float(df.iloc[-1]["close"])
        res   = Scavenger().scan("TEST", df, regime=Overseer.NUKED_ZONE,
                                 cost_basis=close * 0.95)
        assert res["signal"] == "NONE"


class TestScavengerParameters:
    def test_otm_put_pct_is_conservative(self):
        """Put sold at least 3% OTM — avoid getting assigned on small dips."""
        from vault76.armory.scavenger import Scavenger
        assert Scavenger().otm_put_pct >= 0.03

    def test_otm_call_pct_is_conservative(self):
        """Call sold at least 5% OTM — don't cap upside too aggressively."""
        from vault76.armory.scavenger import Scavenger
        assert Scavenger().otm_call_pct >= 0.05

    def test_dte_is_reasonable(self):
        from vault76.armory.scavenger import Scavenger
        assert 21 <= Scavenger().sell_dte <= 45
