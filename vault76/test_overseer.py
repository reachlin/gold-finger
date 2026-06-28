"""
Tests for overseer.py — Vault 76 market regime classifier.

Run: /Users/lincai/anaconda3/envs/gold-finger/bin/python -m pytest vault76/test_overseer.py -v
"""
import sys, os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_spy(n=120, drift=0.002, daily_vol=0.01, seed=1):
    np.random.seed(seed)
    close = 500 * np.cumprod(1 + drift + np.random.randn(n) * daily_vol)
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open":  close * 0.999,
        "high":  close * 1.008,
        "low":   close * 0.992,
        "close": close,
        "volume": np.ones(n) * 5e7,
    })


def _make_crashing_spy(n=120):
    np.random.seed(3)
    close = 500 * np.cumprod(1 - 0.004 + np.random.randn(n) * 0.015)
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open":  close * 1.001,
        "high":  close * 1.005,
        "low":   close * 0.985,
        "close": close,
        "volume": np.ones(n) * 5e7,
    })


class TestOverseerRegimes:
    def test_reclamation_in_uptrend_low_vix(self):
        from vault76.overseer import Overseer
        o = Overseer()
        spy = _make_spy(drift=0.003)
        assert o.classify(spy, vix=15.0) == Overseer.RECLAMATION

    def test_wasteland_in_downtrend(self):
        from vault76.overseer import Overseer
        o = Overseer()
        spy = _make_crashing_spy()
        assert o.classify(spy, vix=22.0) == Overseer.WASTELAND

    def test_nuked_zone_when_vix_high(self):
        from vault76.overseer import Overseer
        o = Overseer()
        spy = _make_spy()
        assert o.classify(spy, vix=35.0) == Overseer.NUKED_ZONE

    def test_nuked_zone_threshold_at_30(self):
        from vault76.overseer import Overseer
        o = Overseer()
        spy = _make_spy()
        assert o.classify(spy, vix=30.0) == Overseer.NUKED_ZONE

    def test_below_nuked_threshold_not_blocked(self):
        from vault76.overseer import Overseer
        o = Overseer()
        spy = _make_spy()
        assert o.classify(spy, vix=29.9) != Overseer.NUKED_ZONE

    def test_wasteland_when_spy_below_ema50(self):
        from vault76.overseer import Overseer
        o = Overseer()
        # Build uptrend then crash below EMA50
        np.random.seed(7)
        close_up   = 500 * np.cumprod(1 + 0.003 + np.random.randn(100) * 0.008)
        close_down = close_up[-1] * np.cumprod(1 - 0.025 + np.random.randn(30) * 0.01)
        close = np.concatenate([close_up, close_down])
        n = len(close)
        spy = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n, freq="B"),
            "open": close * 0.999, "high": close * 1.005,
            "low": close * 0.990, "close": close,
            "volume": np.ones(n) * 5e7,
        })
        assert o.classify(spy, vix=22.0) == Overseer.WASTELAND

    def test_classify_returns_string(self):
        from vault76.overseer import Overseer
        o = Overseer()
        result = o.classify(_make_spy(), vix=18.0)
        assert isinstance(result, str)
        assert result in (Overseer.RECLAMATION, Overseer.WASTELAND, Overseer.NUKED_ZONE)

    def test_regime_label_is_human_readable(self):
        from vault76.overseer import Overseer
        o = Overseer()
        for regime in (Overseer.RECLAMATION, Overseer.WASTELAND, Overseer.NUKED_ZONE):
            label = o.describe(regime)
            assert isinstance(label, str) and len(label) > 0

    def test_insufficient_spy_data_defaults_to_wasteland(self):
        from vault76.overseer import Overseer
        o = Overseer()
        tiny = _make_spy(n=10)
        assert o.classify(tiny, vix=18.0) == Overseer.WASTELAND


class TestOverseerWeaponSelection:
    def test_scavenger_recommended_in_wasteland(self):
        from vault76.overseer import Overseer
        o = Overseer()
        assert "scavenger" in o.recommend_roles(Overseer.WASTELAND)

    def test_scavenger_recommended_in_reclamation_no_stock(self):
        """Without stock_ind, static mapping applies — backward compat."""
        from vault76.overseer import Overseer
        o = Overseer()
        assert "scavenger" in o.recommend_roles(Overseer.RECLAMATION)

    def test_chemist_active_in_nuked_zone(self):
        from vault76.overseer import Overseer
        o = Overseer()
        assert "chemist" in o.recommend_roles(Overseer.NUKED_ZONE)

    def test_recommend_returns_list(self):
        from vault76.overseer import Overseer
        o = Overseer()
        for regime in (Overseer.RECLAMATION, Overseer.WASTELAND, Overseer.NUKED_ZONE):
            assert isinstance(o.recommend_roles(regime), list)


class TestOverseerStockRouting:
    """Stock-aware routing: Overseer picks role based on per-stock ADX."""

    def test_runner_routes_to_raider_in_reclamation(self):
        """High-ADX stock in RECLAMATION → Raider (ride the trend)."""
        from vault76.overseer import Overseer, ADX_RUNNER
        o = Overseer()
        ind = {"adx": ADX_RUNNER + 5}
        roles = o.recommend_roles(Overseer.RECLAMATION, ind)
        assert "raider" in roles
        assert "scavenger" not in roles

    def test_sideways_routes_to_scavenger_in_reclamation(self):
        """Low-ADX stock in RECLAMATION → Scavenger (collect premium)."""
        from vault76.overseer import Overseer, ADX_RUNNER
        o = Overseer()
        ind = {"adx": ADX_RUNNER - 5}
        roles = o.recommend_roles(Overseer.RECLAMATION, ind)
        assert "scavenger" in roles
        assert "raider" not in roles

    def test_wasteland_always_scavenger(self):
        """WASTELAND → Scavenger regardless of stock ADX (income focus)."""
        from vault76.overseer import Overseer
        o = Overseer()
        for adx in [10, 30, 50]:
            roles = o.recommend_roles(Overseer.WASTELAND, {"adx": adx})
            assert "scavenger" in roles
            assert "raider" not in roles

    def test_nuked_zone_always_chemist(self):
        """NUKED_ZONE → Chemist regardless of stock ADX."""
        from vault76.overseer import Overseer
        o = Overseer()
        for adx in [10, 30, 50]:
            roles = o.recommend_roles(Overseer.NUKED_ZONE, {"adx": adx})
            assert "chemist" in roles
            assert "scavenger" not in roles
            assert "raider" not in roles

    def test_adx_at_threshold_routes_to_raider(self):
        """ADX exactly at ADX_RUNNER → Raider (boundary inclusive)."""
        from vault76.overseer import Overseer, ADX_RUNNER
        o = Overseer()
        roles = o.recommend_roles(Overseer.RECLAMATION, {"adx": ADX_RUNNER})
        assert "raider" in roles

    def test_backward_compat_no_stock_ind(self):
        """No stock_ind → static mapping unchanged."""
        from vault76.overseer import Overseer
        o = Overseer()
        roles = o.recommend_roles(Overseer.RECLAMATION)
        assert "scavenger" in roles and "raider" in roles
