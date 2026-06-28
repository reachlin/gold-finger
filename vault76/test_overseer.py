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
        cards = o.recommend_roles(Overseer.WASTELAND)
        assert "scavenger" in cards

    def test_scavenger_recommended_in_reclamation(self):
        from vault76.overseer import Overseer
        o = Overseer()
        cards = o.recommend_roles(Overseer.RECLAMATION)
        assert "scavenger" in cards

    def test_chemist_active_in_nuked_zone(self):
        from vault76.overseer import Overseer
        o = Overseer()
        roles = o.recommend_roles(Overseer.NUKED_ZONE)
        assert "chemist" in roles

    def test_recommend_returns_list(self):
        from vault76.overseer import Overseer
        o = Overseer()
        for regime in (Overseer.RECLAMATION, Overseer.WASTELAND, Overseer.NUKED_ZONE):
            assert isinstance(o.recommend_roles(regime), list)
