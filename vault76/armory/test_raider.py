"""
Tests for armory/raider.py — The Raider role.

Run: /Users/lincai/anaconda3/envs/gold-finger/bin/python -m pytest vault76/armory/test_raider.py -v
"""
import sys, os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _make_trending_df(n=120, drift=0.002, seed=1):
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


def _make_flat_df(n=120, seed=2):
    np.random.seed(seed)
    close = 100 + np.random.randn(n) * 1.5
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open":  close * 0.999,
        "high":  close * 1.01,
        "low":   close * 0.99,
        "close": close,
        "volume": np.ones(n) * 5_000_000,
    })


class TestRaiderIdentity:
    def test_codename(self):
        from vault76.armory.raider import Raider
        assert Raider().codename == "raider"

    def test_name_contains_raider(self):
        from vault76.armory.raider import Raider
        assert "Raider" in Raider().name

    def test_optimal_regimes_listed(self):
        from vault76.armory.raider import Raider
        from vault76.overseer import Overseer
        assert Overseer.RECLAMATION in Raider().optimal_regimes

    def test_should_deploy_in_wasteland(self):
        from vault76.armory.raider import Raider
        from vault76.overseer import Overseer
        assert Raider().should_deploy(Overseer.WASTELAND) is True

    def test_should_deploy_in_reclamation(self):
        from vault76.armory.raider import Raider
        from vault76.overseer import Overseer
        assert Raider().should_deploy(Overseer.RECLAMATION) is True

    def test_should_not_deploy_in_nuked_zone(self):
        from vault76.armory.raider import Raider
        from vault76.overseer import Overseer
        assert Raider().should_deploy(Overseer.NUKED_ZONE) is False


class TestRaiderScan:
    def test_scan_returns_dict_with_signal(self):
        from vault76.armory.raider import Raider
        from schwab.trend_scanner import compute_indicators
        df  = compute_indicators(_make_trending_df()).dropna().reset_index(drop=True)
        res = Raider().scan("TEST", df)
        assert "signal" in res
        assert res["signal"] in ("BUY", "NONE")

    def test_no_signal_on_flat_market(self):
        from vault76.armory.raider import Raider
        from schwab.trend_scanner import compute_indicators
        df  = compute_indicators(_make_flat_df()).dropna().reset_index(drop=True)
        res = Raider().scan("TEST", df)
        assert res["signal"] == "NONE"

    def test_no_signal_in_nuked_zone(self):
        from vault76.armory.raider import Raider
        from vault76.overseer import Overseer
        from schwab.trend_scanner import compute_indicators
        df  = compute_indicators(_make_trending_df(drift=0.003)).dropna().reset_index(drop=True)
        res = Raider().scan("TEST", df, regime=Overseer.NUKED_ZONE)
        assert res["signal"] == "NONE"

    def test_buy_has_entry_target_stop(self):
        from vault76.armory.raider import Raider
        from schwab.trend_scanner import compute_indicators
        df = compute_indicators(_make_trending_df(n=200, drift=0.003)).dropna().reset_index(drop=True)
        res = Raider().scan("TEST", df)
        if res["signal"] == "BUY":
            assert res["entry"] is not None
            assert res["target"] > res["entry"]
            assert res["stop"]   < res["entry"]

    def test_result_includes_card_name(self):
        from vault76.armory.raider import Raider
        from schwab.trend_scanner import compute_indicators
        df  = compute_indicators(_make_trending_df()).dropna().reset_index(drop=True)
        res = Raider().scan("TEST", df)
        assert res.get("card") == "raider"


class TestRaiderParameters:
    def test_rsi_threshold_is_stricter_than_50(self):
        from vault76.armory.raider import Raider
        assert Raider().rsi_pullback_hi <= 47

    def test_adx_min_is_at_least_20(self):
        from vault76.armory.raider import Raider
        assert Raider().adx_min >= 20

    def test_no_fixed_stop(self):
        from vault76.armory.raider import Raider
        assert Raider().use_fixed_stop is False
