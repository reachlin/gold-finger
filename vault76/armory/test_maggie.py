"""
Tests for armory/maggie.py — The Maggie role.

Run: /opt/miniconda3/envs/trader/bin/python -m pytest vault76/armory/test_maggie.py -v
"""
import sys, os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _make_breakout_df(runup_pct=0.35, breakout_pct=0.08, breakout_vol_mult=3.0,
                       runup_bars=80, consolidation_bars=15,
                       shrink_from=0.03, shrink_to=0.01, prehistory_bars=60):
    """Run-up, then a tightening consolidation, then a range-expansion
    breakout on a volume surge — the Qullamaggie Breakout setup."""
    close, high, low, volume = [], [], [], []

    for _ in range(prehistory_bars):
        close.append(100.0); high.append(101.0); low.append(99.0); volume.append(5_000_000)

    base_start, base_end = 100.0, 100.0 * (1 + runup_pct)
    for i in range(runup_bars):
        frac = i / (runup_bars - 1)
        c = base_start * (base_end / base_start) ** frac
        close.append(c); high.append(c * 1.02); low.append(c * 0.98); volume.append(5_000_000)

    for i in range(consolidation_bars):
        frac = i / (consolidation_bars - 1)
        rng = shrink_from - (shrink_from - shrink_to) * frac
        center = base_end * (1 + 0.003 * frac)
        close.append(center)
        high.append(center * (1 + rng / 2))
        low.append(center * (1 - rng / 2))
        volume.append(4_000_000)

    last_center = close[-1]
    close.append(last_center * 1.002); high.append(last_center * 1.008)
    low.append(last_center * 0.996); volume.append(4_200_000)

    breakout_close = close[-1] * (1 + breakout_pct)
    avg_recent_vol = sum(volume[-14:]) / 14
    close.append(breakout_close)
    high.append(breakout_close * 1.01)
    low.append(close[-2] * 1.0)
    volume.append(avg_recent_vol * breakout_vol_mult)

    n = len(close)
    return pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=n, freq="B"),
        "open": low, "high": high, "low": low, "close": close, "volume": volume,
    })


def _make_flat_df(n=160, seed=2):
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


class TestMaggieIdentity:
    def test_codename(self):
        from vault76.armory.maggie import Maggie
        assert Maggie().codename == "maggie"

    def test_name_contains_maggie(self):
        from vault76.armory.maggie import Maggie
        assert "Maggie" in Maggie().name

    def test_optimal_regime_is_reclamation(self):
        from vault76.armory.maggie import Maggie
        from vault76.overseer import Overseer
        assert Maggie().optimal_regimes == [Overseer.RECLAMATION]

    def test_should_deploy_in_reclamation(self):
        from vault76.armory.maggie import Maggie
        from vault76.overseer import Overseer
        assert Maggie().should_deploy(Overseer.RECLAMATION) is True

    def test_should_not_deploy_in_wasteland(self):
        from vault76.armory.maggie import Maggie
        from vault76.overseer import Overseer
        assert Maggie().should_deploy(Overseer.WASTELAND) is False

    def test_should_not_deploy_in_nuked_zone(self):
        from vault76.armory.maggie import Maggie
        from vault76.overseer import Overseer
        assert Maggie().should_deploy(Overseer.NUKED_ZONE) is False


class TestMaggieScan:
    def test_buy_on_qualifying_breakout(self):
        from vault76.armory.maggie import Maggie
        from schwab.trend_scanner import compute_indicators
        df = compute_indicators(_make_breakout_df()).dropna().reset_index(drop=True)
        res = Maggie().scan("TEST", df)
        assert res["signal"] == "BUY"

    def test_no_signal_on_flat_market(self):
        from vault76.armory.maggie import Maggie
        from schwab.trend_scanner import compute_indicators
        df = compute_indicators(_make_flat_df()).dropna().reset_index(drop=True)
        res = Maggie().scan("TEST", df)
        assert res["signal"] == "NONE"

    def test_no_signal_before_breakout_bar(self):
        """Consolidating but hasn't broken out yet -> NONE."""
        from vault76.armory.maggie import Maggie
        from schwab.trend_scanner import compute_indicators
        full = compute_indicators(_make_breakout_df()).dropna().reset_index(drop=True)
        df = full.iloc[:-1].reset_index(drop=True)
        res = Maggie().scan("TEST", df)
        assert res["signal"] == "NONE"

    def test_no_signal_without_volume_surge(self):
        from vault76.armory.maggie import Maggie
        from schwab.trend_scanner import compute_indicators
        df = compute_indicators(_make_breakout_df(breakout_vol_mult=1.0)).dropna().reset_index(drop=True)
        res = Maggie().scan("TEST", df)
        assert res["signal"] == "NONE"

    def test_no_signal_in_wasteland(self):
        from vault76.armory.maggie import Maggie
        from vault76.overseer import Overseer
        from schwab.trend_scanner import compute_indicators
        df = compute_indicators(_make_breakout_df()).dropna().reset_index(drop=True)
        res = Maggie().scan("TEST", df, regime=Overseer.WASTELAND)
        assert res["signal"] == "NONE"
        assert "benched" in res["reason"]

    def test_no_signal_in_nuked_zone(self):
        from vault76.armory.maggie import Maggie
        from vault76.overseer import Overseer
        from schwab.trend_scanner import compute_indicators
        df = compute_indicators(_make_breakout_df()).dropna().reset_index(drop=True)
        res = Maggie().scan("TEST", df, regime=Overseer.NUKED_ZONE)
        assert res["signal"] == "NONE"

    def test_buy_has_entry_target_stop_below_entry(self):
        from vault76.armory.maggie import Maggie
        from schwab.trend_scanner import compute_indicators
        df = compute_indicators(_make_breakout_df()).dropna().reset_index(drop=True)
        res = Maggie().scan("TEST", df)
        assert res["signal"] == "BUY"
        assert res["entry"] is not None
        assert res["stop"] < res["entry"]
        assert res["target"] > res["entry"]

    def test_stop_not_wider_than_atr_or_adr(self):
        """Stop distance must never exceed both ATR and ADR% of entry —
        Qullamaggie's core risk rule for this setup."""
        from vault76.armory.maggie import Maggie
        from schwab.trend_scanner import compute_indicators
        df = compute_indicators(_make_breakout_df()).dropna().reset_index(drop=True)
        res = Maggie().scan("TEST", df)
        assert res["signal"] == "BUY"
        last = df.iloc[-1]
        stop_distance = res["entry"] - res["stop"]
        # +0.01 tolerance: entry/stop are independently rounded to cents
        assert stop_distance <= float(last["atr"]) + 0.01
        assert stop_distance <= res["entry"] * float(last["adr_pct"]) + 0.01

    def test_result_includes_card_name(self):
        from vault76.armory.maggie import Maggie
        from schwab.trend_scanner import compute_indicators
        df = compute_indicators(_make_breakout_df()).dropna().reset_index(drop=True)
        res = Maggie().scan("TEST", df)
        assert res.get("card") == "maggie"

    def test_insufficient_data_returns_none(self):
        from vault76.armory.maggie import Maggie
        df = pd.DataFrame({
            "datetime": pd.date_range("2025-01-01", periods=10, freq="B"),
            "open": [100.0] * 10, "high": [101.0] * 10,
            "low": [99.0] * 10, "close": [100.0] * 10, "volume": [1_000_000.0] * 10,
        })
        res = Maggie().scan("TEST", df)
        assert res["signal"] == "NONE"
        assert res["reason"] == "insufficient data"
