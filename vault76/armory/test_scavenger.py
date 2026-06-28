"""
Tests for armory/scavenger.py — The Scavenger role (wheel strategy).

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

    def test_wider_call_strike_when_stock_trending(self):
        """High-ADX stock gets a wider call strike — don't cap a runner."""
        import numpy as np
        from vault76.armory.scavenger import Scavenger, ADX_CALL_WIDE
        from schwab.trend_scanner import compute_indicators

        # Build a strongly-trending df so ADX > ADX_CALL_WIDE
        np.random.seed(7)
        n = 200
        close = np.linspace(80, 160, n) + np.random.randn(n) * 0.5  # strong uptrend
        df_trend = pd.DataFrame({
            "datetime": pd.date_range("2022-01-01", periods=n, freq="B"),
            "open":  close * 0.999, "high": close * 1.005,
            "low":   close * 0.995, "close": close,
            "volume": np.ones(n) * 5e6,
        })
        df_trend = compute_indicators(df_trend).dropna().reset_index(drop=True)
        df_side  = compute_indicators(_make_sideways_df()).dropna().reset_index(drop=True)

        close_t = float(df_trend.iloc[-1]["close"])
        close_s = float(df_side.iloc[-1]["close"])

        res_trend = Scavenger().scan("TEST", df_trend, cost_basis=close_t * 0.95)
        res_side  = Scavenger().scan("TEST", df_side,  cost_basis=close_s * 0.95)

        if res_trend["signal"] == "SELL_CALL" and res_side["signal"] == "SELL_CALL":
            assert res_trend["strike"] / close_t > res_side["strike"] / close_s, (
                "Trending stock should get wider OTM call than sideways stock"
            )

    def test_wide_call_otm_pct_larger_than_base(self):
        """OTM_CALL_PCT_RECLAMATION (wide) must exceed OTM_CALL_PCT (base)."""
        from vault76.armory.scavenger import OTM_CALL_PCT_RECLAMATION, OTM_CALL_PCT
        assert OTM_CALL_PCT_RECLAMATION > OTM_CALL_PCT

    def test_no_call_when_stock_in_runaway_trend(self):
        """ADX >= ADX_CALL_BLOCK: hold shares naked — don't cap a runaway runner."""
        import numpy as np
        from vault76.armory.scavenger import Scavenger, ADX_CALL_BLOCK
        from schwab.trend_scanner import compute_indicators

        np.random.seed(11)
        n = 300
        # Extremely strong trend to push ADX well above block threshold
        close = np.linspace(50, 300, n) + np.random.randn(n) * 0.2
        df = pd.DataFrame({
            "datetime": pd.date_range("2022-01-01", periods=n, freq="B"),
            "open":  close * 0.998, "high": close * 1.006,
            "low":   close * 0.994, "close": close,
            "volume": np.ones(n) * 8e6,
        })
        df = compute_indicators(df).dropna().reset_index(drop=True)
        adx_val = float(df.iloc[-1]["adx"])
        if adx_val < ADX_CALL_BLOCK:
            pytest.skip(f"Synthetic df ADX={adx_val:.1f} didn't reach {ADX_CALL_BLOCK} — skip")

        close_last = float(df.iloc[-1]["close"])
        res = Scavenger().scan("TEST", df, cost_basis=close_last * 0.95)
        assert res["signal"] == "NONE", (
            f"Expected NONE for runaway trend (ADX={adx_val:.1f}), got {res['signal']}"
        )


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
