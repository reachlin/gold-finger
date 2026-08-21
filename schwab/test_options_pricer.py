"""
Tests for options_pricer.py — Black-Scholes put pricing utilities.

Run: /Users/lincai/anaconda3/envs/gold-finger/bin/python -m pytest schwab/test_options_pricer.py -v
"""
import sys, os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))


class TestBlackScholesPut:
    def test_atm_put_positive(self):
        from options_pricer import black_scholes_put
        p = black_scholes_put(S=100, K=100, T=30/365, r=0.05, sigma=0.50)
        assert p > 0

    def test_deep_itm_put_near_intrinsic(self):
        from options_pricer import black_scholes_put
        # S=80, K=100 → deep ITM, value should be close to intrinsic $20
        p = black_scholes_put(S=80, K=100, T=1/365, r=0.05, sigma=0.50)
        assert p == pytest.approx(20.0, abs=0.5)

    def test_deep_otm_put_near_zero(self):
        from options_pricer import black_scholes_put
        p = black_scholes_put(S=150, K=100, T=7/365, r=0.05, sigma=0.40)
        assert p < 0.10

    def test_expired_put_is_intrinsic(self):
        from options_pricer import black_scholes_put
        assert black_scholes_put(S=90, K=100, T=0, r=0.05, sigma=0.50) == pytest.approx(10.0)
        assert black_scholes_put(S=110, K=100, T=0, r=0.05, sigma=0.50) == pytest.approx(0.0)

    def test_longer_dte_more_expensive(self):
        from options_pricer import black_scholes_put
        p7  = black_scholes_put(S=100, K=100, T=7/365,  r=0.05, sigma=0.50)
        p21 = black_scholes_put(S=100, K=100, T=21/365, r=0.05, sigma=0.50)
        assert p21 > p7

    def test_higher_vol_more_expensive(self):
        from options_pricer import black_scholes_put
        p_lo = black_scholes_put(S=100, K=100, T=21/365, r=0.05, sigma=0.30)
        p_hi = black_scholes_put(S=100, K=100, T=21/365, r=0.05, sigma=0.70)
        assert p_hi > p_lo

    def test_never_negative(self):
        from options_pricer import black_scholes_put
        for S in [50, 100, 150, 200]:
            for T in [0, 1/365, 7/365, 30/365]:
                assert black_scholes_put(S=S, K=100, T=T, r=0.05, sigma=0.50) >= 0


class TestHistoricalVol:
    def _make_prices(self, n=60, daily_vol=0.02):
        np.random.seed(42)
        returns = np.random.randn(n) * daily_vol
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.Series(prices)

    def test_returns_float(self):
        from options_pricer import historical_vol
        prices = self._make_prices()
        vol = historical_vol(prices)
        assert isinstance(vol, float)

    def test_annualized_reasonable(self):
        from options_pricer import historical_vol
        # daily_vol=2% → annualized ~32%
        prices = self._make_prices(daily_vol=0.02)
        vol = historical_vol(prices)
        assert 0.20 < vol < 0.60

    def test_higher_vol_series_gives_higher_vol(self):
        from options_pricer import historical_vol
        lo = historical_vol(self._make_prices(daily_vol=0.01))
        hi = historical_vol(self._make_prices(daily_vol=0.04))
        assert hi > lo

    def test_needs_minimum_prices(self):
        from options_pricer import historical_vol
        short = pd.Series([100.0, 101.0, 102.0])
        assert historical_vol(short) == 0.0


class TestAtmStrike:
    def test_rounds_to_nearest_5(self):
        from options_pricer import atm_strike
        assert atm_strike(132.5) == 130.0
        assert atm_strike(133.0) == 135.0
        assert atm_strike(137.4) == 135.0
        assert atm_strike(137.6) == 140.0

    def test_custom_step(self):
        from options_pricer import atm_strike
        assert atm_strike(103.0, step=10.0) == 100.0
        assert atm_strike(106.0, step=10.0) == 110.0


class TestSimulatePutTrade:
    """simulate_put_trade(future_df, entry_S, K, sigma, r, dte, max_hold, profit_target_mult)"""

    def _make_falling_df(self, n=15, start=130.0, end=115.0):
        closes = np.linspace(start, end, n)
        return pd.DataFrame({
            "open":  closes * 0.999,
            "high":  closes * 1.005,
            "low":   closes * 0.990,
            "close": closes,
        })

    def _make_rising_df(self, n=15, start=130.0, end=150.0):
        closes = np.linspace(start, end, n)
        return pd.DataFrame({
            "open":  closes * 1.001,
            "high":  closes * 1.015,
            "low":   closes * 0.998,
            "close": closes,
        })

    def test_put_profits_when_stock_falls(self):
        from options_pricer import simulate_put_trade
        df = self._make_falling_df()
        result = simulate_put_trade(df, entry_S=130.0, K=130.0, sigma=0.60,
                                    r=0.05, dte=21, max_hold=15)
        assert result["pnl_dollar"] > 0

    def test_put_loses_when_stock_rallies(self):
        from options_pricer import simulate_put_trade
        df = self._make_rising_df()
        result = simulate_put_trade(df, entry_S=130.0, K=130.0, sigma=0.60,
                                    r=0.05, dte=21, max_hold=15)
        assert result["pnl_dollar"] < 0

    def test_profit_target_exits_early(self):
        from options_pricer import simulate_put_trade
        df = self._make_falling_df(n=15, start=130.0, end=100.0)
        result = simulate_put_trade(df, entry_S=130.0, K=130.0, sigma=0.60,
                                    r=0.05, dte=21, max_hold=15,
                                    profit_target_mult=1.5)
        assert result["exit_reason"] == "target"
        assert result["hold_days"] < 14

    def test_new_high_exits_early(self):
        from options_pricer import simulate_put_trade
        df = self._make_rising_df(n=15, start=130.0, end=145.0)
        result = simulate_put_trade(df, entry_S=130.0, K=130.0, sigma=0.60,
                                    r=0.05, dte=21, max_hold=15)
        assert result["exit_reason"] == "new_high"

    def test_timeout(self):
        from options_pricer import simulate_put_trade
        closes = np.full(10, 128.0)
        df = pd.DataFrame({"open": closes, "high": closes*1.002,
                           "low": closes*0.998, "close": closes})
        result = simulate_put_trade(df, entry_S=130.0, K=130.0, sigma=0.50,
                                    r=0.05, dte=21, max_hold=10)
        assert result["exit_reason"] == "timeout"

    def test_result_has_required_keys(self):
        from options_pricer import simulate_put_trade
        df = self._make_falling_df()
        result = simulate_put_trade(df, entry_S=130.0, K=130.0, sigma=0.60,
                                    r=0.05, dte=21, max_hold=15)
        for k in ("exit_reason", "hold_days", "entry_put_price",
                  "exit_put_price", "pnl_dollar", "pnl_pct"):
            assert k in result


class TestYangZhangVol:
    def _make_ohlc(self, n=60, daily_vol=0.02, seed=42):
        np.random.seed(seed)
        closes = 100 * np.exp(np.cumsum(np.random.randn(n) * daily_vol))
        opens  = closes * np.exp(np.random.randn(n) * 0.005)
        highs  = np.maximum(opens, closes) * np.exp(np.abs(np.random.randn(n) * 0.003))
        lows   = np.minimum(opens, closes) * np.exp(-np.abs(np.random.randn(n) * 0.003))
        return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})

    def test_returns_float(self):
        from options_pricer import yang_zhang_vol
        df  = self._make_ohlc()
        vol = yang_zhang_vol(df)
        assert isinstance(vol, float)

    def test_annualized_reasonable(self):
        from options_pricer import yang_zhang_vol
        # daily_vol=2% → annualized ~32%
        vol = yang_zhang_vol(self._make_ohlc(daily_vol=0.02))
        assert 0.10 < vol < 0.80

    def test_higher_vol_series_gives_higher_vol(self):
        from options_pricer import yang_zhang_vol
        lo = yang_zhang_vol(self._make_ohlc(daily_vol=0.01, seed=1))
        hi = yang_zhang_vol(self._make_ohlc(daily_vol=0.04, seed=1))
        assert hi > lo

    def test_needs_minimum_bars(self):
        from options_pricer import yang_zhang_vol
        short = pd.DataFrame({"open": [100]*5, "high": [101]*5,
                               "low": [99]*5,  "close": [100]*5})
        assert yang_zhang_vol(short) == 0.0

    def test_different_from_close_only_vol(self):
        from options_pricer import yang_zhang_vol, historical_vol
        df  = self._make_ohlc(daily_vol=0.02)
        yz  = yang_zhang_vol(df)
        hv  = historical_vol(df["close"])
        # They should be in the same ballpark but not identical
        assert abs(yz - hv) < 0.30


class TestImpliedVol:
    def test_roundtrip_call(self):
        from options_pricer import black_scholes_call, implied_vol
        S, K, T, r, sigma = 100.0, 100.0, 30/365, 0.05, 0.30
        price = black_scholes_call(S, K, T, r, sigma)
        iv    = implied_vol(price, S, K, T, r, option_type="call")
        assert abs(iv - sigma) < 1e-4

    def test_roundtrip_put(self):
        from options_pricer import black_scholes_put, implied_vol
        S, K, T, r, sigma = 100.0, 95.0, 21/365, 0.05, 0.45
        price = black_scholes_put(S, K, T, r, sigma)
        iv    = implied_vol(price, S, K, T, r, option_type="put")
        assert abs(iv - sigma) < 1e-4

    def test_high_iv_call(self):
        from options_pricer import black_scholes_call, implied_vol
        S, K, T, r, sigma = 128.0, 130.0, 26/365, 0.05, 0.90
        price = black_scholes_call(S, K, T, r, sigma)
        iv    = implied_vol(price, S, K, T, r, option_type="call")
        assert abs(iv - sigma) < 1e-3

    def test_zero_price_returns_none(self):
        from options_pricer import implied_vol
        iv = implied_vol(0.0, 100.0, 200.0, 1/365, 0.05, option_type="call")
        assert iv is None

    def test_returns_float_or_none(self):
        from options_pricer import black_scholes_put, implied_vol
        price = black_scholes_put(100.0, 100.0, 30/365, 0.05, 0.25)
        iv    = implied_vol(price, 100.0, 100.0, 30/365, 0.05, option_type="put")
        assert isinstance(iv, float)


class TestEarlyTakeProfit:
    """v1 dynamic early take-profit: bank a fast winner, don't grind the tail."""

    def test_fast_thirty_pct_triggers(self):
        from options_pricer import should_take_early_profit
        # entry $4.37, mark $3.00 -> 31% captured, only 4 days held -> take it
        assert should_take_early_profit(4.37, 3.00, days_held=4) is True

    def test_thirty_pct_but_slow_does_not_trigger(self):
        from options_pricer import should_take_early_profit
        # same 31% profit but it took 20 days -> normal grind, leave the GTC
        assert should_take_early_profit(4.37, 3.00, days_held=20) is False

    def test_fast_but_under_threshold_does_not_trigger(self):
        from options_pricer import should_take_early_profit
        # $4.37 -> $3.60 is only 18% captured -> not enough yet
        assert should_take_early_profit(4.37, 3.60, days_held=2) is False

    def test_missing_mark_is_safe(self):
        from options_pricer import should_take_early_profit
        assert should_take_early_profit(4.37, None, days_held=1) is False

    def test_bad_entry_is_safe(self):
        from options_pricer import should_take_early_profit
        assert should_take_early_profit(0.0, 1.0, days_held=1) is False

    def test_boundaries_inclusive(self):
        from options_pricer import should_take_early_profit
        # exactly 30% captured on exactly day 10 -> triggers (>=, <=)
        assert should_take_early_profit(10.0, 7.0, days_held=10) is True
        # one day too slow -> no
        assert should_take_early_profit(10.0, 7.0, days_held=11) is False

    def test_custom_thresholds_override(self):
        from options_pricer import should_take_early_profit
        assert should_take_early_profit(10.0, 5.0, days_held=3,
                                        min_profit=0.60, max_days=5) is False
        assert should_take_early_profit(10.0, 3.0, days_held=3,
                                        min_profit=0.60, max_days=5) is True
