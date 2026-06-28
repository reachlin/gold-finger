"""
Tests for backtest_raider.py

Run: /Users/lincai/anaconda3/envs/gold-finger/bin/python -m pytest schwab/test_backtest_raider.py -v
"""
import sys, os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_vix(n, level):
    return pd.DataFrame({
        "datetime": pd.date_range("2022-06-01", periods=n, freq="B"),
        "open": np.full(n, level), "high": np.full(n, level),
        "low":  np.full(n, level), "close": np.full(n, level),
        "volume": np.zeros(n),
    })


def _make_spy(n, trend="up"):
    np.random.seed(1)
    close = np.linspace(380, 500, n) if trend == "up" else np.linspace(500, 350, n)
    return pd.DataFrame({
        "datetime": pd.date_range("2022-06-01", periods=n, freq="B"),
        "open":  close * 0.999, "high": close * 1.004,
        "low":   close * 0.996, "close": close,
        "volume": np.ones(n) * 1e8,
    })


def _make_trending_df(n=250, seed=42):
    """Strong uptrend with realistic pullbacks — maximises chance of Raider signals."""
    np.random.seed(seed)
    # Strong uptrend with occasional dips
    returns = np.where(
        np.random.rand(n) < 0.15,          # 15% of days: pullback
        np.random.randn(n) * 0.015 - 0.008,
        np.random.randn(n) * 0.010 + 0.004, # rest: grind higher
    )
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "datetime": pd.date_range("2022-06-01", periods=n, freq="B"),
        "open":  close * np.where(np.random.rand(n) > 0.5, 0.997, 1.003),
        "high":  close * (1 + np.abs(np.random.randn(n)) * 0.008),
        "low":   close * (1 - np.abs(np.random.randn(n)) * 0.008),
        "close": close,
        "volume": np.random.uniform(3e6, 8e6, n),
    })


class TestWalkForwardRaider:
    def test_returns_list(self):
        from backtest_raider import walk_forward_raider
        df = _make_trending_df()
        assert isinstance(walk_forward_raider(df, "TEST"), list)

    def test_no_trades_in_nuked_zone(self):
        """VIX=40 → NUKED_ZONE → Raider blocked."""
        from backtest_raider import walk_forward_raider
        df  = _make_trending_df(n=300)
        vix = _make_vix(300, 40.0)
        spy = _make_spy(300, "up")
        events = walk_forward_raider(df, "TEST", spy_df=spy, vix_df=vix)
        assert len(events) == 0, f"Expected 0 Raider trades in NUKED_ZONE, got {len(events)}"

    def test_event_keys_present(self):
        from backtest_raider import walk_forward_raider
        df  = _make_trending_df(n=400)
        spy = _make_spy(400, "up")
        vix = _make_vix(400, 15.0)
        events = walk_forward_raider(df, "TEST", spy_df=spy, vix_df=vix)
        for e in events:
            for k in ("symbol", "event", "pnl", "entry_price",
                      "exit_price", "entry_date", "exit_date"):
                assert k in e, f"Missing key: {k}"

    def test_exit_event_types_valid(self):
        from backtest_raider import walk_forward_raider
        VALID = {"raider_target_hit", "raider_trend_end",
                 "raider_stop_hit",   "raider_max_hold"}
        df  = _make_trending_df(n=400)
        spy = _make_spy(400, "up")
        vix = _make_vix(400, 15.0)
        events = walk_forward_raider(df, "TEST", spy_df=spy, vix_df=vix)
        for e in events:
            assert e["event"] in VALID

    def test_pnl_matches_price_diff(self):
        from backtest_raider import walk_forward_raider
        df  = _make_trending_df(n=400)
        spy = _make_spy(400, "up")
        vix = _make_vix(400, 15.0)
        events = walk_forward_raider(df, "TEST", spy_df=spy, vix_df=vix)
        for e in events:
            expected = round((e["exit_price"] - e["entry_price"]) * 100, 2)
            assert abs(e["pnl"] - expected) < 0.01

    def test_symbol_recorded(self):
        from backtest_raider import walk_forward_raider
        df     = _make_trending_df(n=400)
        spy    = _make_spy(400, "up")
        vix    = _make_vix(400, 15.0)
        events = walk_forward_raider(df, "NVDA", spy_df=spy, vix_df=vix)
        for e in events:
            assert e["symbol"] == "NVDA"
