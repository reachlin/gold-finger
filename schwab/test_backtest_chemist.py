"""
Tests for backtest_chemist.py — bear put spread simulation.

Run: /Users/lincai/anaconda3/envs/gold-finger/bin/python -m pytest schwab/test_backtest_chemist.py -v
"""
import sys, os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_df(n=200, trend="down", seed=7):
    """Crash-like daily vol (2.5%/day ≈ 40% HV annualized) for NUKED_ZONE realism."""
    np.random.seed(seed)
    daily_sigma = 0.025
    drift = {"down": -0.003, "up": +0.002, "flat": 0.0}[trend]
    returns = drift + np.random.randn(n) * daily_sigma
    close   = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "datetime": pd.date_range("2022-06-01", periods=n, freq="B"),
        "open":  close * 0.998,
        "high":  close * 1.020,
        "low":   close * 0.980,
        "close": close,
        "volume": np.ones(n) * 5_000_000,
    })


def _make_vix(n=200, level=35.0):
    dates = pd.date_range("2022-06-01", periods=n, freq="B")
    return pd.DataFrame({
        "datetime": dates,
        "open":  np.full(n, level),
        "high":  np.full(n, level),
        "low":   np.full(n, level),
        "close": np.full(n, level),
        "volume": np.zeros(n),
    })


def _make_spy(n=200, trend="down"):
    np.random.seed(99)
    if trend == "down":
        close = np.linspace(430, 320, n) + np.random.randn(n)
    else:
        close = np.linspace(380, 480, n) + np.random.randn(n)
    return pd.DataFrame({
        "datetime": pd.date_range("2022-06-01", periods=n, freq="B"),
        "open":  close * 0.999,
        "high":  close * 1.005,
        "low":   close * 0.995,
        "close": close,
        "volume": np.ones(n) * 1e8,
    })


class TestWalkForwardChemist:
    def test_returns_list(self):
        from backtest_chemist import walk_forward_chemist
        df = _make_df()
        result = walk_forward_chemist(df, "TEST")
        assert isinstance(result, list)

    def test_no_trades_without_nuked_zone(self):
        """VIX=15 (calm) → no Chemist trades ever fire."""
        from backtest_chemist import walk_forward_chemist
        df    = _make_df(trend="down")
        vix   = _make_vix(level=15.0)
        spy   = _make_spy()
        events = walk_forward_chemist(df, "TEST", spy_df=spy, vix_df=vix)
        assert len(events) == 0, f"Expected 0 trades with VIX=15, got {len(events)}"

    def test_trades_fire_in_nuked_zone(self):
        """VIX=40 + downtrend → Chemist should enter at least one spread."""
        from backtest_chemist import walk_forward_chemist
        df    = _make_df(n=300, trend="down")
        vix   = _make_vix(n=300, level=40.0)
        spy   = _make_spy(n=300, trend="down")
        events = walk_forward_chemist(df, "TEST", spy_df=spy, vix_df=vix)
        assert len(events) > 0, "Expected trades in sustained NUKED_ZONE downtrend"

    def test_event_has_required_keys(self):
        from backtest_chemist import walk_forward_chemist
        df    = _make_df(n=300, trend="down")
        vix   = _make_vix(n=300, level=40.0)
        events = walk_forward_chemist(df, "TEST", vix_df=vix)
        for e in events:
            for key in ("symbol", "event", "pnl", "net_credit",
                        "long_strike", "short_strike", "entry_date"):
                assert key in e, f"Missing key: {key}"

    def test_pnl_bounded_by_max_profit_and_loss(self):
        """P&L must stay within [-max_loss, max_profit] per trade (with 5% tolerance)."""
        from backtest_chemist import walk_forward_chemist
        df    = _make_df(n=300, trend="down")
        vix   = _make_vix(n=300, level=40.0)
        events = walk_forward_chemist(df, "TEST", vix_df=vix)
        for e in events:
            # Credit spread: max loss = spread_width - net_credit; max profit = net_credit
            assert e["pnl"] >= -e["max_loss"] * 1.05
            assert e["pnl"] <= e["max_profit"] * 1.05

    def test_no_spy_data_still_runs(self):
        """Falls back gracefully when SPY is missing (only VIX used for regime)."""
        from backtest_chemist import walk_forward_chemist
        df  = _make_df(trend="down")
        vix = _make_vix(level=40.0)
        events = walk_forward_chemist(df, "TEST", spy_df=None, vix_df=vix)
        assert isinstance(events, list)

    def test_nuked_zone_required_for_any_trades(self):
        """Credit spreads only fire during NUKED_ZONE — confirmed by VIX=15 giving 0 trades."""
        from backtest_chemist import walk_forward_chemist
        vix_calm = _make_vix(n=300, level=15.0)
        vix_nuke = _make_vix(n=300, level=40.0)
        calm_events = walk_forward_chemist(_make_df(n=300), "C", vix_df=vix_calm)
        nuke_events = walk_forward_chemist(_make_df(n=300), "N", vix_df=vix_nuke)
        assert len(calm_events) == 0
        assert len(nuke_events) >= len(calm_events)

    def test_exit_event_types_valid(self):
        from backtest_chemist import walk_forward_chemist
        VALID_EVENTS = {"spread_expired", "spread_profit_target", "spread_loss_limit"}
        df    = _make_df(n=300, trend="down")
        vix   = _make_vix(n=300, level=40.0)
        events = walk_forward_chemist(df, "TEST", vix_df=vix)
        for e in events:
            assert e["event"] in VALID_EVENTS, f"Unknown event: {e['event']}"
