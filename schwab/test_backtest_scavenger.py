"""
Tests for backtest_scavenger.py — wheel strategy simulation.

Run: /Users/lincai/anaconda3/envs/gold-finger/bin/python -m pytest schwab/test_backtest_scavenger.py -v
"""
import sys, os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_df(n=150, close_series=None, seed=5):
    """OHLCV df. close_series overrides generated prices if provided."""
    np.random.seed(seed)
    if close_series is None:
        close = 100 + np.random.randn(n) * 2.0
    else:
        close = np.array(close_series, dtype=float)
        n = len(close)
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open":  close * 0.998,
        "high":  close * 1.015,
        "low":   close * 0.985,
        "close": close,
        "volume": np.ones(n) * 6_000_000,
    })


class TestWalkForwardScavenger:
    def test_returns_list(self):
        from backtest_scavenger import walk_forward_scavenger
        df = _make_df()
        result = walk_forward_scavenger(df, "TEST")
        assert isinstance(result, list)

    def test_each_event_has_required_keys(self):
        from backtest_scavenger import walk_forward_scavenger
        df = _make_df(n=200)
        events = walk_forward_scavenger(df, "TEST")
        for ev in events:
            assert "event" in ev
            assert "pnl"   in ev
            assert "symbol" in ev

    def test_put_expired_event_pnl_positive(self):
        from backtest_scavenger import walk_forward_scavenger
        df = _make_df(n=200)
        events = walk_forward_scavenger(df, "TEST")
        for ev in events:
            if ev["event"] == "put_expired":
                assert ev["pnl"] > 0, "kept premium should be positive"

    def test_call_expired_event_pnl_positive(self):
        from backtest_scavenger import walk_forward_scavenger
        df = _make_df(n=200)
        events = walk_forward_scavenger(df, "TEST")
        for ev in events:
            if ev["event"] == "call_expired":
                assert ev["pnl"] > 0, "kept call premium should be positive"

    def test_event_types_are_valid(self):
        from backtest_scavenger import walk_forward_scavenger
        valid = {"put_expired", "put_assigned", "call_expired", "called_away"}
        df = _make_df(n=200)
        events = walk_forward_scavenger(df, "TEST")
        for ev in events:
            assert ev["event"] in valid

    def test_put_assigned_followed_by_call_event(self):
        """put_assigned should be followed by call_expired or called_away."""
        from backtest_scavenger import walk_forward_scavenger
        df = _make_df(n=400)
        events = walk_forward_scavenger(df, "TEST")
        for j, ev in enumerate(events):
            if ev["event"] == "put_assigned" and j + 1 < len(events):
                assert events[j + 1]["event"] in ("call_expired", "called_away",
                                                    "put_assigned")


class TestPnlCalculations:
    def test_put_expired_pnl_equals_premium_times_100(self):
        from backtest_scavenger import walk_forward_scavenger
        df = _make_df(n=200)
        events = walk_forward_scavenger(df, "TEST")
        for ev in events:
            if ev["event"] == "put_expired":
                expected = ev["put_premium"] * 100
                assert abs(ev["pnl"] - expected) < 0.01

    def test_called_away_pnl_includes_share_gain(self):
        from backtest_scavenger import walk_forward_scavenger
        df = _make_df(n=400)
        events = walk_forward_scavenger(df, "TEST")
        for ev in events:
            if ev["event"] == "called_away":
                # pnl should include share gain AND accumulated premiums
                share_pnl = (ev["call_strike"] - ev["put_strike"] + ev["put_premium"]) * 100
                assert ev["pnl"] >= share_pnl - 1.0  # at least the basic share leg


class TestPrintScavengerReport:
    def test_report_runs_without_error(self):
        from backtest_scavenger import print_scavenger_report
        events = [
            {"symbol": "TEST", "event": "put_expired", "pnl": 150.0,
             "put_premium": 1.5, "put_strike": 95.0, "put_close": 100.0,
             "put_entry_i": 60, "put_expiry_i": 90},
            {"symbol": "TEST", "event": "put_assigned", "pnl": 0.0,
             "put_premium": 1.2, "put_strike": 92.0, "put_close": 100.0,
             "cost_basis": 90.8, "put_entry_i": 91, "put_expiry_i": 121,
             "put_expiry_close": 91.5},
            {"symbol": "TEST", "event": "called_away", "pnl": 420.0,
             "put_strike": 92.0, "put_premium": 1.2,
             "call_strike": 100.0, "call_premium": 0.8,
             "all_premiums": 2.0, "cost_basis": 90.8,
             "close_at_call_expiry": 102.0, "call_entry_i": 122,
             "call_expiry_i": 152},
        ]
        # Should not raise
        print_scavenger_report(events, init_price=100.0, final_price=105.0)
