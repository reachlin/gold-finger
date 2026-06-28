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
        valid = {"put_expired", "put_assigned", "put_early_exit",
                 "call_expired", "called_away", "call_early_exit"}
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


class TestOverseerIntegration:
    def _make_nuked_spy(self, n=150):
        """SPY df where EMA50 is falling — WASTELAND, and VIX is spiked."""
        close = np.linspace(100, 70, n)  # steady decline
        return pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n, freq="B"),
            "open":  close * 0.998,
            "high":  close * 1.005,
            "low":   close * 0.990,
            "close": close,
            "volume": np.ones(n) * 1e8,
        })

    def _make_vix_high(self, n=150, vix_level=35.0):
        return pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n, freq="B"),
            "open":  np.full(n, vix_level),
            "high":  np.full(n, vix_level),
            "low":   np.full(n, vix_level),
            "close": np.full(n, vix_level),
            "volume": np.zeros(n),
        })

    def test_nuked_zone_blocks_all_new_puts(self):
        from backtest_scavenger import walk_forward_scavenger
        df      = _make_df(n=200)
        spy_df  = self._make_nuked_spy(n=200)
        vix_df  = self._make_vix_high(n=200, vix_level=40.0)
        events  = walk_forward_scavenger(df, "TEST", spy_df=spy_df, vix_df=vix_df)
        # With VIX=40 throughout, no new puts should be opened
        put_events = [e for e in events if "put" in e["event"]]
        assert len(put_events) == 0, f"Expected 0 puts in NUKED_ZONE, got {len(put_events)}"

    def test_no_spy_data_still_runs(self):
        from backtest_scavenger import walk_forward_scavenger
        df = _make_df(n=200)
        events = walk_forward_scavenger(df, "TEST", spy_df=None, vix_df=None)
        assert isinstance(events, list)

    def test_regime_recorded_in_cycle(self):
        from backtest_scavenger import walk_forward_scavenger
        from vault76.overseer import Overseer
        df  = _make_df(n=300)
        # SPY in bull trend (rising prices)
        spy_close = np.linspace(400, 600, 300)
        spy_df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=300, freq="B"),
            "open":  spy_close * 0.999,
            "high":  spy_close * 1.005,
            "low":   spy_close * 0.995,
            "close": spy_close,
            "volume": np.ones(300) * 1e8,
        })
        events = walk_forward_scavenger(df, "TEST", spy_df=spy_df)
        for e in events:
            if "put_regime" in e:
                assert e["put_regime"] in (Overseer.RECLAMATION,
                                           Overseer.WASTELAND,
                                           Overseer.NUKED_ZONE)


class TestAdaptiveProfitTarget:
    def test_returns_float(self):
        from backtest_scavenger import adaptive_profit_target
        assert isinstance(adaptive_profit_target(0.40, 30), float)

    def test_higher_iv_gives_higher_target(self):
        from backtest_scavenger import adaptive_profit_target
        low  = adaptive_profit_target(0.20, 30)
        high = adaptive_profit_target(0.80, 30)
        assert high > low

    def test_longer_dte_gives_higher_target(self):
        from backtest_scavenger import adaptive_profit_target
        short = adaptive_profit_target(0.40, 15)
        long_ = adaptive_profit_target(0.40, 60)
        assert long_ > short

    def test_range_always_35_to_65(self):
        from backtest_scavenger import adaptive_profit_target
        for iv in [0.10, 0.20, 0.50, 1.00, 2.00]:
            for dte in [5, 15, 30, 60, 90]:
                t = adaptive_profit_target(iv, dte)
                assert 0.35 <= t <= 0.65, f"iv={iv} dte={dte} → {t}"

    def test_midpoint_gives_50_pct(self):
        from backtest_scavenger import adaptive_profit_target
        # iv=0.60: iv_factor=(0.60-0.20)/0.80=0.5, dte=45: dte_factor=(45-15)/60=0.5
        # score=0.6*0.5+0.4*0.5=0.5 → target=0.35+0.30*0.5=0.50
        t = adaptive_profit_target(0.60, 45)
        assert abs(t - 0.50) < 0.01


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
