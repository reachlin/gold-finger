"""
Tests for wheel_router.py — walk-forward P(>8% rally in 30d) series that
routes symbols between wheel mode (sell premium) and hold mode (own shares
uncapped). Mechanical rule → directly backtestable, unlike the LLM advisories.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import wheel_router as wr


def _trendy(n=700, base=100.0, seed=11) -> pd.DataFrame:
    """Alternating flat and strong-run segments — both label classes."""
    rng = np.random.default_rng(seed)
    drifts = np.where((np.arange(n) // 120) % 2 == 0, 0.0, 0.004)
    closes = base * np.cumprod(1 + drifts + rng.normal(0, 0.006, n))
    return pd.DataFrame({
        "datetime": pd.date_range("2019-01-01", periods=n, freq="B"),
        "open": closes * 0.999, "high": closes * 1.01,
        "low": closes * 0.99, "close": closes,
        "volume": np.full(n, 2e6),
    })


class TestWalkForwardProbs:
    def test_shape_warmup_and_range(self):
        df = _trendy()
        probs = wr.walk_forward_probs(df, min_train=300, refit_every=63)
        assert len(probs) == len(df)
        assert np.isnan(probs[:300]).all()          # warm-up is NaN
        valid = probs[~np.isnan(probs)]
        assert len(valid) > 0
        assert ((valid >= 0.0) & (valid <= 1.0)).all()

    def test_no_lookahead_truncation_invariance(self):
        """P at bar i must not change when future bars are removed."""
        df = _trendy()
        full = wr.walk_forward_probs(df, min_train=300, refit_every=63)
        cut  = wr.walk_forward_probs(df.iloc[:500].reset_index(drop=True),
                                     min_train=300, refit_every=63)
        np.testing.assert_allclose(full[:500], cut, equal_nan=True)

    def test_single_class_history_stays_nan(self):
        """A dead-flat series never rallies 8% — no trainable labels."""
        n = 500
        closes = np.full(n, 100.0)
        df = pd.DataFrame({
            "datetime": pd.date_range("2019-01-01", periods=n, freq="B"),
            "open": closes, "high": closes * 1.001,
            "low": closes * 0.999, "close": closes,
            "volume": np.full(n, 2e6),
        })
        probs = wr.walk_forward_probs(df, min_train=300, refit_every=63)
        assert np.isnan(probs).all()


class TestWalkForwardTimesFM:
    """TimesFM-as-predictor: per-bar zero-shot 30d SMA5 forecast pct."""

    @staticmethod
    def _ramp_fn(pct: float):
        """Every forecast ends pct above the context's last value."""
        def fn(contexts, horizon):
            return np.array([
                np.linspace(c[-1], c[-1] * (1 + pct / 100), horizon)
                for c in contexts
            ])
        return fn

    def test_shape_warmup_and_values(self):
        df = _trendy()
        out = wr.walk_forward_timesfm(df, forecast_fn=self._ramp_fn(4.0))
        assert len(out) == len(df)
        assert np.isnan(out[:50]).all()            # SMA/context warm-up
        valid = out[~np.isnan(out)]
        assert len(valid) > 0
        np.testing.assert_allclose(valid, 4.0, atol=0.01)

    def test_no_lookahead_truncation_invariance(self):
        """Forecast at bar i uses only the trailing context up to bar i."""
        def echo_fn(contexts, horizon):
            # forecast = last context value scaled by context mean parity —
            # any deterministic function of the context works here
            return np.array([np.full(horizon, c[-1] * (1 + (c.mean() % 0.01)))
                             for c in contexts])
        df   = _trendy()
        full = wr.walk_forward_timesfm(df, forecast_fn=echo_fn)
        cut  = wr.walk_forward_timesfm(df.iloc[:500].reset_index(drop=True),
                                       forecast_fn=echo_fn)
        np.testing.assert_allclose(full[:500], cut, equal_nan=True)

    def test_routes_backtest_with_pct_threshold(self):
        """The pct series plugs into the same router harness (τ in %)."""
        from backtest_scavenger import walk_forward_scavenger
        df  = _trendy()
        out = wr.walk_forward_timesfm(df, forecast_fn=self._ramp_fn(6.0))
        events = walk_forward_scavenger(df, "TEST", router_probs=out,
                                        router_threshold=4.0)
        first_valid = int(np.argmax(~np.isnan(out)))
        assert not any(e["event"].startswith("put_")
                       and e["put_entry_i"] >= first_valid for e in events)


class TestRouterGateInBacktest:
    def test_hot_router_produces_holds_not_puts(self):
        from backtest_scavenger import walk_forward_scavenger
        df = _trendy()
        probs = np.full(len(df), 1.0)               # always "runner"
        events = walk_forward_scavenger(df, "TEST", router_probs=probs,
                                        router_threshold=0.6)
        kinds = {e["event"] for e in events}
        assert "router_hold_exit" in kinds or len(events) == 0
        assert not any(e["event"].startswith("put_") for e in events)

    def test_cold_router_matches_baseline(self):
        from backtest_scavenger import walk_forward_scavenger
        df = _trendy()
        probs = np.zeros(len(df))                   # never routes
        base   = walk_forward_scavenger(df, "TEST")
        routed = walk_forward_scavenger(df, "TEST", router_probs=probs,
                                        router_threshold=0.6)
        assert [e["event"] for e in base] == [e["event"] for e in routed]
        assert sum(e["pnl"] for e in base) == sum(e["pnl"] for e in routed)

    def test_hold_pnl_is_share_move(self):
        from backtest_scavenger import walk_forward_scavenger, SHARES
        df = _trendy()
        probs = np.full(len(df), 1.0)
        events = walk_forward_scavenger(df, "TEST", router_probs=probs,
                                        router_threshold=0.6)
        holds = [e for e in events if e["event"] == "router_hold_exit"]
        assert holds, "expected at least one router hold"
        for e in holds:
            expected = (e["hold_exit_close"] - e["hold_entry_close"]) * SHARES
            assert e["pnl"] == pytest.approx(expected, abs=0.02)
