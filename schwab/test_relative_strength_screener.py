import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from relative_strength_screener import (
    compute_return,
    rank_relative_strength,
    top_percentile,
    LOOKBACKS,
)


def _make_trend_df(n=150, drift=0.0, seed=1):
    np.random.seed(seed)
    close = 100 * np.cumprod(1 + drift + np.random.randn(n) * 0.005)
    return pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=n, freq="B"),
        "open":  close * 0.999,
        "high":  close * 1.01,
        "low":   close * 0.99,
        "close": close,
        "volume": np.ones(n) * 1_000_000,
    })


def test_compute_return_positive_for_uptrend():
    df = _make_trend_df(drift=0.01, seed=1)
    r = compute_return(df, lookback=21)
    assert r > 0


def test_compute_return_none_when_insufficient_history():
    df = _make_trend_df(n=10)
    assert compute_return(df, lookback=21) is None


def test_rank_relative_strength_orders_strong_above_weak():
    universe = {
        "STRONG": _make_trend_df(drift=0.012, seed=1),
        "MID":    _make_trend_df(drift=0.002, seed=2),
        "WEAK":   _make_trend_df(drift=-0.010, seed=3),
    }
    ranked = rank_relative_strength(universe)
    assert list(ranked.iloc[0:1]["symbol"]) == ["STRONG"]
    assert list(ranked.iloc[-1:]["symbol"]) == ["WEAK"]


def test_rank_relative_strength_columns():
    universe = {"A": _make_trend_df(drift=0.005, seed=4),
                "B": _make_trend_df(drift=0.001, seed=5)}
    ranked = rank_relative_strength(universe)
    for col in ("symbol", "rs_score", "ret_1m", "ret_3m", "ret_6m"):
        assert col in ranked.columns


def test_rank_relative_strength_score_is_percentile_0_100():
    universe = {f"S{i}": _make_trend_df(drift=0.001 * i, seed=i) for i in range(1, 11)}
    ranked = rank_relative_strength(universe)
    assert ranked["rs_score"].max() <= 100
    assert ranked["rs_score"].min() >= 0


def test_rank_relative_strength_drops_insufficient_history():
    universe = {
        "GOOD":       _make_trend_df(n=150, drift=0.005, seed=1),
        "TOO_SHORT":  _make_trend_df(n=20, drift=0.005, seed=2),
    }
    ranked = rank_relative_strength(universe)
    assert "TOO_SHORT" not in list(ranked["symbol"])
    assert "GOOD" in list(ranked["symbol"])


def _make_deterministic_trend_df(n=150, drift=0.0):
    """No noise — return is a pure function of drift, so ranking order is exact."""
    close = 100 * (1 + drift) ** np.arange(n)
    return pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=n, freq="B"),
        "open":  close * 0.999,
        "high":  close * 1.005,
        "low":   close * 0.995,
        "close": close,
        "volume": np.ones(n) * 1_000_000,
    })


def test_top_percentile_returns_top_slice():
    universe = {f"S{i}": _make_deterministic_trend_df(drift=0.0005 * i) for i in range(1, 101)}
    ranked = rank_relative_strength(universe)
    top = top_percentile(ranked, pct=0.02)
    assert len(top) == 2
    assert top.iloc[0]["symbol"] == "S100"   # highest drift -> highest RS, no noise to contest it


def test_top_percentile_returns_at_least_one():
    universe = {"A": _make_trend_df(drift=0.01, seed=1),
                "B": _make_trend_df(drift=0.002, seed=2),
                "C": _make_trend_df(drift=-0.01, seed=3)}
    ranked = rank_relative_strength(universe)
    top = top_percentile(ranked, pct=0.02)
    assert len(top) >= 1


def test_lookbacks_are_roughly_1_3_6_months_in_trading_days():
    assert LOOKBACKS == (21, 63, 126)
