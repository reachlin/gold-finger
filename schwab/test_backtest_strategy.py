import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from backtest_strategy import (
    simulate_trade,
    walk_forward,
    summarize,
)


def _make_df(n=150, drift=0.002):
    np.random.seed(42)
    close = 100 * np.cumprod(1 + drift + np.random.randn(n) * 0.01)
    df = pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=n, freq="B"),
        "open":  close * 0.995,
        "high":  close * 1.02,
        "low":   close * 0.98,
        "close": close,
        "volume": np.random.randint(10_000_000, 20_000_000, n).astype(float),
    })
    return df


def test_simulate_trade_hits_target():
    # Price goes straight up — should hit +20% target
    n = 30
    close = np.linspace(100, 130, n)
    df = pd.DataFrame({
        "open":  close * 0.998,
        "high":  close * 1.02,
        "low":   close * 0.99,
        "close": close,
    })
    result = simulate_trade(df, entry=100.0, target=120.0, stop=92.0)
    assert result["exit_reason"] == "target"
    assert result["pnl_pct"] == pytest.approx(20.0, abs=1.0)


def test_simulate_trade_hits_stop():
    # Price drops — should hit -8% stop
    n = 30
    close = np.linspace(100, 80, n)
    df = pd.DataFrame({
        "open":  close * 1.002,
        "high":  close * 1.01,
        "low":   close * 0.98,
        "close": close,
    })
    result = simulate_trade(df, entry=100.0, target=120.0, stop=92.0)
    assert result["exit_reason"] == "stop"
    assert result["pnl_pct"] < 0


def test_simulate_trade_timeout():
    # Price stays flat — exits at end of window
    n = 20
    close = np.full(n, 100.0)
    df = pd.DataFrame({
        "open":  close,
        "high":  close * 1.005,
        "low":   close * 0.995,
        "close": close,
    })
    result = simulate_trade(df, entry=100.0, target=120.0, stop=92.0, max_hold=n)
    assert result["exit_reason"] == "timeout"


def test_walk_forward_returns_list():
    df = _make_df(150)
    trades = walk_forward(df, min_history=60)
    assert isinstance(trades, list)


def test_walk_forward_no_lookahead():
    # Each trade entry must be after the signal date
    df = _make_df(150)
    trades = walk_forward(df, min_history=60)
    for t in trades:
        assert t["entry_date"] > t["signal_date"]


def test_summarize_empty():
    result = summarize([])
    assert result["total_trades"] == 0


def test_summarize_metrics():
    trades = [
        {"pnl_pct": 20.0, "exit_reason": "target", "hold_days": 10},
        {"pnl_pct": -8.0, "exit_reason": "stop",   "hold_days": 3},
        {"pnl_pct": 20.0, "exit_reason": "target",  "hold_days": 8},
    ]
    result = summarize(trades)
    assert result["total_trades"] == 3
    assert result["win_rate"] == pytest.approx(66.7, abs=0.1)
    assert result["avg_pnl_pct"] == pytest.approx((20 - 8 + 20) / 3, abs=0.1)
