import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from trend_scanner import (
    compute_indicators,
    detect_trend,
    detect_pullback,
    detect_entry,
    compute_levels,
    scan_symbol,
)


def _make_trending_df(n=120, drift=0.002):
    """Steadily uptrending OHLCV data."""
    np.random.seed(1)
    close = 100 * np.cumprod(1 + drift + np.random.randn(n) * 0.01)
    df = pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=n, freq="B"),
        "open":  close * 0.995,
        "high":  close * 1.015,
        "low":   close * 0.985,
        "close": close,
        "volume": np.random.randint(10_000_000, 20_000_000, n).astype(float),
    })
    return df


def _make_flat_df(n=120):
    """Flat/choppy data — no trend."""
    np.random.seed(2)
    close = 100 + np.random.randn(n) * 1.5
    df = pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=n, freq="B"),
        "open":  close * 0.999,
        "high":  close * 1.01,
        "low":   close * 0.99,
        "close": close,
        "volume": np.random.randint(5_000_000, 10_000_000, n).astype(float),
    })
    return df


def test_compute_indicators_columns():
    df = _make_trending_df()
    out = compute_indicators(df)
    for col in ("ema20", "ema50", "adx", "rsi", "atr"):
        assert col in out.columns, f"Missing column: {col}"


def test_detect_trend_true_for_uptrend():
    df = compute_indicators(_make_trending_df(120))
    df = df.dropna().reset_index(drop=True)
    assert detect_trend(df) is True


def test_detect_trend_false_for_flat():
    df = compute_indicators(_make_flat_df(120))
    df = df.dropna().reset_index(drop=True)
    assert detect_trend(df) is False


def test_detect_pullback_returns_bool():
    df = compute_indicators(_make_trending_df())
    df = df.dropna().reset_index(drop=True)
    result = detect_pullback(df)
    assert isinstance(result, bool)


def test_detect_entry_returns_bool():
    df = compute_indicators(_make_trending_df())
    df = df.dropna().reset_index(drop=True)
    result = detect_entry(df)
    assert isinstance(result, bool)


def test_compute_levels():
    levels = compute_levels(entry=200.0)
    assert levels["target"] == pytest.approx(240.0)
    assert levels["stop"]   == pytest.approx(184.0)
    assert levels["risk_reward"] == pytest.approx(2.5)


def test_scan_symbol_no_signal_on_flat():
    df = compute_indicators(_make_flat_df())
    df = df.dropna().reset_index(drop=True)
    result = scan_symbol("TEST", df)
    assert result["signal"] == "NONE"


def test_scan_symbol_structure():
    df = compute_indicators(_make_trending_df())
    df = df.dropna().reset_index(drop=True)
    result = scan_symbol("NVDA", df)
    assert "symbol" in result
    assert "signal" in result
    assert result["signal"] in ("BUY", "NONE")
    if result["signal"] == "BUY":
        assert result["target"] > result["entry"]
        assert result["stop"]   < result["entry"]
