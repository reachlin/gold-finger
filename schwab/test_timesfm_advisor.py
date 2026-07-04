"""
Tests for timesfm_advisor.py — zero-shot 30-day SMA5 direction forecast,
attached to signals as `timesfm_30d_pct` (advisory only).

The real TimesFM model is never loaded here — tests inject a fake
forecast_fn(contexts, horizon) -> (batch, horizon) array.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import timesfm_advisor as ta


def _history(n=600, base=100.0, drift=0.0005, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = base * np.cumprod(1 + drift + rng.normal(0, 0.005, n))
    return pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=n, freq="B"),
        "open": closes, "high": closes * 1.01, "low": closes * 0.99,
        "close": closes, "volume": np.full(n, 1e6),
    })


def _fake_fn(pct: float):
    """Forecast a flat ramp ending pct above each context's last value."""
    def fn(contexts, horizon):
        return np.array([
            np.linspace(c[-1], c[-1] * (1 + pct / 100), horizon)
            for c in contexts
        ])
    return fn


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_short_series_returns_none(self):
        closes = pd.Series(np.full(50, 100.0))
        assert ta.build_context(closes) is None

    def test_context_is_trimmed_sma(self):
        df = _history()
        ctx = ta.build_context(df["close"])
        assert ctx is not None
        assert len(ctx) <= ta.CONTEXT_LEN
        expected_last = df["close"].rolling(ta.SMA_WINDOW).mean().iloc[-1]
        assert ctx[-1] == pytest.approx(expected_last, rel=1e-5)


# ---------------------------------------------------------------------------
# Cache loader
# ---------------------------------------------------------------------------

class TestLoadCache:
    def test_bullish_forecast_positive_pct(self, tmp_path):
        _history().to_csv(tmp_path / "ko_history.csv", index=False)
        cache = ta.load_cache(["KO"], data_dir=str(tmp_path),
                              forecast_fn=_fake_fn(+5.0))
        assert cache["KO"]["pct"] == pytest.approx(5.0, abs=0.1)

    def test_bearish_forecast_negative_pct(self, tmp_path):
        _history().to_csv(tmp_path / "ko_history.csv", index=False)
        cache = ta.load_cache(["KO"], data_dir=str(tmp_path),
                              forecast_fn=_fake_fn(-3.0))
        assert cache["KO"]["pct"] == pytest.approx(-3.0, abs=0.1)

    def test_missing_history_skipped(self, tmp_path):
        _history().to_csv(tmp_path / "ko_history.csv", index=False)
        cache = ta.load_cache(["KO", "MISSING"], data_dir=str(tmp_path),
                              forecast_fn=_fake_fn(1.0))
        assert "KO" in cache and "MISSING" not in cache

    def test_unavailable_model_returns_empty(self, tmp_path, monkeypatch):
        def boom():
            raise ImportError("no timesfm")
        monkeypatch.setattr(ta, "_load_forecast_fn", boom)
        _history().to_csv(tmp_path / "ko_history.csv", index=False)
        assert ta.load_cache(["KO"], data_dir=str(tmp_path)) == {}


# ---------------------------------------------------------------------------
# Advise
# ---------------------------------------------------------------------------

class TestAdvise:
    def test_advise_returns_pct(self, tmp_path):
        _history().to_csv(tmp_path / "ko_history.csv", index=False)
        cache = ta.load_cache(["KO"], data_dir=str(tmp_path),
                              forecast_fn=_fake_fn(2.0))
        assert ta.advise(cache, "KO") == pytest.approx(2.0, abs=0.1)

    def test_advise_unknown_symbol_returns_none(self):
        assert ta.advise({}, "ZZZ") is None
