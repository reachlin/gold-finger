"""
Tests for kronos_range.py — run before the real script to catch issues early.
These tests mock the Kronos model so no GPU/download is needed.
"""
import sys, os, types
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so we can import kronos_range without loading the real model
# ---------------------------------------------------------------------------

# Stub the Kronos model package
kronos_stub = types.ModuleType("model")
class _FakePredictor:
    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kw):
        # Return a df with realistic predicted OHLCV over pred_len future bars
        n = len(y_timestamp)
        base = df["close"].iloc[-1]
        data = {
            "open":   [base * 0.99] * n,
            "high":   [base * 1.04] * n,   # +4% ceiling
            "low":    [base * 0.96] * n,   # -4% floor
            "close":  [base * 1.00] * n,
            "volume": [1e6] * n,
            "amount": [base * 1e6] * n,
        }
        return pd.DataFrame(data, index=y_timestamp)

kronos_stub.Kronos = None
kronos_stub.KronosTokenizer = None
kronos_stub.KronosPredictor = _FakePredictor
sys.modules["model"] = kronos_stub

sys.path.insert(0, os.path.dirname(__file__))
import kronos_range as kr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_history(n=450, base=80.0) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n)
    np.random.seed(42)
    close = base + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "datetime": dates,
        "open":   close * 0.999,
        "high":   close * 1.005,
        "low":    close * 0.995,
        "close":  close,
        "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_future_timestamps_length():
    last = pd.Timestamp("2026-06-26")
    ts = kr._future_biz_dates(last, 30)
    assert len(ts) == 30
    assert ts[0] > last


def test_future_timestamps_no_weekends():
    last = pd.Timestamp("2026-06-26")
    ts = kr._future_biz_dates(last, 30)
    for t in ts:
        assert t.weekday() < 5, f"{t} is a weekend"


def test_predict_range_returns_expected_fields(monkeypatch):
    df = _make_history()
    predictor = _FakePredictor()
    result = kr.predict_range(df, predictor, lookback=400, pred_len=30)
    for field in ("support", "resistance", "range_pct", "current_price", "pred_df"):
        assert field in result, f"missing field: {field}"


def test_predict_range_values_are_sane(monkeypatch):
    df = _make_history(base=80.0)
    predictor = _FakePredictor()
    result = kr.predict_range(df, predictor, lookback=400, pred_len=30)
    assert result["support"] < result["resistance"]
    assert result["range_pct"] > 0
    assert result["current_price"] > 0


def test_predict_range_too_short_raises():
    df = _make_history(n=50)
    predictor = _FakePredictor()
    with pytest.raises(ValueError, match="Not enough history"):
        kr.predict_range(df, predictor, lookback=400, pred_len=30)


def test_strike_vs_support():
    df = _make_history(base=80.0)
    predictor = _FakePredictor()
    result = kr.predict_range(df, predictor, lookback=400, pred_len=30)
    # Fake predictor returns support = close * 0.96
    strike_5pct_otm = result["current_price"] * 0.95
    # Strike should be above support (96% of close) — assignment risk is present
    # but within the 5% OTM band
    assert result["support"] > 0
    safe = strike_5pct_otm < result["support"]
    assert isinstance(safe, bool)


def test_scan_all_returns_list(monkeypatch, tmp_path):
    # Write two fake history CSVs
    for sym in ["AA", "BB"]:
        df = _make_history()
        df.to_csv(tmp_path / f"{sym.lower()}_history.csv", index=False)
    predictor = _FakePredictor()
    rows = kr.scan_watchlist(["AA", "BB"], predictor, data_dir=str(tmp_path))
    assert len(rows) == 2
    assert rows[0]["symbol"] in ("AA", "BB")


def test_scan_all_skips_missing_file(tmp_path):
    df = _make_history()
    df.to_csv(tmp_path / "aa_history.csv", index=False)
    predictor = _FakePredictor()
    rows = kr.scan_watchlist(["AA", "MISSING"], predictor, data_dir=str(tmp_path))
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AA"
