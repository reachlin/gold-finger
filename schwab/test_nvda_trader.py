import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nvda_trader import (
    candles_to_df,
    compute_trade_signal,
    build_order,
    get_position_shares,
)


def _make_df(n=100):
    """Minimal OHLCV dataframe for testing."""
    np.random.seed(42)
    close = 190 + np.cumsum(np.random.randn(n) * 2)
    df = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open":  close * 0.99,
        "high":  close * 1.01,
        "low":   close * 0.98,
        "close": close,
        "volume": np.random.randint(10_000_000, 30_000_000, n).astype(float),
    })
    return df


def test_candles_to_df():
    raw = [
        {"datetime": 1700000000000, "open": 190, "high": 195,
         "low": 188, "close": 193, "volume": 15000000}
    ]
    df = candles_to_df(raw)
    assert list(df.columns) == ["datetime", "open", "high", "low", "close", "volume"]
    assert len(df) == 1
    assert df["close"].iloc[0] == 193


def test_compute_trade_signal_returns_valid():
    df = _make_df(100)
    signal = compute_trade_signal(df)
    assert signal in ("strong_buy", "mild_buy", "hold", "mild_sell", "strong_sell")


def test_get_position_shares_zero_when_no_nvda():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{
        "securitiesAccount": {
            "accountNumber": "12345678",
            "positions": []
        }
    }]
    mock_resp.raise_for_status = MagicMock()
    mock_client.get_accounts.return_value = mock_resp

    shares = get_position_shares(mock_client, "NVDA")
    assert shares == 0


def test_get_position_shares_found():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{
        "securitiesAccount": {
            "positions": [
                {"instrument": {"symbol": "NVDA"}, "longQuantity": 5.0, "shortQuantity": 0.0}
            ]
        }
    }]
    mock_resp.raise_for_status = MagicMock()
    mock_client.get_accounts.return_value = mock_resp

    shares = get_position_shares(mock_client, "NVDA")
    assert shares == 5


def test_build_order_buy():
    order = build_order("NVDA", "BUY", 3, 193.50)
    assert order["orderType"] == "LIMIT"
    assert order["orderLegCollection"][0]["instruction"] == "BUY"
    assert order["orderLegCollection"][0]["quantity"] == 3
    assert order["price"] == 193.50


def test_build_order_sell():
    order = build_order("NVDA", "SELL", 3, 193.50)
    assert order["orderLegCollection"][0]["instruction"] == "SELL"


def test_build_order_zero_quantity_raises():
    with pytest.raises(ValueError):
        build_order("NVDA", "BUY", 0, 193.50)
