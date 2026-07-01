"""Tests for fetch_us_options.py — written first (TDD)."""
import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

EXPECTED_COLUMNS = [
    "symbol", "expiration", "option_type", "strike",
    "bid", "ask", "last", "volume", "open_interest",
    "implied_volatility", "delta", "gamma", "theta", "vega",
]

DEFAULT_END_DATE = (date.today() + timedelta(days=60)).strftime("%Y-%m-%d")


def _make_yfinance_chain(expiry: str):
    row = {
        "strike": 150.0, "lastPrice": 3.5, "bid": 3.4, "ask": 3.6,
        "volume": 100, "openInterest": 500, "impliedVolatility": 0.3,
        "delta": 0.5, "gamma": 0.05, "theta": -0.02, "vega": 0.1,
    }
    calls = pd.DataFrame([row])
    puts = pd.DataFrame([row])
    chain = MagicMock()
    chain.calls = calls
    chain.puts = puts
    return chain


def _make_tradier_response(expiry: str):
    return {
        "options": {
            "option": [
                {
                    "option_type": "call",
                    "expiration_date": expiry,
                    "strike": 150.0,
                    "bid": 3.4, "ask": 3.6, "last": 3.5,
                    "volume": 100, "open_interest": 500,
                    "greeks": {
                        "delta": 0.5, "gamma": 0.05,
                        "theta": -0.02, "vega": 0.1,
                        "mid_iv": 0.3,
                    },
                }
            ]
        }
    }


def _make_polygon_response(expiry: str):
    return {
        "results": [
            {
                "details": {
                    "contract_type": "call",
                    "expiration_date": expiry,
                    "strike_price": 150.0,
                },
                "day": {"last_price": 3.5, "volume": 100},
                "greeks": {"delta": 0.5, "gamma": 0.05, "theta": -0.02, "vega": 0.1},
                "implied_volatility": 0.3,
                "open_interest": 500,
                "last_quote": {"bid": 3.4, "ask": 3.6},
            }
        ],
        "next_url": None,
    }


class TestFetchUsOptions(unittest.TestCase):

    def _near_expiry(self, days=15):
        return (date.today() + timedelta(days=days)).strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # 1. yfinance happy path — returns a DataFrame
    # ------------------------------------------------------------------
    @patch("fetch_us_options.yf.Ticker")
    def test_fetch_returns_dataframe(self, mock_ticker_cls):
        expiry = self._near_expiry(15)
        ticker = MagicMock()
        ticker.options = (expiry,)
        ticker.option_chain.return_value = _make_yfinance_chain(expiry)
        mock_ticker_cls.return_value = ticker

        from fetch_us_options import fetch_us_options
        df = fetch_us_options("AAPL")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    # ------------------------------------------------------------------
    # 2. All 14 standardized columns are present
    # ------------------------------------------------------------------
    @patch("fetch_us_options.yf.Ticker")
    def test_columns_standardized(self, mock_ticker_cls):
        expiry = self._near_expiry(15)
        ticker = MagicMock()
        ticker.options = (expiry,)
        ticker.option_chain.return_value = _make_yfinance_chain(expiry)
        mock_ticker_cls.return_value = ticker

        from fetch_us_options import fetch_us_options
        df = fetch_us_options("AAPL")
        for col in EXPECTED_COLUMNS:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    # ------------------------------------------------------------------
    # 3. Both call and put option_type values present
    # ------------------------------------------------------------------
    @patch("fetch_us_options.yf.Ticker")
    def test_option_types_present(self, mock_ticker_cls):
        expiry = self._near_expiry(15)
        ticker = MagicMock()
        ticker.options = (expiry,)
        ticker.option_chain.return_value = _make_yfinance_chain(expiry)
        mock_ticker_cls.return_value = ticker

        from fetch_us_options import fetch_us_options
        df = fetch_us_options("AAPL")
        types = set(df["option_type"].unique())
        self.assertIn("call", types)
        self.assertIn("put", types)

    # ------------------------------------------------------------------
    # 4. Date range filter: expirations beyond end_date are excluded
    # ------------------------------------------------------------------
    @patch("fetch_us_options.yf.Ticker")
    def test_date_range_filter_applied(self, mock_ticker_cls):
        near = self._near_expiry(10)
        far = self._near_expiry(90)   # beyond default 60-day window
        ticker = MagicMock()
        ticker.options = (near, far)
        ticker.option_chain.side_effect = lambda d: _make_yfinance_chain(d)
        mock_ticker_cls.return_value = ticker

        from fetch_us_options import fetch_us_options
        df = fetch_us_options("AAPL")  # default end_date = today + 60 days
        cutoff = DEFAULT_END_DATE
        for exp in df["expiration"].unique():
            self.assertLessEqual(exp, cutoff)

    # ------------------------------------------------------------------
    # 4b. Custom end_date respected
    # ------------------------------------------------------------------
    @patch("fetch_us_options.yf.Ticker")
    def test_custom_end_date_respected(self, mock_ticker_cls):
        near = self._near_expiry(10)
        mid = self._near_expiry(25)
        ticker = MagicMock()
        ticker.options = (near, mid)
        ticker.option_chain.side_effect = lambda d: _make_yfinance_chain(d)
        mock_ticker_cls.return_value = ticker

        from fetch_us_options import fetch_us_options
        custom_end = self._near_expiry(15)
        df = fetch_us_options("AAPL", end_date=custom_end)
        for exp in df["expiration"].unique():
            self.assertLessEqual(exp, custom_end)

    # ------------------------------------------------------------------
    # 5. Fallback to Tradier when yfinance fails
    # ------------------------------------------------------------------
    @patch("fetch_us_options._fetch_tradier")
    @patch("fetch_us_options._fetch_yfinance")
    def test_fallback_to_tradier_on_yfinance_error(self, mock_yf, mock_tradier):
        mock_yf.side_effect = Exception("yfinance down")
        row = {c: None for c in EXPECTED_COLUMNS}
        row["expiration"] = self._near_expiry(15)
        mock_tradier.return_value = pd.DataFrame([row])

        from fetch_us_options import fetch_us_options
        df = fetch_us_options("AAPL")
        mock_tradier.assert_called_once()
        self.assertFalse(df.empty)

    # ------------------------------------------------------------------
    # 6. Fallback to Polygon when yfinance and Tradier both fail
    # ------------------------------------------------------------------
    @patch("fetch_us_options._fetch_polygon")
    @patch("fetch_us_options._fetch_tradier")
    @patch("fetch_us_options._fetch_yfinance")
    def test_fallback_to_polygon_on_both_errors(self, mock_yf, mock_tradier, mock_polygon):
        mock_yf.side_effect = Exception("yfinance down")
        mock_tradier.side_effect = Exception("tradier down")
        row = {c: None for c in EXPECTED_COLUMNS}
        row["expiration"] = self._near_expiry(15)
        mock_polygon.return_value = pd.DataFrame([row])

        from fetch_us_options import fetch_us_options
        df = fetch_us_options("AAPL")
        mock_polygon.assert_called_once()
        self.assertFalse(df.empty)

    # ------------------------------------------------------------------
    # 7. All providers fail → raises RuntimeError
    # ------------------------------------------------------------------
    @patch("fetch_us_options._fetch_polygon")
    @patch("fetch_us_options._fetch_tradier")
    @patch("fetch_us_options._fetch_yfinance")
    def test_all_providers_fail_raises(self, mock_yf, mock_tradier, mock_polygon):
        mock_yf.side_effect = Exception("yfinance down")
        mock_tradier.side_effect = Exception("tradier down")
        mock_polygon.side_effect = Exception("polygon down")

        from fetch_us_options import fetch_us_options
        with self.assertRaises(RuntimeError):
            fetch_us_options("AAPL")

    # ------------------------------------------------------------------
    # 8. _fetch_tradier parses REST response correctly
    # ------------------------------------------------------------------
    @patch("fetch_us_options.requests.get")
    def test_tradier_fetch(self, mock_get):
        expiry = self._near_expiry(15)

        # First call: expirations endpoint
        exp_resp = MagicMock()
        exp_resp.raise_for_status.return_value = None
        exp_resp.json.return_value = {"expirations": {"date": [expiry]}}

        # Second call: chains endpoint
        chain_resp = MagicMock()
        chain_resp.raise_for_status.return_value = None
        chain_resp.json.return_value = _make_tradier_response(expiry)

        mock_get.side_effect = [exp_resp, chain_resp]

        from fetch_us_options import _fetch_tradier
        df = _fetch_tradier("AAPL", end_date=DEFAULT_END_DATE)
        self.assertFalse(df.empty)
        for col in EXPECTED_COLUMNS:
            self.assertIn(col, df.columns)

    # ------------------------------------------------------------------
    # 9. _fetch_polygon parses REST response correctly
    # ------------------------------------------------------------------
    @patch("fetch_us_options.requests.get")
    def test_polygon_fetch(self, mock_get):
        expiry = self._near_expiry(15)
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = _make_polygon_response(expiry)
        mock_get.return_value = resp

        from fetch_us_options import _fetch_polygon
        df = _fetch_polygon("AAPL", end_date=DEFAULT_END_DATE)
        self.assertFalse(df.empty)
        for col in EXPECTED_COLUMNS:
            self.assertIn(col, df.columns)


if __name__ == "__main__":
    unittest.main()
