#!/usr/bin/env python3
"""Tests for TimesFM SMA5 feature computation."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


def _make_df(n=100, seed=0):
    rng = np.random.default_rng(seed)
    close = 10.0 + np.cumsum(rng.standard_normal(n) * 0.1)
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="B").strftime("%Y-%m-%d"),
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": rng.integers(100_000, 500_000, n).astype(float),
        "amount": close * 200_000,
        "amplitude": rng.random(n),
        "pct_change": rng.standard_normal(n),
        "change": rng.standard_normal(n) * 0.1,
        "turnover_rate": rng.random(n),
    })


def _mock_model(ret_value=0.01):
    """Mock model whose forecast always returns current * (1 + ret_value)."""
    from unittest.mock import MagicMock
    mock = MagicMock()
    def forecast_fn(horizon, inputs):
        n = len(inputs)
        # Return last value of each context * (1 + ret_value)
        pts = np.array([[float(inp[-1]) * (1 + ret_value)] * horizon
                        for inp in inputs], dtype=np.float32)
        quantiles = np.zeros((n, horizon, 9), dtype=np.float32)
        return pts, quantiles
    mock.forecast.side_effect = forecast_fn
    return mock


class TestAddTimesFMFeature(unittest.TestCase):

    def _run(self, df, ret_value=0.01, context_len=32):
        from timesfm_feature import add_timesfm_feature
        # Inject mock directly — _load_timesfm is never called when _model is provided
        result = add_timesfm_feature(df, context_len=context_len, _model=_mock_model(ret_value))
        return result

    def test_column_added(self):
        df = _make_df(60)
        result = self._run(df)
        self.assertIn("tfm_sma5_ret", result.columns)

    def test_original_columns_preserved(self):
        df = _make_df(60)
        result = self._run(df)
        for col in df.columns:
            self.assertIn(col, result.columns)

    def test_length_unchanged(self):
        df = _make_df(60)
        result = self._run(df)
        self.assertEqual(len(result), len(df))

    def test_warmup_rows_are_zero(self):
        """First 4 rows have no SMA5, feature should be 0."""
        df = _make_df(60)
        result = self._run(df)
        self.assertTrue((result["tfm_sma5_ret"].iloc[:4] == 0.0).all())

    def test_feature_reflects_forecast_direction(self):
        """With mock forecasting a large positive value, mean tfm_sma5_ret > 0."""
        df = _make_df(60)
        # Use extreme ret so forecast is unambiguously above any current sma5
        result = self._run(df, ret_value=10.0)   # forecast = 11x last context price
        valid = result["tfm_sma5_ret"].iloc[4:]
        self.assertGreater(valid.mean(), 0,
                           "Mean feature should be positive when forecast is very high")

    def test_feature_negative_when_forecast_falls(self):
        df = _make_df(60)
        result = self._run(df, ret_value=-0.02)
        valid = result["tfm_sma5_ret"].iloc[4:]
        self.assertTrue((valid < 0).all())

    def test_idempotent(self):
        """Calling twice returns same values."""
        df = _make_df(60)
        r1 = self._run(df, ret_value=0.01)
        r2 = self._run(df, ret_value=0.01)
        pd.testing.assert_series_equal(r1["tfm_sma5_ret"], r2["tfm_sma5_ret"])

    def test_does_not_modify_input(self):
        df = _make_df(60)
        original_cols = list(df.columns)
        _ = self._run(df)
        self.assertEqual(list(df.columns), original_cols)


class TestFeatureColsExt(unittest.TestCase):

    def test_feature_cols_ext_contains_tfm(self):
        from trading_bot import FEATURE_COLS_EXT
        self.assertIn("tfm_sma5_ret", FEATURE_COLS_EXT)

    def test_feature_cols_ext_superset_of_feature_cols(self):
        from trading_bot import FEATURE_COLS, FEATURE_COLS_EXT
        for col in FEATURE_COLS:
            self.assertIn(col, FEATURE_COLS_EXT)


class TestLGBMUsesExtendedFeatures(unittest.TestCase):

    def test_lgbm_uses_ext_when_column_present(self):
        """LGBMTradingBot should use FEATURE_COLS_EXT when tfm_sma5_ret exists."""
        from trading_bot import compute_indicators, FEATURE_COLS, FEATURE_COLS_EXT
        from lgbm_trading_bot import LGBMTradingBot

        df = _make_df(120)
        df = compute_indicators(df).dropna(subset=FEATURE_COLS).reset_index(drop=True)
        df["tfm_sma5_ret"] = 0.01  # inject fake feature

        bot = LGBMTradingBot()
        bot.fit(df)
        self.assertEqual(bot.feature_cols, FEATURE_COLS_EXT)

    def test_lgbm_falls_back_without_column(self):
        from trading_bot import compute_indicators, FEATURE_COLS
        from lgbm_trading_bot import LGBMTradingBot

        df = _make_df(120)
        df = compute_indicators(df).dropna(subset=FEATURE_COLS).reset_index(drop=True)

        bot = LGBMTradingBot()
        bot.fit(df)
        self.assertEqual(bot.feature_cols, FEATURE_COLS)


if __name__ == "__main__":
    unittest.main()
