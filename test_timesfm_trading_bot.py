#!/usr/bin/env python3
"""Tests for TimesFM trading bot.

Mocks the model to avoid slow downloads/inference during unit tests.
Run integration test with --integration flag to use the real model.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


def _make_df(n=200):
    """Generate synthetic OHLCV DataFrame with indicators."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 10.0 + np.cumsum(np.random.randn(n) * 0.1)
    close = np.maximum(close, 1.0)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": close * (1 + np.random.randn(n) * 0.005),
        "high": close * (1 + np.abs(np.random.randn(n) * 0.01)),
        "low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
        "close": close,
        "volume": np.random.randint(100_000, 500_000, n).astype(float),
        "amount": close * 200_000,
        "amplitude": np.random.rand(n) * 5,
        "pct_change": np.random.randn(n) * 0.5,
        "change": np.random.randn(n) * 0.1,
        "turnover_rate": np.random.rand(n),
    })
    return df


def _make_mock_model(forecast_values):
    """Create a mock TimesFM model that returns given forecast_values per call."""
    mock = MagicMock()
    # forecast returns (point_forecast, quantile_forecast)
    # point_forecast shape: (batch_size, horizon)
    def forecast_fn(horizon, inputs):
        n = len(inputs)
        pts = np.array([[forecast_values[i % len(forecast_values)]] * horizon
                        for i in range(n)], dtype=np.float32)
        quantiles = np.zeros((n, horizon, 9), dtype=np.float32)
        return pts, quantiles
    mock.forecast.side_effect = forecast_fn
    return mock


class TestTimesFMTradingBotSignals(unittest.TestCase):
    """Test signal generation logic with mocked model."""

    def setUp(self):
        # Patch _load_model so no HuggingFace download happens
        self.patcher = patch(
            "timesfm_trading_bot.TimesFMTradingBot._load_model",
            autospec=True,
        )
        self.mock_load = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def _make_bot_with_model(self, mock_model):
        from timesfm_trading_bot import TimesFMTradingBot
        bot = TimesFMTradingBot(context_len=64, horizon=1)
        bot._model = mock_model
        return bot

    def test_fit_stores_train_close(self):
        from timesfm_trading_bot import TimesFMTradingBot
        bot = TimesFMTradingBot()
        df = _make_df(120)
        bot.fit(df)
        self.assertIsNotNone(bot._train_close)
        self.assertEqual(len(bot._train_close), 120)

    def test_fit_sets_thresholds(self):
        from timesfm_trading_bot import TimesFMTradingBot
        bot = TimesFMTradingBot()
        df = _make_df(120)
        bot.fit(df)
        self.assertLess(bot.strong_sell_threshold, bot.sell_threshold)
        self.assertLess(bot.sell_threshold, bot.buy_threshold)
        self.assertLess(bot.buy_threshold, bot.strong_buy_threshold)

    def test_predict_length_matches_input(self):
        from timesfm_trading_bot import TimesFMTradingBot
        n_train, n_test = 120, 80
        train_df = _make_df(n_train)
        test_df = _make_df(n_test)

        bot = TimesFMTradingBot(context_len=64, horizon=1)
        bot._model = _make_mock_model([10.0])
        bot.fit(train_df)
        bot._model = _make_mock_model([10.0])  # reset after fit

        signals = bot.predict(test_df)
        self.assertEqual(len(signals), n_test)

    def test_predict_returns_valid_signal_names(self):
        from timesfm_trading_bot import TimesFMTradingBot, SIGNAL_NAMES
        train_df = _make_df(100)
        test_df = _make_df(50)

        bot = TimesFMTradingBot(context_len=32, horizon=1)
        bot._model = _make_mock_model([10.0])
        bot.fit(train_df)
        bot._model = _make_mock_model([10.0])

        signals = bot.predict(test_df)
        for sig in signals:
            self.assertIn(sig, SIGNAL_NAMES)

    def test_strong_buy_when_forecast_much_higher(self):
        """If model predicts price will rise sharply, expect strong_buy."""
        from timesfm_trading_bot import TimesFMTradingBot
        train_df = _make_df(100)
        test_df = _make_df(10)

        # Force all test close prices to 10.0
        test_df["close"] = 10.0

        # Model always forecasts 10.5 (+5%), well above thresholds
        bot = TimesFMTradingBot(context_len=32, horizon=1)
        bot._model = _make_mock_model([10.0])
        bot.fit(train_df)

        # Override thresholds to known values
        bot.strong_buy_threshold = 0.03
        bot.buy_threshold = 0.01
        bot.sell_threshold = -0.01
        bot.strong_sell_threshold = -0.03

        bot._model = _make_mock_model([10.5])  # +5%
        signals = bot.predict(test_df)
        for sig in signals:
            self.assertEqual(sig, "strong_buy")

    def test_strong_sell_when_forecast_much_lower(self):
        """If model predicts price will drop sharply, expect strong_sell."""
        from timesfm_trading_bot import TimesFMTradingBot
        train_df = _make_df(100)
        test_df = _make_df(10)
        test_df["close"] = 10.0

        bot = TimesFMTradingBot(context_len=32, horizon=1)
        bot._model = _make_mock_model([10.0])
        bot.fit(train_df)

        bot.strong_buy_threshold = 0.03
        bot.buy_threshold = 0.01
        bot.sell_threshold = -0.01
        bot.strong_sell_threshold = -0.03

        bot._model = _make_mock_model([9.5])  # -5%
        signals = bot.predict(test_df)
        for sig in signals:
            self.assertEqual(sig, "strong_sell")

    def test_hold_when_forecast_flat(self):
        """If model predicts negligible change, expect hold."""
        from timesfm_trading_bot import TimesFMTradingBot
        train_df = _make_df(100)
        test_df = _make_df(10)
        test_df["close"] = 10.0

        bot = TimesFMTradingBot(context_len=32, horizon=1)
        bot._model = _make_mock_model([10.0])
        bot.fit(train_df)

        bot.strong_buy_threshold = 0.03
        bot.buy_threshold = 0.01
        bot.sell_threshold = -0.01
        bot.strong_sell_threshold = -0.03

        bot._model = _make_mock_model([10.001])  # ~0.01% change, within hold band
        signals = bot.predict(test_df)
        for sig in signals:
            self.assertEqual(sig, "hold")


class TestRunTimesFMBacktest(unittest.TestCase):
    """Test backtest runner with mocked model."""

    def setUp(self):
        self.patcher = patch(
            "timesfm_trading_bot.TimesFMTradingBot._load_model",
            autospec=True,
        )
        self.mock_load = self.patcher.start()

        # Also patch the model's forecast call globally after fit
        self.forecast_patcher = patch(
            "timesfm_trading_bot.TimesFMTradingBot.predict",
            autospec=True,
        )
        self.mock_predict = self.forecast_patcher.start()
        # Return all holds by default
        self.mock_predict.side_effect = lambda self_bot, df: ["hold"] * len(df)

    def tearDown(self):
        self.patcher.stop()
        self.forecast_patcher.stop()

    def test_backtest_returns_required_keys(self):
        from timesfm_trading_bot import run_timesfm_backtest
        df = _make_df(200)
        result = run_timesfm_backtest(df, train_ratio=0.6, initial_capital=100_000)

        required_keys = [
            "total_return", "buy_and_hold_return", "max_drawdown",
            "sharpe_ratio", "win_rate", "profit_factor",
            "num_trades", "final_value", "trades", "bot", "test_df",
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_backtest_zero_trades_on_all_hold(self):
        from timesfm_trading_bot import run_timesfm_backtest
        df = _make_df(200)
        result = run_timesfm_backtest(df, train_ratio=0.6, initial_capital=100_000)
        self.assertEqual(result["num_trades"], 0)
        self.assertEqual(result["trades"], [])

    def test_backtest_trades_on_buy_signals(self):
        from timesfm_trading_bot import run_timesfm_backtest
        df = _make_df(200)
        # Alternate strong_buy / strong_sell signals
        def alternating_signals(self_bot, test_df):
            signals = []
            for i in range(len(test_df)):
                signals.append("strong_buy" if i % 4 < 2 else "strong_sell")
            return signals
        self.mock_predict.side_effect = alternating_signals

        result = run_timesfm_backtest(df, train_ratio=0.6, initial_capital=100_000)
        self.assertGreater(result["num_trades"], 0)

    def test_backtest_capital_non_negative(self):
        from timesfm_trading_bot import run_timesfm_backtest
        df = _make_df(200)
        result = run_timesfm_backtest(df, train_ratio=0.6, initial_capital=100_000)
        self.assertGreater(result["final_value"], 0)

    def test_test_df_has_signal_column(self):
        from timesfm_trading_bot import run_timesfm_backtest
        df = _make_df(200)
        result = run_timesfm_backtest(df, train_ratio=0.6, initial_capital=100_000)
        self.assertIn("signal", result["test_df"].columns)


if __name__ == "__main__":
    unittest.main()
