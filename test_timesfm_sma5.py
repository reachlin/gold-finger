#!/usr/bin/env python3
"""Tests for TimesFM SMA5 predictor bot."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


def _make_df(n=200, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 10.0 + np.cumsum(rng.standard_normal(n) * 0.1)
    close = np.maximum(close, 1.0)
    sma5 = pd.Series(close).rolling(5).mean().values
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": close * (1 + rng.standard_normal(n) * 0.005),
        "high": close * (1 + np.abs(rng.standard_normal(n) * 0.01)),
        "low": close * (1 - np.abs(rng.standard_normal(n) * 0.01)),
        "close": close,
        "volume": rng.integers(100_000, 500_000, n).astype(float),
        "amount": close * 200_000,
        "amplitude": rng.random(n) * 5,
        "pct_change": rng.standard_normal(n) * 0.5,
        "change": rng.standard_normal(n) * 0.1,
        "turnover_rate": rng.random(n),
        "sma5": sma5,
    })
    return df


def _mock_model(forecast_price):
    """Return a mock TimesFM model that always forecasts forecast_price."""
    from unittest.mock import MagicMock
    mock = MagicMock()
    def forecast_fn(horizon, inputs):
        n = len(inputs)
        pts = np.full((n, horizon), forecast_price, dtype=np.float32)
        quantiles = np.zeros((n, horizon, 9), dtype=np.float32)
        return pts, quantiles
    mock.forecast.side_effect = forecast_fn
    return mock


class TestTimesFMSMA5BotSignals(unittest.TestCase):

    def setUp(self):
        self.patcher = patch(
            "timesfm_sma5_bot.TimesFMSMA5Bot._load_model", autospec=True
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_fit_stores_train_sma5(self):
        from timesfm_sma5_bot import TimesFMSMA5Bot
        bot = TimesFMSMA5Bot()
        df = _make_df(120)
        bot.fit(df)
        self.assertIsNotNone(bot._train_sma5)
        # sma5 has 4 NaN warmup rows; stored series should be shorter
        self.assertGreater(len(bot._train_sma5), 0)

    def test_fit_sets_thresholds(self):
        from timesfm_sma5_bot import TimesFMSMA5Bot
        bot = TimesFMSMA5Bot()
        bot.fit(_make_df(120))
        self.assertLess(bot.strong_sell_threshold, bot.sell_threshold)
        self.assertLess(bot.sell_threshold, bot.buy_threshold)
        self.assertLess(bot.buy_threshold, bot.strong_buy_threshold)

    def test_predict_length_matches_input(self):
        from timesfm_sma5_bot import TimesFMSMA5Bot
        bot = TimesFMSMA5Bot(context_len=64, horizon=5)
        bot._model = _mock_model(10.0)
        bot.fit(_make_df(120))
        bot._model = _mock_model(10.0)
        signals = bot.predict(_make_df(50))
        self.assertEqual(len(signals), 50)

    def test_predict_returns_valid_signals(self):
        from timesfm_sma5_bot import TimesFMSMA5Bot, SIGNAL_NAMES
        bot = TimesFMSMA5Bot(context_len=32, horizon=5)
        bot._model = _mock_model(10.0)
        bot.fit(_make_df(100))
        bot._model = _mock_model(10.0)
        for sig in bot.predict(_make_df(40)):
            self.assertIn(sig, SIGNAL_NAMES)

    def test_strong_buy_when_sma5_rises(self):
        from timesfm_sma5_bot import TimesFMSMA5Bot
        bot = TimesFMSMA5Bot(context_len=32, horizon=5)
        bot._model = _mock_model(10.0)
        bot.fit(_make_df(100))

        bot.strong_buy_threshold = 0.03
        bot.buy_threshold = 0.01
        bot.sell_threshold = -0.01
        bot.strong_sell_threshold = -0.03

        test_df = _make_df(10)
        test_df["sma5"] = 10.0  # current SMA5 = 10, forecast = +5% → strong_buy
        bot._model = _mock_model(10.5)
        signals = bot.predict(test_df)
        for sig in signals:
            self.assertEqual(sig, "strong_buy")

    def test_strong_sell_when_sma5_falls(self):
        from timesfm_sma5_bot import TimesFMSMA5Bot
        bot = TimesFMSMA5Bot(context_len=32, horizon=5)
        bot._model = _mock_model(10.0)
        bot.fit(_make_df(100))

        bot.strong_buy_threshold = 0.03
        bot.buy_threshold = 0.01
        bot.sell_threshold = -0.01
        bot.strong_sell_threshold = -0.03

        test_df = _make_df(10)
        test_df["sma5"] = 10.0
        bot._model = _mock_model(9.5)  # -5% → strong_sell
        signals = bot.predict(test_df)
        for sig in signals:
            self.assertEqual(sig, "strong_sell")

    def test_hold_when_sma5_flat(self):
        from timesfm_sma5_bot import TimesFMSMA5Bot
        bot = TimesFMSMA5Bot(context_len=32, horizon=5)
        bot._model = _mock_model(10.0)
        bot.fit(_make_df(100))

        bot.strong_buy_threshold = 0.03
        bot.buy_threshold = 0.01
        bot.sell_threshold = -0.01
        bot.strong_sell_threshold = -0.03

        test_df = _make_df(10)
        test_df["sma5"] = 10.0
        bot._model = _mock_model(10.005)  # +0.05% → hold
        signals = bot.predict(test_df)
        for sig in signals:
            self.assertEqual(sig, "hold")

    def test_uses_sma5_not_close(self):
        """Forecast is compared against sma5, not close price."""
        from timesfm_sma5_bot import TimesFMSMA5Bot
        bot = TimesFMSMA5Bot(context_len=32, horizon=5)
        bot._model = _mock_model(10.0)
        bot.fit(_make_df(100))

        bot.strong_buy_threshold = 0.03
        bot.buy_threshold = 0.01
        bot.sell_threshold = -0.01
        bot.strong_sell_threshold = -0.03

        test_df = _make_df(10)
        # close=15 (high), sma5=10.0 → forecast 10.5 is +5% vs sma5 → strong_buy
        test_df["close"] = 15.0
        test_df["sma5"] = 10.0
        bot._model = _mock_model(10.5)
        signals = bot.predict(test_df)
        for sig in signals:
            self.assertEqual(sig, "strong_buy")


class TestRunTimesFMSMA5Backtest(unittest.TestCase):

    def setUp(self):
        self.load_patcher = patch(
            "timesfm_sma5_bot.TimesFMSMA5Bot._load_model", autospec=True
        )
        self.load_patcher.start()
        self.predict_patcher = patch(
            "timesfm_sma5_bot.TimesFMSMA5Bot.predict", autospec=True
        )
        mock_predict = self.predict_patcher.start()
        mock_predict.side_effect = lambda self_bot, df: ["hold"] * len(df)

    def tearDown(self):
        self.load_patcher.stop()
        self.predict_patcher.stop()

    def test_returns_required_keys(self):
        from timesfm_sma5_bot import run_timesfm_sma5_backtest
        result = run_timesfm_sma5_backtest(_make_df(200))
        for key in ["total_return", "buy_and_hold_return", "max_drawdown",
                    "sharpe_ratio", "win_rate", "profit_factor",
                    "num_trades", "final_value", "trades", "bot", "test_df"]:
            self.assertIn(key, result)

    def test_zero_trades_on_all_hold(self):
        from timesfm_sma5_bot import run_timesfm_sma5_backtest
        result = run_timesfm_sma5_backtest(_make_df(200))
        self.assertEqual(result["num_trades"], 0)

    def test_test_df_has_signal_column(self):
        from timesfm_sma5_bot import run_timesfm_sma5_backtest
        result = run_timesfm_sma5_backtest(_make_df(200))
        self.assertIn("signal", result["test_df"].columns)

    def test_capital_non_negative(self):
        from timesfm_sma5_bot import run_timesfm_sma5_backtest
        result = run_timesfm_sma5_backtest(_make_df(200))
        self.assertGreater(result["final_value"], 0)


if __name__ == "__main__":
    unittest.main()
