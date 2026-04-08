"""Tests for model save/load across all bot classes."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from trading_bot import TradingBot, compute_indicators, FEATURE_COLS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def make_ohlcv(n=200, seed=42):
    """Create synthetic OHLCV data sufficient for all bots."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 5.0 + np.cumsum(rng.randn(n) * 0.05)
    close = np.maximum(close, 1.0)
    high = close + rng.uniform(0.01, 0.1, n)
    low = close - rng.uniform(0.01, 0.1, n)
    low = np.maximum(low, 0.5)
    open_ = close + rng.uniform(-0.05, 0.05, n)
    volume = rng.randint(100000, 1000000, n).astype(float)
    return pd.DataFrame({
        "date": dates, "open": open_, "close": close,
        "high": high, "low": low, "volume": volume,
    })


@pytest.fixture
def train_df():
    """Return a DataFrame with indicators computed and NaN dropped."""
    df = make_ohlcv(200)
    df = compute_indicators(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df


# ===========================================================================
# TradingBot (K-Means)
# ===========================================================================
class TestTradingBotSaveLoad:
    def test_roundtrip(self, train_df, tmp_path):
        bot = TradingBot(n_clusters=5)
        bot.fit(train_df)
        orig_signals = bot.predict(train_df)

        path = str(tmp_path / "kmeans.joblib")
        bot.save(path)
        assert os.path.exists(path)

        loaded = TradingBot.load(path)
        loaded_signals = loaded.predict(train_df)
        assert orig_signals == loaded_signals

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TradingBot.load(str(tmp_path / "nope.joblib"))


# ===========================================================================
# DNNTradingBot (LSTM)
# ===========================================================================
class TestDNNTradingBotSaveLoad:
    def test_roundtrip(self, train_df, tmp_path):
        from dnn_trading_bot import DNNTradingBot

        bot = DNNTradingBot(window_size=10, epochs=2, batch_size=16)
        bot.fit(train_df)
        orig_signals = bot.predict(train_df)

        path = str(tmp_path / "lstm.pt")
        bot.save(path)
        assert os.path.exists(path)

        loaded = DNNTradingBot.load(path)
        loaded_signals = loaded.predict(train_df)
        assert orig_signals == loaded_signals

    def test_load_missing_file(self, tmp_path):
        from dnn_trading_bot import DNNTradingBot
        with pytest.raises(FileNotFoundError):
            DNNTradingBot.load(str(tmp_path / "nope.pt"))


# ===========================================================================
# LGBMTradingBot
# ===========================================================================
class TestLGBMTradingBotSaveLoad:
    def test_roundtrip(self, train_df, tmp_path):
        from lgbm_trading_bot import LGBMTradingBot

        bot = LGBMTradingBot(n_estimators=10)
        bot.fit(train_df)
        orig_signals = bot.predict(train_df)

        path = str(tmp_path / "lgbm.joblib")
        bot.save(path)
        assert os.path.exists(path)

        loaded = LGBMTradingBot.load(path)
        loaded_signals = loaded.predict(train_df)
        assert orig_signals == loaded_signals

    def test_load_missing_file(self, tmp_path):
        from lgbm_trading_bot import LGBMTradingBot
        with pytest.raises(FileNotFoundError):
            LGBMTradingBot.load(str(tmp_path / "nope.joblib"))


# ===========================================================================
# PPOTradingBot
# ===========================================================================
class TestPPOTradingBotSaveLoad:
    def test_roundtrip(self, train_df, tmp_path):
        from ppo_trading_bot import PPOTradingBot

        bot = PPOTradingBot(total_timesteps=500)
        bot.fit(train_df)
        orig_signals = bot.predict(train_df)

        model_dir = str(tmp_path / "ppo")
        os.makedirs(model_dir)
        bot.save(model_dir)
        assert os.path.exists(os.path.join(model_dir, "ppo_model.zip"))
        assert os.path.exists(os.path.join(model_dir, "ppo_scaler.joblib"))

        loaded = PPOTradingBot.load(model_dir)
        loaded_signals = loaded.predict(train_df)
        assert orig_signals == loaded_signals

    def test_load_missing_dir(self, tmp_path):
        from ppo_trading_bot import PPOTradingBot
        with pytest.raises(FileNotFoundError):
            PPOTradingBot.load(str(tmp_path / "nope"))


# ===========================================================================
# TD3TradingBot
# ===========================================================================
class TestTD3TradingBotSaveLoad:
    def test_roundtrip(self, train_df, tmp_path):
        from td3_trading_bot import TD3TradingBot, _get_base_signals
        from dnn_trading_bot import DNNTradingBot
        from lgbm_trading_bot import LGBMTradingBot
        from ppo_trading_bot import PPOTradingBot

        # TD3 needs base model signals in the df
        km_bot = TradingBot(n_clusters=5)
        km_bot.fit(train_df)
        lstm_bot = DNNTradingBot(window_size=10, epochs=2, batch_size=16)
        lstm_bot.fit(train_df)
        lgbm_bot = LGBMTradingBot(n_estimators=10)
        lgbm_bot.fit(train_df)
        ppo_bot = PPOTradingBot(total_timesteps=500)
        ppo_bot.fit(train_df)

        aug_df = _get_base_signals(train_df, km_bot, lstm_bot, lgbm_bot, ppo_bot)

        bot = TD3TradingBot(total_timesteps=500)
        bot.fit(aug_df)
        orig_signals = bot.predict(aug_df)

        model_dir = str(tmp_path / "td3")
        os.makedirs(model_dir)
        bot.save(model_dir)
        assert os.path.exists(os.path.join(model_dir, "td3_model.zip"))
        assert os.path.exists(os.path.join(model_dir, "td3_scaler.joblib"))

        loaded = TD3TradingBot.load(model_dir)
        loaded_signals = loaded.predict(aug_df)
        assert orig_signals == loaded_signals

    def test_load_missing_dir(self, tmp_path):
        from td3_trading_bot import TD3TradingBot
        with pytest.raises(FileNotFoundError):
            TD3TradingBot.load(str(tmp_path / "nope"))


# ===========================================================================
# TimesFMTradingBot (mock model to avoid HF download)
# ===========================================================================
class TestTimesFMTradingBotSaveLoad:
    def test_roundtrip(self, train_df, tmp_path):
        from timesfm_trading_bot import TimesFMTradingBot
        from unittest.mock import patch, MagicMock

        bot = TimesFMTradingBot(context_len=64)
        # Manually set fit state without loading the real model
        closes = train_df["close"].values.astype(np.float32)
        bot._train_close = closes
        returns = np.diff(closes) / closes[:-1]
        bot.strong_sell_threshold = float(np.percentile(returns, 15))
        bot.sell_threshold = float(np.percentile(returns, 30))
        bot.buy_threshold = float(np.percentile(returns, 70))
        bot.strong_buy_threshold = float(np.percentile(returns, 85))

        path = str(tmp_path / "timesfm.joblib")
        bot.save(path)
        assert os.path.exists(path)

        loaded = TimesFMTradingBot.load(path)
        assert loaded.context_len == 64
        assert np.array_equal(loaded._train_close, bot._train_close)
        assert loaded.buy_threshold == bot.buy_threshold
        assert loaded.strong_buy_threshold == bot.strong_buy_threshold
        assert loaded.sell_threshold == bot.sell_threshold
        assert loaded.strong_sell_threshold == bot.strong_sell_threshold

    def test_load_missing_file(self, tmp_path):
        from timesfm_trading_bot import TimesFMTradingBot
        with pytest.raises(FileNotFoundError):
            TimesFMTradingBot.load(str(tmp_path / "nope.joblib"))


# ===========================================================================
# TimesFMSMA5Bot (mock model to avoid HF download)
# ===========================================================================
class TestTimesFMSMA5BotSaveLoad:
    def test_roundtrip(self, train_df, tmp_path):
        from timesfm_sma5_bot import TimesFMSMA5Bot

        bot = TimesFMSMA5Bot(context_len=64, horizon=5)
        sma5 = train_df["sma5"].dropna().values.astype(np.float32)
        bot._train_sma5 = sma5
        returns = np.diff(sma5) / sma5[:-1]
        bot.strong_sell_threshold = float(np.percentile(returns, 15))
        bot.sell_threshold = float(np.percentile(returns, 30))
        bot.buy_threshold = float(np.percentile(returns, 70))
        bot.strong_buy_threshold = float(np.percentile(returns, 85))

        path = str(tmp_path / "timesfm_sma5.joblib")
        bot.save(path)
        assert os.path.exists(path)

        loaded = TimesFMSMA5Bot.load(path)
        assert loaded.context_len == 64
        assert loaded.horizon == 5
        assert np.array_equal(loaded._train_sma5, bot._train_sma5)
        assert loaded.buy_threshold == bot.buy_threshold

    def test_load_missing_file(self, tmp_path):
        from timesfm_sma5_bot import TimesFMSMA5Bot
        with pytest.raises(FileNotFoundError):
            TimesFMSMA5Bot.load(str(tmp_path / "nope.joblib"))


# ===========================================================================
# Metadata save/load
# ===========================================================================
class TestMetadataSaveLoad:
    def test_metadata_roundtrip(self, tmp_path):
        metadata = {
            "symbol": "601933",
            "train_date": "2024-01-15",
            "data_range": {"start": "2016-01-01", "end": "2024-01-15"},
            "metrics": {"km_sharpe": 0.5, "lgbm_sharpe": 0.8},
        }
        path = str(tmp_path / "metadata.json")
        with open(path, "w") as f:
            json.dump(metadata, f)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == metadata
