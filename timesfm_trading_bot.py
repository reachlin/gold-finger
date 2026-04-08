#!/usr/bin/env python3
"""TimesFM foundation model trading bot for China A-shares.

Uses Google's pretrained TimesFM 2.5 (200M params) for zero-shot time-series
forecasting. Predicts next-day close price from a sliding context window of
historical closes, then converts the forecast return into a 5-class signal.

No supervised training is required — only the training window is stored as
context for the foundation model.
"""

import os
import sys

import numpy as np
import pandas as pd
import joblib

from trading_bot import (
    LOT_SIZE,
    Portfolio,
    compute_indicators,
    FEATURE_COLS,
)

# Path to local timesfm clone
_TIMESFM_SRC = "/Users/lincai/dev/3rd-party/timesfm/src"

SIGNAL_NAMES = ["strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"]


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------
class TimesFMTradingBot:
    """Zero-shot trading bot using TimesFM 2.5 for 1-step price forecasting.

    fit()    — stores training close prices and calibrates signal thresholds
    predict() — runs batched TimesFM inference; returns one signal per row
    """

    def __init__(
        self,
        context_len: int = 512,
        horizon: int = 1,
    ):
        self.context_len = context_len
        self.horizon = horizon
        self._train_close: np.ndarray | None = None
        self._model = None

        # Calibrated from training return percentiles
        self.strong_buy_threshold: float = 0.02
        self.buy_threshold: float = 0.005
        self.sell_threshold: float = -0.005
        self.strong_sell_threshold: float = -0.02

    def _load_model(self):
        """Lazy-load and compile TimesFM 2.5."""
        if _TIMESFM_SRC not in sys.path:
            sys.path.insert(0, _TIMESFM_SRC)
        import timesfm  # noqa: PLC0415
        print("Loading TimesFM 2.5 (200M)…")
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
        model.compile(
            timesfm.ForecastConfig(
                max_context=self.context_len,
                max_horizon=self.horizon,
                normalize_inputs=True,
                per_core_batch_size=32,
            )
        )
        self._model = model
        print("TimesFM loaded and compiled.")

    def fit(self, df: pd.DataFrame):
        """Store training close prices and calibrate return thresholds.

        TimesFM is a pretrained foundation model — no gradient updates occur.
        Thresholds are set from the empirical return distribution of the
        training split so signals scale sensibly to each stock's volatility.
        """
        closes = df["close"].values.astype(np.float32)
        self._train_close = closes

        returns = np.diff(closes) / closes[:-1]
        self.strong_sell_threshold = float(np.percentile(returns, 15))
        self.sell_threshold = float(np.percentile(returns, 30))
        self.buy_threshold = float(np.percentile(returns, 70))
        self.strong_buy_threshold = float(np.percentile(returns, 85))

        if self._model is None:
            self._load_model()

    def predict(self, df: pd.DataFrame) -> list[str]:
        """Forecast next-day return for every row; return signal per row.

        For row i the context is: [train_close + test_close[:i]] trimmed to
        context_len.  All N contexts are batched into a single model call.
        """
        assert self._train_close is not None, "Call fit() before predict()"

        test_close = df["close"].values.astype(np.float32)
        combined = np.concatenate([self._train_close, test_close])

        inputs = []
        for i in range(len(test_close)):
            end_idx = len(self._train_close) + i
            start_idx = max(0, end_idx - self.context_len)
            inputs.append(combined[start_idx:end_idx])

        point_forecast, _ = self._model.forecast(
            horizon=self.horizon, inputs=inputs
        )

        signals = []
        for i, forecast in enumerate(point_forecast):
            current_price = test_close[i]
            next_price = float(forecast[0])
            if current_price <= 0:
                signals.append("hold")
                continue
            ret = (next_price - current_price) / current_price

            if ret >= self.strong_buy_threshold:
                signals.append("strong_buy")
            elif ret >= self.buy_threshold:
                signals.append("mild_buy")
            elif ret <= self.strong_sell_threshold:
                signals.append("strong_sell")
            elif ret <= self.sell_threshold:
                signals.append("mild_sell")
            else:
                signals.append("hold")

        return signals

    def save(self, path: str):
        """Save calibrated thresholds and training context to a joblib file."""
        joblib.dump({
            "context_len": self.context_len,
            "horizon": self.horizon,
            "train_close": self._train_close,
            "strong_buy_threshold": self.strong_buy_threshold,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "strong_sell_threshold": self.strong_sell_threshold,
        }, path)

    @classmethod
    def load(cls, path: str) -> "TimesFMTradingBot":
        """Load a TimesFMTradingBot from a joblib file (model loaded lazily)."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        data = joblib.load(path)
        bot = cls(context_len=data["context_len"], horizon=data["horizon"])
        bot._train_close = data["train_close"]
        bot.strong_buy_threshold = data["strong_buy_threshold"]
        bot.buy_threshold = data["buy_threshold"]
        bot.sell_threshold = data["sell_threshold"]
        bot.strong_sell_threshold = data["strong_sell_threshold"]
        return bot


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def run_timesfm_backtest(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    initial_capital: float = 100_000,
    context_len: int = 512,
    horizon: int = 1,
    buy_pct: int | None = None,
    strong_spread: int | None = None,
) -> dict:
    """Walk-forward backtest: context from training split, test on rest.

    Returns metrics dict compatible with compare_models.py comparison table.
    """
    df = compute_indicators(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    split = int(len(df) * train_ratio)
    train_df = df.iloc[:split].copy().reset_index(drop=True)
    test_df = df.iloc[split:].copy().reset_index(drop=True)

    bot = TimesFMTradingBot(context_len=context_len, horizon=horizon)
    bot.fit(train_df)

    # Override thresholds if tuned params are provided
    if buy_pct is not None:
        closes = train_df["close"].values
        returns = np.diff(closes) / closes[:-1]
        spread = strong_spread if strong_spread is not None else 10
        sell_pct = 100 - buy_pct
        bot.buy_threshold = float(np.percentile(returns, buy_pct))
        bot.sell_threshold = float(np.percentile(returns, sell_pct))
        bot.strong_buy_threshold = float(np.percentile(returns, min(99, buy_pct + spread)))
        bot.strong_sell_threshold = float(np.percentile(returns, max(1, sell_pct - spread)))

    signals = bot.predict(test_df)
    test_df["signal"] = signals

    portfolio = Portfolio(capital=initial_capital)
    trades = []
    daily_values = []

    for i in range(len(test_df) - 1):
        signal = test_df.loc[i, "signal"]
        exec_price = test_df.loc[i + 1, "open"]
        trade_date = str(test_df.loc[i + 1, "date"])
        price_below_sma5 = test_df.loc[i, "close"] < test_df.loc[i, "sma5"]

        shares_traded = 0
        action = "hold"

        if signal == "strong_buy" and price_below_sma5:
            shares_traded = portfolio.buy(exec_price, fraction=1.0, trade_date=trade_date)
            if shares_traded > 0:
                action = "buy"
        elif signal == "mild_buy" and price_below_sma5:
            shares_traded = portfolio.buy(exec_price, fraction=0.5, trade_date=trade_date)
            if shares_traded > 0:
                action = "buy"
        elif signal == "strong_sell":
            shares_traded = portfolio.sell(exec_price, fraction=1.0, trade_date=trade_date)
            if shares_traded > 0:
                action = "sell"
        elif signal == "mild_sell":
            shares_traded = portfolio.sell(exec_price, fraction=0.5, trade_date=trade_date)
            if shares_traded > 0:
                action = "sell"

        if shares_traded > 0:
            trades.append({
                "date": trade_date,
                "action": action,
                "price": exec_price,
                "shares": shares_traded,
                "signal": signal,
            })

        daily_values.append(portfolio.value(test_df.loc[i + 1, "close"]))

    final_price = test_df.iloc[-1]["close"]
    final_value = portfolio.value(final_price)

    bh_shares = int(initial_capital / test_df.iloc[0]["open"] // LOT_SIZE) * LOT_SIZE
    bh_cost = bh_shares * test_df.iloc[0]["open"]
    bh_value = bh_shares * final_price + (initial_capital - bh_cost)
    bh_return = (bh_value - initial_capital) / initial_capital * 100

    total_return = (final_value - initial_capital) / initial_capital * 100
    values = np.array([initial_capital] + daily_values)
    peak = np.maximum.accumulate(values)
    drawdowns = (values - peak) / peak
    max_drawdown = drawdowns.min() * 100

    daily_returns = np.diff(values) / values[:-1]
    sharpe = (
        np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        if np.std(daily_returns) > 0
        else 0.0
    )

    buy_trades = [t for t in trades if t["action"] == "buy"]
    sell_trades = [t for t in trades if t["action"] == "sell"]
    trade_pnl = []
    for i in range(min(len(buy_trades), len(sell_trades))):
        pnl = (sell_trades[i]["price"] - buy_trades[i]["price"]) * buy_trades[i]["shares"]
        trade_pnl.append(pnl)

    wins = [p for p in trade_pnl if p > 0]
    losses = [p for p in trade_pnl if p <= 0]
    win_rate = len(wins) / len(trade_pnl) * 100 if trade_pnl else 0
    profit_factor = (
        sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    )

    return {
        "total_return": total_return,
        "buy_and_hold_return": bh_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": len(trades),
        "final_value": final_value,
        "trades": trades,
        "daily_values": daily_values,
        "bot": bot,
        "test_df": test_df,
        "train_end_idx": split,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="TimesFM trading bot backtest")
    parser.add_argument("--csv", default="data/601933_10yr.csv", help="CSV file path")
    parser.add_argument("--context-len", type=int, default=512,
                        help="Context length for TimesFM")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}\n")

    print("=" * 60)
    print("TIMESFM TRADING BOT")
    print("=" * 60)
    results = run_timesfm_backtest(
        df, train_ratio=0.6, initial_capital=100_000,
        context_len=args.context_len,
    )

    print(f"\nTotal Return:   {results['total_return']:+.2f}%")
    print(f"Buy & Hold:     {results['buy_and_hold_return']:+.2f}%")
    print(f"Max Drawdown:   {results['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio:   {results['sharpe_ratio']:.3f}")
    print(f"Win Rate:       {results['win_rate']:.1f}%")
    print(f"Profit Factor:  {results['profit_factor']:.2f}")
    print(f"Num Trades:     {results['num_trades']}")
    print(f"Final Value:    {results['final_value']:,.0f}")

    trades = results["trades"]
    if trades:
        print(f"\nRecent Trades (last 10):")
        for t in trades[-10:]:
            print(f"  {t['date']}  {t['action']:4s}  {t['shares']:6d} shares @ "
                  f"{t['price']:.2f}  ({t['signal']})")


if __name__ == "__main__":
    main()
