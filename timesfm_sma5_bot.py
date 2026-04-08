#!/usr/bin/env python3
"""TimesFM trading bot using 5-day moving average (SMA5) prediction.

Instead of forecasting raw close prices (which are noisy day-to-day),
this bot feeds TimesFM the smoothed SMA5 series and forecasts where SMA5
is heading.  The predicted SMA5 direction is more stable than single-day
price changes, reducing overtrading.

Signal logic:
  forecast_sma5 vs current_sma5 → pct change
  >= strong_buy_threshold  → strong_buy
  >= buy_threshold          → mild_buy
  <= strong_sell_threshold  → strong_sell
  <= sell_threshold         → mild_sell
  else                      → hold
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

_TIMESFM_SRC = "/Users/lincai/dev/3rd-party/timesfm/src"

SIGNAL_NAMES = ["strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"]


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------
class TimesFMSMA5Bot:
    """Zero-shot TimesFM bot that forecasts SMA5 instead of raw close price.

    fit()    — stores training SMA5 series; calibrates thresholds from its
               return distribution (smoother than raw close returns)
    predict() — batch-forecasts next SMA5 value for every test row using a
               sliding context window; returns one signal per row
    """

    def __init__(self, context_len: int = 512, horizon: int = 5):
        self.context_len = context_len
        self.horizon = horizon          # forecast H steps; use last step as signal
        self._train_sma5: np.ndarray | None = None
        self._model = None

        self.strong_buy_threshold: float = 0.01
        self.buy_threshold: float = 0.003
        self.sell_threshold: float = -0.003
        self.strong_sell_threshold: float = -0.01

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
        """Store training SMA5 and calibrate thresholds from its returns.

        SMA5 must already exist in df (added by compute_indicators).
        """
        sma5 = df["sma5"].dropna().values.astype(np.float32)
        self._train_sma5 = sma5

        # Thresholds from SMA5 return distribution (smoother than close returns)
        returns = np.diff(sma5) / sma5[:-1]
        self.strong_sell_threshold = float(np.percentile(returns, 15))
        self.sell_threshold = float(np.percentile(returns, 30))
        self.buy_threshold = float(np.percentile(returns, 70))
        self.strong_buy_threshold = float(np.percentile(returns, 85))

        if self._model is None:
            self._load_model()

    def predict(self, df: pd.DataFrame) -> list[str]:
        """Forecast next SMA5 for each row; return one signal per row.

        Context for row i: [train_sma5 + test_sma5[:i]] trimmed to context_len.
        All N contexts are batched into a single model call.
        NaN SMA5 rows (first 4 rows of any series) receive 'hold'.
        """
        assert self._train_sma5 is not None, "Call fit() first"

        test_sma5_series = df["sma5"].values.astype(np.float64)
        # Combine train + test SMA5 for sliding context
        combined = np.concatenate([self._train_sma5,
                                   test_sma5_series[~np.isnan(test_sma5_series)]])

        # Build per-row context inputs; track which rows have valid SMA5
        inputs = []
        valid_idx = []  # indices in df that have non-NaN sma5
        train_len = len(self._train_sma5)

        test_valid_count = 0
        for i in range(len(df)):
            if np.isnan(test_sma5_series[i]):
                continue
            end_idx = train_len + test_valid_count
            start_idx = max(0, end_idx - self.context_len)
            inputs.append(combined[start_idx:end_idx].astype(np.float32))
            valid_idx.append(i)
            test_valid_count += 1

        if not inputs:
            return ["hold"] * len(df)

        point_forecast, _ = self._model.forecast(
            horizon=self.horizon, inputs=inputs
        )

        # Build signal list; default hold for NaN rows
        signals = ["hold"] * len(df)
        for k, row_i in enumerate(valid_idx):
            current_sma5 = test_sma5_series[row_i]
            # Use the last horizon step as the "end of window" prediction
            forecast_sma5 = float(point_forecast[k, -1])
            if current_sma5 <= 0:
                continue
            ret = (forecast_sma5 - current_sma5) / current_sma5

            if ret >= self.strong_buy_threshold:
                signals[row_i] = "strong_buy"
            elif ret >= self.buy_threshold:
                signals[row_i] = "mild_buy"
            elif ret <= self.strong_sell_threshold:
                signals[row_i] = "strong_sell"
            elif ret <= self.sell_threshold:
                signals[row_i] = "mild_sell"
            # else: hold (already set)

        return signals

    def save(self, path: str):
        """Save calibrated thresholds and training SMA5 context to a joblib file."""
        joblib.dump({
            "context_len": self.context_len,
            "horizon": self.horizon,
            "train_sma5": self._train_sma5,
            "strong_buy_threshold": self.strong_buy_threshold,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "strong_sell_threshold": self.strong_sell_threshold,
        }, path)

    @classmethod
    def load(cls, path: str) -> "TimesFMSMA5Bot":
        """Load a TimesFMSMA5Bot from a joblib file (model loaded lazily)."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        data = joblib.load(path)
        bot = cls(context_len=data["context_len"], horizon=data["horizon"])
        bot._train_sma5 = data["train_sma5"]
        bot.strong_buy_threshold = data["strong_buy_threshold"]
        bot.buy_threshold = data["buy_threshold"]
        bot.sell_threshold = data["sell_threshold"]
        bot.strong_sell_threshold = data["strong_sell_threshold"]
        return bot


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def run_timesfm_sma5_backtest(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    initial_capital: float = 100_000,
    context_len: int = 512,
    horizon: int = 5,
) -> dict:
    """Walk-forward backtest using SMA5-based TimesFM signals."""
    df = compute_indicators(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    split = int(len(df) * train_ratio)
    train_df = df.iloc[:split].copy().reset_index(drop=True)
    test_df = df.iloc[split:].copy().reset_index(drop=True)

    bot = TimesFMSMA5Bot(context_len=context_len, horizon=horizon)
    bot.fit(train_df)

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
    max_drawdown = ((values - peak) / peak).min() * 100

    daily_returns = np.diff(values) / values[:-1]
    sharpe = (
        np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        if np.std(daily_returns) > 0 else 0.0
    )

    buy_trades = [t for t in trades if t["action"] == "buy"]
    sell_trades = [t for t in trades if t["action"] == "sell"]
    trade_pnl = [
        (sell_trades[i]["price"] - buy_trades[i]["price"]) * buy_trades[i]["shares"]
        for i in range(min(len(buy_trades), len(sell_trades)))
    ]
    wins = [p for p in trade_pnl if p > 0]
    losses = [p for p in trade_pnl if p <= 0]
    win_rate = len(wins) / len(trade_pnl) * 100 if trade_pnl else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

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

    parser = argparse.ArgumentParser(description="TimesFM SMA5 trading bot backtest")
    parser.add_argument("--csv", default="data/601933_10yr.csv")
    parser.add_argument("--horizon", type=int, default=5,
                        help="SMA5 forecast horizon (steps ahead)")
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--compare", action="store_true",
                        help="Also run raw-close TimesFM for side-by-side comparison")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}\n")

    print("=" * 60)
    print(f"TIMESFM SMA5 BOT  (horizon={args.horizon})")
    print("=" * 60)
    sma5_results = run_timesfm_sma5_backtest(
        df, train_ratio=0.6, initial_capital=100_000,
        context_len=args.context_len, horizon=args.horizon,
    )

    if args.compare:
        from timesfm_trading_bot import run_timesfm_backtest
        print(f"\n{'=' * 60}")
        print("TIMESFM RAW-CLOSE BOT  (horizon=1, for comparison)")
        print("=" * 60)
        raw_results = run_timesfm_backtest(
            df, train_ratio=0.6, initial_capital=100_000,
            context_len=args.context_len,
        )

    # --- Results table ---
    print(f"\n{'=' * 70}")
    if args.compare:
        print(f"  {'Metric':<22s} {'SMA5 (h=%d)' % args.horizon:>15s} {'Raw Close':>15s} {'Buy&Hold':>12s}")
        print("  " + "-" * 68)
        bh = sma5_results["buy_and_hold_return"]
        rows = [
            ("Total Return",
             f"{sma5_results['total_return']:+.2f}%",
             f"{raw_results['total_return']:+.2f}%",
             f"{bh:+.2f}%"),
            ("Max Drawdown",
             f"{sma5_results['max_drawdown']:.2f}%",
             f"{raw_results['max_drawdown']:.2f}%", "N/A"),
            ("Sharpe Ratio",
             f"{sma5_results['sharpe_ratio']:.3f}",
             f"{raw_results['sharpe_ratio']:.3f}", "N/A"),
            ("Win Rate",
             f"{sma5_results['win_rate']:.1f}%",
             f"{raw_results['win_rate']:.1f}%", "N/A"),
            ("Profit Factor",
             f"{sma5_results['profit_factor']:.2f}",
             f"{raw_results['profit_factor']:.2f}", "N/A"),
            ("Num Trades",
             f"{sma5_results['num_trades']}",
             f"{raw_results['num_trades']}", "1"),
            ("Final Value",
             f"{sma5_results['final_value']:,.0f}",
             f"{raw_results['final_value']:,.0f}", "N/A"),
        ]
        for label, s, r, bh_v in rows:
            print(f"  {label:<22s} {s:>15s} {r:>15s} {bh_v:>12s}")
    else:
        print(f"  {'Metric':<22s} {'SMA5 (h=%d)' % args.horizon:>15s} {'Buy&Hold':>12s}")
        print("  " + "-" * 52)
        bh = sma5_results["buy_and_hold_return"]
        rows = [
            ("Total Return", f"{sma5_results['total_return']:+.2f}%", f"{bh:+.2f}%"),
            ("Max Drawdown", f"{sma5_results['max_drawdown']:.2f}%", "N/A"),
            ("Sharpe Ratio", f"{sma5_results['sharpe_ratio']:.3f}", "N/A"),
            ("Win Rate",     f"{sma5_results['win_rate']:.1f}%", "N/A"),
            ("Profit Factor",f"{sma5_results['profit_factor']:.2f}", "N/A"),
            ("Num Trades",   f"{sma5_results['num_trades']}", "1"),
            ("Final Value",  f"{sma5_results['final_value']:,.0f}", "N/A"),
        ]
        for label, s, bh_v in rows:
            print(f"  {label:<22s} {s:>15s} {bh_v:>12s}")

    trades = sma5_results["trades"]
    if trades:
        print(f"\nRecent Trades (last 10):")
        for t in trades[-10:]:
            print(f"  {t['date']}  {t['action']:4s}  {t['shares']:6d} @ "
                  f"{t['price']:.2f}  ({t['signal']})")

    # Signal distribution
    test_df = sma5_results["test_df"]
    dist = test_df["signal"].value_counts()
    print(f"\nSignal distribution:")
    for sig in ["strong_buy", "mild_buy", "hold", "mild_sell", "strong_sell"]:
        count = dist.get(sig, 0)
        print(f"  {sig:<14s} {count:4d}  ({count/len(test_df)*100:.1f}%)")


if __name__ == "__main__":
    main()
