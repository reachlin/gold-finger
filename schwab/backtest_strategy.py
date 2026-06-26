"""
Walk-forward backtest for the pullback-in-trend strategy.

Rules:
  - At each bar, run trend_scanner on all data up to that bar (no lookahead)
  - On BUY signal: enter at next bar's open
  - Exit when: high >= target (+20%), low <= stop (-8%), trend ends, or max_hold days
  - One trade at a time (no pyramiding)

Usage:
    python schwab/backtest_strategy.py                  # uses saved nvda_history.csv
    python schwab/backtest_strategy.py --fetch          # re-fetch from Schwab first
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trend_scanner import compute_indicators, scan_symbol, detect_trend

TARGET_ATR  = 5.0       # target = entry + 5 * ATR
STOP_ATR    = 2.0       # stop   = entry - 2 * ATR
MAX_HOLD    = 30        # max calendar days to hold a position
MIN_HISTORY = 60        # minimum bars needed before generating signals
BUDGET      = 2400      # USD per trade


def simulate_trade(future_df: pd.DataFrame, entry: float,
                   target: float, stop: float, max_hold: int = MAX_HOLD) -> dict:
    """
    Simulate a trade given future price bars after entry.
    Checks high for target and low for stop on each bar.
    """
    for i, (_, row) in enumerate(future_df.iterrows()):
        if i >= max_hold:
            return {"exit_reason": "timeout", "pnl_pct": round((row["close"] - entry) / entry * 100, 2), "hold_days": i}
        if row["high"] >= target:
            return {"exit_reason": "target",  "pnl_pct": round((target - entry) / entry * 100, 2), "hold_days": i + 1}
        if row["low"] <= stop:
            return {"exit_reason": "stop",    "pnl_pct": round((stop - entry) / entry * 100, 2),   "hold_days": i + 1}
        # Trend ended — exit at close
        hist = future_df.iloc[:i + 1].copy()
        if len(hist) >= 10:
            ema20 = hist["close"].ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = hist["close"].ewm(span=50, adjust=False).mean().iloc[-1]
            if ema20 < ema50:
                return {"exit_reason": "trend_end", "pnl_pct": round((row["close"] - entry) / entry * 100, 2), "hold_days": i + 1}

    last_close = future_df["close"].iloc[-1]
    return {"exit_reason": "timeout", "pnl_pct": round((last_close - entry) / entry * 100, 2), "hold_days": len(future_df)}


def walk_forward(df: pd.DataFrame, min_history: int = MIN_HISTORY) -> list:
    """
    Walk forward through df day by day.
    At each bar i, use df[0:i] to scan for signal.
    If BUY, enter at bar i+1 open and simulate the trade.
    """
    trades = []
    in_trade = False
    i = min_history

    while i < len(df) - 1:
        history = df.iloc[:i].copy()
        history_ind = compute_indicators(history).dropna().reset_index(drop=True)

        if len(history_ind) < min_history:
            i += 1
            continue

        result = scan_symbol("", history_ind)

        if not in_trade and result["signal"] == "BUY":
            entry_bar  = df.iloc[i + 1]
            entry      = round(entry_bar["open"], 2)
            atr        = round(history_ind["atr"].iloc[-1], 2)
            target     = round(entry + TARGET_ATR * atr, 2)
            stop       = round(entry - STOP_ATR * atr, 2)
            future     = df.iloc[i + 1:].reset_index(drop=True)
            signal_date = df.iloc[i]["datetime"]
            entry_date  = entry_bar["datetime"]

            trade_result = simulate_trade(future, entry, target, stop)
            hold = trade_result["hold_days"]

            trade = {
                "signal_date": signal_date,
                "entry_date":  entry_date,
                "entry":       entry,
                "target":      target,
                "stop":        stop,
                "shares":      max(1, int(BUDGET / entry)),
                **trade_result,
            }
            trade["pnl_$"] = round(trade["shares"] * entry * trade["pnl_pct"] / 100, 2)
            trades.append(trade)

            # Skip ahead past this trade
            i += hold + 1
        else:
            i += 1

    return trades


def summarize(trades: list) -> dict:
    if not trades:
        return {"total_trades": 0}

    pnls    = [t["pnl_pct"] for t in trades]
    wins    = [t for t in trades if t["pnl_pct"] > 0]
    pnl_usd = sum(t.get("pnl_$", 0) for t in trades)

    by_reason = {}
    for t in trades:
        by_reason[t["exit_reason"]] = by_reason.get(t["exit_reason"], 0) + 1

    return {
        "total_trades":  len(trades),
        "win_rate":      round(len(wins) / len(trades) * 100, 1),
        "avg_pnl_pct":   round(np.mean(pnls), 2),
        "total_pnl_$":   round(pnl_usd, 2),
        "best_trade_%":  round(max(pnls), 2),
        "worst_trade_%": round(min(pnls), 2),
        "avg_hold_days": round(np.mean([t["hold_days"] for t in trades]), 1),
        "exit_reasons":  by_reason,
    }


def print_report(symbol: str, df: pd.DataFrame, trades: list, stats: dict):
    bh_shares  = int(BUDGET / df["close"].iloc[0])
    bh_return  = round((df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100, 1)
    bh_pnl     = round(bh_shares * (df["close"].iloc[-1] - df["close"].iloc[0]), 2)
    date_range = f"{df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}"

    print(f"\n{'='*60}")
    print(f"  Pullback-in-Trend Backtest: {symbol}")
    print(f"  Period: {date_range}  ({len(df)} trading days)")
    print(f"{'='*60}")

    if not trades:
        print("  No trades generated.")
        return

    print(f"\n  {'Date':<12} {'Entry':>7} {'Exit':>7} {'PnL%':>7} {'PnL$':>8}  Reason")
    print(f"  {'-'*60}")
    for t in trades:
        exit_price = round(t["entry"] * (1 + t["pnl_pct"] / 100), 2)
        print(f"  {str(t['entry_date'].date()):<12} ${t['entry']:>6.2f} ${exit_price:>6.2f} "
              f"{t['pnl_pct']:>+6.1f}% ${t['pnl_$']:>7.2f}  {t['exit_reason']}")

    print(f"\n  --- Summary ---")
    print(f"  Trades:        {stats['total_trades']}")
    print(f"  Win rate:      {stats['win_rate']}%")
    print(f"  Avg P&L:       {stats['avg_pnl_pct']:+.1f}% per trade")
    print(f"  Total P&L:    ${stats['total_pnl_$']:+,.2f}")
    print(f"  Best trade:    {stats['best_trade_%']:+.1f}%")
    print(f"  Worst trade:   {stats['worst_trade_%']:+.1f}%")
    print(f"  Avg hold:      {stats['avg_hold_days']} days")
    print(f"  Exit reasons:  {stats['exit_reasons']}")
    print(f"\n  --- vs Buy & Hold ---")
    print(f"  B&H return:   {bh_return:+.1f}%  (${bh_pnl:+,.2f}) on {bh_shares} shares")
    print(f"  Strategy:     ${stats['total_pnl_$']:+,.2f} total P&L")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="Re-fetch data from Schwab")
    parser.add_argument("--symbol", default="NVDA")
    args = parser.parse_args()

    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "nvda_history.csv")

    if args.fetch or not os.path.exists(csv_path):
        from dotenv import load_dotenv
        import schwab as schwab_lib
        from nvda_trader import get_client, fetch_nvda_history, candles_to_df
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
        print("Fetching from Schwab...")
        client  = get_client()
        candles = fetch_nvda_history(client, days=730)
        df      = candles_to_df(candles)
        df.to_csv(csv_path, index=False)
        print(f"  Saved {len(df)} bars to {csv_path}")
    else:
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
        print(f"Loaded {len(df)} bars from {csv_path}")

    print("Running walk-forward backtest...")
    trades = walk_forward(df)
    stats  = summarize(trades)
    print_report(args.symbol, df, trades, stats)


if __name__ == "__main__":
    main()
