"""
Combined bull+bear walk-forward backtest.

Strategy:
  BEAR leg — pullback detected (trend intact, RSI dipped) but entry not confirmed yet:
    → Buy ATM put (21 DTE). Profit from the pullback drop.
    → Exit: stock makes new high (+3%), put doubles, or 10 days.

  BULL leg — full entry signal (trend + pullback + bounce confirmed):
    → Close any open put first (lock in put profit), then buy stock.
    → Exit: +5×ATR target, -2×ATR stop, trend ends, or 30 days.

  State machine per bar:
    FLAT      → on pullback signal: enter PUT
    IN_PUT    → on new_high/target/timeout: close put → FLAT
              → on entry signal: close put → enter STOCK
    IN_STOCK  → on target/stop/trend_end/timeout: close stock → FLAT

Usage:
    python schwab/backtest_combo.py
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trend_scanner import compute_indicators, detect_trend, detect_pullback, detect_entry, detect_market_regime
from options_pricer import (black_scholes_put, historical_vol, atm_strike,
                            simulate_put_trade, RISK_FREE_RATE,
                            PUT_DTE, PUT_MAX_HOLD, PUT_PROFIT_MULT, NEW_HIGH_PCT)
from backtest_strategy import summarize

MIN_HISTORY  = 60
BUDGET_STOCK = 2_400   # USD per stock trade
BUDGET_PUT   = 300     # USD per put trade (cap on premium)
TARGET_ATR   = 5.0
STOP_ATR     = 2.0
MAX_HOLD_STK = 30


def _put_value(S, K, days_remaining, sigma, r=RISK_FREE_RATE):
    return black_scholes_put(S, K, T=max(days_remaining, 0) / 365, r=r, sigma=sigma)


def walk_forward_combo(df: pd.DataFrame, spy_df: pd.DataFrame = None) -> dict:
    """
    Walk-forward through df. Returns {"put_trades": [...], "stock_trades": [...]}.
    Each bar we only use history[:i] to avoid lookahead.
    If spy_df provided, applies market-regime gate before entering any position.
    """
    put_trades   = []
    stock_trades = []

    state       = "FLAT"   # FLAT | IN_PUT | IN_STOCK
    put_pos     = {}
    stock_pos   = {}

    i = MIN_HISTORY
    while i < len(df) - 1:
        history     = df.iloc[:i].copy()
        history_ind = compute_indicators(history).dropna().reset_index(drop=True)

        if len(history_ind) < MIN_HISTORY:
            i += 1
            continue

        row  = df.iloc[i]       # today's bar (signal bar)
        date = row["datetime"]

        in_trend    = detect_trend(history_ind)
        in_pullback = detect_pullback(history_ind)
        in_entry    = detect_entry(history_ind)

        regime_ok = True
        if spy_df is not None:
            spy_hist = spy_df[spy_df["datetime"] <= date].copy()
            if len(spy_hist) >= MIN_HISTORY:
                spy_ind   = compute_indicators(spy_hist).dropna().reset_index(drop=True)
                regime_ok = detect_market_regime(spy_ind)
        sigma       = historical_vol(history_ind["close"])
        if sigma < 0.01:
            sigma = 0.50   # fallback for low-vol history

        # ── State: IN_PUT ────────────────────────────────────────────────
        if state == "IN_PUT":
            days_held = i - put_pos["entry_bar"]
            cur_put   = _put_value(row["close"], put_pos["K"],
                                   PUT_DTE - days_held, sigma)
            entry_put = put_pos["entry_put_price"]

            close_reason = None
            exit_put     = cur_put

            if row["high"] >= put_pos["entry_S"] * (1 + NEW_HIGH_PCT):
                close_reason = "new_high"
            elif cur_put >= entry_put * PUT_PROFIT_MULT:
                close_reason = "target"
            elif days_held >= PUT_MAX_HOLD:
                close_reason = "timeout"
            elif in_entry and in_trend:
                close_reason = "bull_signal"   # switch to stock

            if close_reason:
                pnl = (exit_put - entry_put) * 100
                put_trades.append({
                    "type":            "PUT",
                    "entry_date":      put_pos["entry_date"],
                    "exit_date":       date,
                    "entry_S":         put_pos["entry_S"],
                    "K":               put_pos["K"],
                    "entry_put_price": round(entry_put, 4),
                    "exit_put_price":  round(exit_put, 4),
                    "hold_days":       days_held,
                    "exit_reason":     close_reason,
                    "pnl_$":           round(pnl, 2),
                    "pnl_pct":         round((exit_put - entry_put) / entry_put * 100
                                             if entry_put > 0 else 0, 2),
                })
                state   = "FLAT"
                put_pos = {}

                # If closed because entry signal → immediately enter stock
                if close_reason == "bull_signal":
                    next_bar = df.iloc[i + 1]
                    entry    = round(next_bar["open"], 2)
                    atr      = round(history_ind["atr"].iloc[-1], 2)
                    target   = round(entry + TARGET_ATR * atr, 2)
                    stop     = round(entry - STOP_ATR * atr, 2)
                    shares   = max(1, int(BUDGET_STOCK / entry))
                    stock_pos = {
                        "entry_bar":  i + 1,
                        "entry_date": next_bar["datetime"],
                        "entry":      entry,
                        "target":     target,
                        "stop":       stop,
                        "shares":     shares,
                    }
                    state = "IN_STOCK"
                    i += 2
                    continue

            i += 1
            continue

        # ── State: IN_STOCK ──────────────────────────────────────────────
        if state == "IN_STOCK":
            days_held    = i - stock_pos["entry_bar"]
            close_reason = None
            exit_price   = row["close"]

            if row["high"] >= stock_pos["target"]:
                close_reason = "target"
                exit_price   = stock_pos["target"]
            elif row["low"] <= stock_pos["stop"]:
                close_reason = "stop"
                exit_price   = stock_pos["stop"]
            elif not detect_trend(history_ind):
                close_reason = "trend_end"
            elif days_held >= MAX_HOLD_STK:
                close_reason = "timeout"

            if close_reason:
                pnl_pct = (exit_price - stock_pos["entry"]) / stock_pos["entry"] * 100
                stock_trades.append({
                    "type":       "STOCK",
                    "entry_date": stock_pos["entry_date"],
                    "exit_date":  date,
                    "entry":      stock_pos["entry"],
                    "exit":       round(exit_price, 2),
                    "target":     stock_pos["target"],
                    "stop":       stock_pos["stop"],
                    "shares":     stock_pos["shares"],
                    "hold_days":  days_held,
                    "exit_reason": close_reason,
                    "pnl_pct":    round(pnl_pct, 2),
                    "pnl_$":      round(stock_pos["shares"] * stock_pos["entry"]
                                        * pnl_pct / 100, 2),
                })
                state     = "FLAT"
                stock_pos = {}
                i += 1
            else:
                i += 1
            continue

        # ── State: FLAT ──────────────────────────────────────────────────
        if not regime_ok:
            i += 1
            continue

        if in_trend and in_pullback and not in_entry:
            # Pullback in progress — buy put
            S      = row["close"]
            K      = atm_strike(S)
            entry_put = _put_value(S, K, PUT_DTE, sigma)

            if entry_put > 0:
                contracts = max(1, int(BUDGET_PUT / (entry_put * 100)))
                put_pos = {
                    "entry_bar":       i,
                    "entry_date":      date,
                    "entry_S":         S,
                    "K":               K,
                    "sigma":           sigma,
                    "entry_put_price": entry_put,
                    "contracts":       contracts,
                }
                state = "IN_PUT"

        elif in_trend and in_entry:
            # Direct entry signal without prior put — buy stock
            if i + 1 >= len(df):
                break
            next_bar = df.iloc[i + 1]
            entry    = round(next_bar["open"], 2)
            atr      = round(history_ind["atr"].iloc[-1], 2)
            target   = round(entry + TARGET_ATR * atr, 2)
            stop     = round(entry - STOP_ATR * atr, 2)
            shares   = max(1, int(BUDGET_STOCK / entry))
            stock_pos = {
                "entry_bar":  i + 1,
                "entry_date": next_bar["datetime"],
                "entry":      entry,
                "target":     target,
                "stop":       stop,
                "shares":     shares,
            }
            state = "IN_STOCK"
            i += 2
            continue

        i += 1

    return {"put_trades": put_trades, "stock_trades": stock_trades}


def print_combo_report(symbol: str, df: pd.DataFrame, results: dict):
    puts   = results["put_trades"]
    stocks = results["stock_trades"]
    all_trades = puts + stocks

    bh_shares = int(BUDGET_STOCK / df["close"].iloc[0])
    bh_return = round((df["close"].iloc[-1] - df["close"].iloc[0])
                      / df["close"].iloc[0] * 100, 1)
    bh_pnl    = round(bh_shares * (df["close"].iloc[-1] - df["close"].iloc[0]), 2)
    date_range = f"{df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}"

    print(f"\n{'='*68}")
    print(f"  Combined Bull+Bear Backtest: {symbol}")
    print(f"  Period: {date_range}  ({len(df)} bars)")
    print(f"{'='*68}")

    if puts:
        put_pnl  = sum(t["pnl_$"] for t in puts)
        put_wins = sum(1 for t in puts if t["pnl_$"] > 0)
        print(f"\n  PUT trades ({len(puts)}, win rate {put_wins/len(puts)*100:.0f}%):")
        for t in puts:
            icon = "+" if t["pnl_$"] > 0 else "-"
            print(f"    {str(t['entry_date'].date()):<12}"
                  f"  S=${t['entry_S']:.2f} K=${t['K']:.2f}"
                  f"  put ${t['entry_put_price']:.2f}→${t['exit_put_price']:.2f}"
                  f"  {icon}${abs(t['pnl_$']):.2f}  [{t['exit_reason']}]")
        print(f"    PUT subtotal: ${put_pnl:+.2f}")

    if stocks:
        stk_pnl  = sum(t["pnl_$"] for t in stocks)
        stk_wins = sum(1 for t in stocks if t["pnl_$"] > 0)
        print(f"\n  STOCK trades ({len(stocks)}, win rate {stk_wins/len(stocks)*100:.0f}%):")
        for t in stocks:
            icon = "+" if t["pnl_$"] > 0 else "-"
            print(f"    {str(t['entry_date'].date()):<12}"
                  f"  ${t['entry']:.2f}→${t['exit']:.2f}"
                  f"  {t['pnl_pct']:+.1f}%  {icon}${abs(t['pnl_$']):.2f}"
                  f"  [{t['exit_reason']}]")
        print(f"    STOCK subtotal: ${stk_pnl:+.2f}")

    total_pnl = sum(t["pnl_$"] for t in all_trades)
    wins      = sum(1 for t in all_trades if t["pnl_$"] > 0)
    win_rate  = wins / len(all_trades) * 100 if all_trades else 0

    print(f"\n  --- Combined Summary ---")
    print(f"  Total trades:  {len(all_trades)}  ({len(puts)} puts + {len(stocks)} stocks)")
    print(f"  Win rate:      {win_rate:.0f}%")
    print(f"  Total P&L:    ${total_pnl:+,.2f}")
    print(f"\n  --- vs Buy & Hold ---")
    print(f"  B&H:          ${bh_pnl:+,.2f}  ({bh_return:+.1f}%)")
    print(f"  Strategy:     ${total_pnl:+,.2f}")
    print(f"{'='*68}\n")


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "nvda_history.csv")
    spy_path = os.path.join(os.path.dirname(__file__), "..", "data", "spy_history.csv")
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    spy_df = pd.read_csv(spy_path, parse_dates=["datetime"]) if os.path.exists(spy_path) else None
    if spy_df is not None:
        print(f"Loaded SPY regime data ({len(spy_df)} bars) — regime filter ON")
    print(f"Loaded {len(df)} NVDA bars. Running combined backtest...")
    results = walk_forward_combo(df, spy_df=spy_df)
    print_combo_report("NVDA", df, results)


if __name__ == "__main__":
    main()
