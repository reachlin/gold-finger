"""
Backtest all existing trading bots on 1 year of NVDA data fetched from Schwab.
Compares: KMeans, LGBM, DNN, and buy-and-hold baseline.
"""
import os
import sys
import schwab
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

CLIENT_ID     = os.environ["SCHWAB_CLIENT_ID"]
CLIENT_SECRET = os.environ["SCHWAB_CLIENT_SECRET"]
REDIRECT_URI  = "https://127.0.0.1"
TOKEN_PATH    = os.path.join(os.path.dirname(__file__), "schwab_token.json")

SYMBOL        = "NVDA"
INITIAL_CASH  = 10_000
LOT           = 3          # shares per trade (matches live trader budget ~$600)


def get_client():
    return schwab.auth.client_from_token_file(TOKEN_PATH, CLIENT_ID, CLIENT_SECRET)


def fetch_history(client, days=400):
    end   = datetime.now()
    start = end - timedelta(days=days)
    resp  = client.get_price_history_every_day(SYMBOL, start_datetime=start, end_datetime=end)
    resp.raise_for_status()
    candles = resp.json().get("candles", [])
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    df["volume"] = df["volume"].astype(float)
    return df


# ---------------------------------------------------------------------------
# Generic backtest engine (no fractional shares, LOT shares per trade)
# ---------------------------------------------------------------------------
def run_backtest(df_with_signals, signal_col, label,
                 strong_only=False, trend_filter=False):
    """
    signal_col values: 4=strong_buy, 3=mild_buy, 2=hold, 1=mild_sell, 0=strong_sell.
    strong_only: only trade on 4/0, ignore mild signals.
    trend_filter: only buy when close > EMA50.
    """
    cash   = float(INITIAL_CASH)
    shares = 0
    equity = []
    trades = 0

    # Precompute EMA50 if needed
    ema50 = df_with_signals["close"].ewm(span=50, adjust=False).mean()

    for i, (_, row) in enumerate(df_with_signals.iterrows()):
        price  = row["close"]
        signal = row[signal_col]

        buy_signals  = (4,) if strong_only else (3, 4)
        sell_signals = (0,) if strong_only else (0, 1)

        above_trend = price > ema50.iloc[i] if trend_filter else True

        if signal in buy_signals and shares == 0 and above_trend:
            affordable = int(cash / price)
            qty = min(LOT, affordable)
            if qty > 0:
                shares += qty
                cash   -= qty * price
                trades += 1
        elif signal in sell_signals and shares > 0:
            cash   += shares * price
            shares  = 0
            trades += 1

        equity.append(cash + shares * price)

    final = equity[-1] if equity else INITIAL_CASH
    ret   = (final - INITIAL_CASH) / INITIAL_CASH * 100
    return {
        "strategy":  label,
        "trades":    trades,
        "final_$":   round(final, 2),
        "return_%":  round(ret, 2),
    }, equity


def buy_and_hold(df):
    shares = int(INITIAL_CASH / df["close"].iloc[0])
    leftover = INITIAL_CASH - shares * df["close"].iloc[0]
    equity = (df["close"] * shares + leftover).tolist()
    final  = equity[-1]
    ret    = (final - INITIAL_CASH) / INITIAL_CASH * 100
    return {
        "strategy": "Buy & Hold",
        "trades":   1,
        "final_$":  round(final, 2),
        "return_%": round(ret, 2),
    }, equity


def main():
    from trading_bot   import compute_indicators, FEATURE_COLS, TradingBot, run_backtest as kmeans_bt
    from lgbm_trading_bot import LGBMTradingBot

    client = get_client()
    print(f"Fetching {SYMBOL} history from Schwab...")
    df = fetch_history(client, days=400)
    print(f"  {len(df)} trading days  |  {df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}")
    print(f"  Price range: ${df['close'].min():.2f} – ${df['close'].max():.2f}")
    print()

    df_ind = compute_indicators(df).dropna(subset=FEATURE_COLS).reset_index(drop=True)

    results = []

    # ── Buy & Hold ──────────────────────────────────────────────────────────
    bh_stat, _ = buy_and_hold(df_ind)
    results.append(bh_stat)
    print(f"Buy & Hold:  {bh_stat['return_%']:+.1f}%  (${bh_stat['final_$']})")

    # ── KMeans ──────────────────────────────────────────────────────────────
    try:
        bot = TradingBot()
        bot.fit(df_ind)
        df_km = df_ind.copy()
        X = bot.scaler.transform(df_ind[FEATURE_COLS])
        df_km["signal"] = bot.model.predict(X)
        # remap cluster labels to buy/sell via bot's cluster_map
        if hasattr(bot, "cluster_map"):
            df_km["signal"] = df_km["signal"].map(bot.cluster_map).fillna(2).astype(int)
        stat, _ = run_backtest(df_km, "signal", "KMeans")
        results.append(stat)
        print(f"KMeans:      {stat['return_%']:+.1f}%  ({stat['trades']} trades)")
    except Exception as e:
        print(f"KMeans:      ERROR — {e}")

    # ── LGBM variants ───────────────────────────────────────────────────────
    try:
        bot = LGBMTradingBot()
        bot.fit(df_ind)
        X = pd.DataFrame(bot.scaler.transform(df_ind[FEATURE_COLS]), columns=FEATURE_COLS)
        df_lgbm = df_ind.copy()
        df_lgbm["signal"] = bot.model.predict(X)

        for label, strong_only, trend_filter in [
            ("LGBM (all signals)",        False, False),
            ("LGBM (strong only)",        True,  False),
            ("LGBM (trend filter)",       False, True),
            ("LGBM (strong+trend)",       True,  True),
        ]:
            stat, _ = run_backtest(df_lgbm, "signal", label,
                                   strong_only=strong_only, trend_filter=trend_filter)
            results.append(stat)
            print(f"{label:<28} {stat['return_%']:+.1f}%  ({stat['trades']} trades)")
    except Exception as e:
        print(f"LGBM:        ERROR — {e}")


    print()
    print("─" * 52)
    print(f"{'Strategy':<14} {'Trades':>6} {'Final $':>10} {'Return':>8}")
    print("─" * 52)
    for r in sorted(results, key=lambda x: x["return_%"], reverse=True):
        print(f"{r['strategy']:<14} {r['trades']:>6} {r['final_$']:>10,.2f} {r['return_%']:>+7.1f}%")
    print("─" * 52)

    best = max(results, key=lambda x: x["return_%"])
    print(f"\nBest strategy: {best['strategy']}  ({best['return_%']:+.1f}%)")

    # Save data for further analysis
    out = os.path.join(os.path.dirname(__file__), "..", "data", "nvda_history.csv")
    df.to_csv(out, index=False)
    print(f"History saved to {out}")


if __name__ == "__main__":
    main()
