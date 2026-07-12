"""
Backtest for The Hunter — BUY_CALL momentum breakout role.

Scans historical daily OHLCV for VCP signals, then measures stock price
outcome at forward horizons (5/10/20 bars).  Option P&L is simulated using
Black-Scholes mark-to-market with Tito's scaling exits:
  - Stop: exit option when underlying stock drops so that premium loses 50%
  - Target 1: close 25% of position at premium × 1.5
  - Target 2: close 50% of remaining at premium × 2.0
  - Hold the rest to expiry (35 DTE → ~17 trading days)

Two result sets are reported:
  1. All signals
  2. Signals that also pass the sector-strength filter (sector ETF above EMA50)

Usage:
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/backtest_hunter.py
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/backtest_hunter.py --symbols NVDA AMD AAPL
"""
import os
import sys
import argparse
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import contextlib, io, yfinance as yf

from trend_scanner import compute_indicators
from vault76.armory.hunter import Hunter, SECTOR_MAP, CALL_DTE
from vault76.overseer import Overseer
from options_pricer import (
    black_scholes_call, historical_vol, RISK_FREE_RATE,
)

# ── Parameters ────────────────────────────────────────────────────────────────
DEFAULT_SYMBOLS = list(dict.fromkeys(SECTOR_MAP.keys()))   # unique, ordered
START_DATE      = "2019-01-01"
SPY_PATH        = os.path.join(os.path.dirname(__file__), "..", "data", "spy_history.csv")
FORWARD_DAYS    = [5, 10, 20]
MIN_HISTORY     = 150    # bars before scanning starts

hunter = Hunter()


# ── Data loading ──────────────────────────────────────────────────────────────

def _download(symbol: str, start: str = START_DATE) -> pd.DataFrame:
    """Download daily OHLCV from yfinance, return in standard format."""
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        raw = yf.download(symbol, start=start, interval="1d",
                          auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    raw = raw.reset_index()
    # flatten MultiIndex columns and lowercase everything
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    # 'date' or 'index' after reset_index — map to 'datetime'
    raw = raw.rename(columns={"date": "datetime", "index": "datetime"})
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    return raw[["datetime", "open", "high", "low", "close", "volume"]].dropna()


def _load_spy() -> pd.DataFrame:
    """SPY for regime detection — cached to disk."""
    if os.path.exists(SPY_PATH):
        df = pd.read_csv(SPY_PATH, parse_dates=["datetime"])
    else:
        print("  Downloading SPY…")
        df = _download("SPY")
        df.to_csv(SPY_PATH, index=False)
    return compute_indicators(df).dropna().reset_index(drop=True)


def _load_sector(etf: str) -> pd.DataFrame:
    """Load sector ETF from pre-downloaded history or fetch."""
    path = os.path.join(os.path.dirname(__file__), "..", "data",
                        f"{etf.lower()}_history.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["datetime"])
    else:
        print(f"  Downloading {etf}…")
        df = _download(etf)
        df.to_csv(path, index=False)
    return compute_indicators(df).dropna().reset_index(drop=True)


# ── Regime helper ─────────────────────────────────────────────────────────────

def _spy_regime(spy_df: pd.DataFrame, as_of_idx: int) -> str:
    """Determine Overseer regime from SPY as of a given row index."""
    row = spy_df.iloc[as_of_idx]
    if row["close"] >= row["ema50"]:
        return Overseer.RECLAMATION
    return Overseer.WASTELAND


def _sector_above_ema50(sector_df: pd.DataFrame, date: pd.Timestamp) -> bool:
    """True if the sector ETF's close >= EMA50 on the given date (or closest prior)."""
    mask = sector_df["datetime"] <= date
    if not mask.any():
        return True   # no data — don't filter
    row = sector_df[mask].iloc[-1]
    return bool(row["close"] >= row["ema50"])


# ── Option P&L simulation ─────────────────────────────────────────────────────

def _simulate_option_pnl(signal: dict, df: pd.DataFrame,
                          entry_idx: int) -> dict:
    """
    Simulate the option trade from entry_idx onward.
    State machine:
      - Check each bar: stop if premium < stop_premium; T1/T2 exits on underlying
      - Hold remaining to min(expiry, end-of-data)

    Returns: {pnl_pct, exit_reason, days_held}
    pnl_pct is the % return on the initial premium paid.
    """
    entry_close   = float(df.iloc[entry_idx]["close"])
    entry_premium = signal["premium"]
    stop_prem     = signal["stop_premium"]
    exit_25pct    = signal["exit_25pct"]     # sell 25% here (premium × 1.5)
    exit_50pct    = signal["exit_50pct"]     # sell 50% of remaining here (premium × 2.0)
    strike        = signal["strike"]
    hv_entry      = historical_vol(df["close"].iloc[max(0, entry_idx-22):entry_idx+1])

    size_remaining = 1.0   # fraction of position still open
    cash_out       = 0.0   # collected proceeds (as fraction of initial premium)
    t1_triggered   = False
    t2_triggered   = False

    max_hold = min(entry_idx + CALL_DTE + 5, len(df) - 1)

    for i in range(entry_idx + 1, max_hold + 1):
        row   = df.iloc[i]
        close = float(row["close"])
        days_left = max_hold - i
        T     = max(days_left, 1) / 365
        hv_now = historical_vol(df["close"].iloc[max(0, i-22):i+1]) or hv_entry
        cur_prem = black_scholes_call(close, strike, T, RISK_FREE_RATE, hv_now)

        # Stop: option lost 50% of entry premium
        if cur_prem <= stop_prem and size_remaining > 0:
            cash_out += cur_prem * size_remaining
            size_remaining = 0
            pnl_pct = (cash_out / entry_premium - 1) * 100
            return {"pnl_pct": round(pnl_pct, 1),
                    "exit_reason": "stop", "days_held": i - entry_idx}

        # T1: underlying up enough that premium × 1.5 is reachable — use current mark
        if not t1_triggered and cur_prem >= exit_25pct:
            sold = 0.25 * size_remaining
            cash_out += exit_25pct * sold
            size_remaining -= sold
            t1_triggered = True

        # T2: sell 50% of remaining at premium × 2.0
        if t1_triggered and not t2_triggered and cur_prem >= exit_50pct:
            sold = 0.50 * size_remaining
            cash_out += exit_50pct * sold
            size_remaining -= sold
            t2_triggered = True

        if size_remaining <= 0:
            break

    # Expiry: mark remaining to current B-S
    if size_remaining > 0:
        row   = df.iloc[min(max_hold, len(df)-1)]
        close = float(row["close"])
        hv_exp = historical_vol(df["close"].iloc[-22:]) or hv_entry
        final_prem = black_scholes_call(close, strike, 1/365, RISK_FREE_RATE, hv_exp)
        cash_out += final_prem * size_remaining

    pnl_pct = (cash_out / entry_premium - 1) * 100
    reason  = ("partial_exit" if (t1_triggered or t2_triggered) else "expiry")
    return {"pnl_pct": round(pnl_pct, 1),
            "exit_reason": reason, "days_held": max_hold - entry_idx}


# ── Per-symbol scan ───────────────────────────────────────────────────────────

def _scan_symbol(symbol: str, spy_df: pd.DataFrame,
                 sector_dfs: dict) -> list[dict]:
    """Scan all bars for VCP signals; simulate each trade. Returns list of rows."""
    print(f"  {symbol}…", end=" ", flush=True)
    raw = _download(symbol)
    if raw.empty or len(raw) < MIN_HISTORY:
        print(f"insufficient data ({len(raw)} bars)")
        return []

    df = compute_indicators(raw).dropna().reset_index(drop=True)
    if len(df) < MIN_HISTORY:
        print(f"insufficient after indicators ({len(df)} bars)")
        return []

    # Build a date→index map for SPY regime lookup
    spy_dates = pd.to_datetime(spy_df["datetime"]).dt.date
    sector_etf = SECTOR_MAP.get(symbol)

    rows = []
    signal_count = 0

    for i in range(MIN_HISTORY, len(df) - 1):
        snap = df.iloc[:i+1].copy()
        date = pd.to_datetime(df.iloc[i]["datetime"])

        # Regime: find closest SPY bar on or before this date
        spy_mask = spy_dates <= date.date()
        if not spy_mask.any():
            continue
        spy_idx   = spy_df[spy_mask].index[-1]
        regime    = _spy_regime(spy_df, spy_idx)

        if regime != Overseer.RECLAMATION:
            continue

        sig = hunter.scan(symbol, snap, regime=regime)
        if sig["signal"] != "BUY_CALL":
            continue

        # Sector filter
        sector_ok = True
        if sector_etf and sector_etf in sector_dfs:
            sector_ok = _sector_above_ema50(sector_dfs[sector_etf], date)

        # Forward returns (stock price)
        fwd = {}
        for d in FORWARD_DAYS:
            if i + d < len(df):
                future_close = float(df.iloc[i + d]["close"])
                fwd[f"fwd_{d}d_pct"] = round(
                    (future_close / float(df.iloc[i]["close"]) - 1) * 100, 2
                )
            else:
                fwd[f"fwd_{d}d_pct"] = np.nan

        # Option P&L simulation
        opt = _simulate_option_pnl(sig, df, i)

        rows.append({
            "symbol":      symbol,
            "date":        date.date().isoformat(),
            "close":       sig["close"],
            "strike":      sig["strike"],
            "premium":     sig["premium"],
            "premium_pct": sig["premium_pct"],
            "adx":         sig["adx"],
            "rsi":         sig["rsi"],
            "hv":          sig["hv"],
            "vcp_tight_pct": sig.get("vcp_tight_pct"),
            "breakout_vol":  sig.get("breakout_vol"),
            "sector_etf":  sector_etf,
            "sector_ok":   sector_ok,
            **fwd,
            "opt_pnl_pct":    opt["pnl_pct"],
            "opt_exit":       opt["exit_reason"],
            "opt_days_held":  opt["days_held"],
        })
        signal_count += 1

    print(f"{signal_count} signals")
    return rows


# ── Summary stats ─────────────────────────────────────────────────────────────

def _summarise(df: pd.DataFrame, label: str):
    if df.empty:
        print(f"\n{label}: no signals")
        return

    n = len(df)
    print(f"\n{'='*60}")
    print(f"{label}  (n={n})")
    print(f"{'='*60}")

    # Stock forward returns
    for d in FORWARD_DAYS:
        col = f"fwd_{d}d_pct"
        vals = df[col].dropna()
        if vals.empty:
            continue
        wins  = (vals > 0).sum()
        print(f"  Stock +{d:2d}d:  win={wins/len(vals)*100:.0f}%  "
              f"avg={vals.mean():+.1f}%  median={vals.median():+.1f}%")

    # Option P&L
    pnl = df["opt_pnl_pct"].dropna()
    if not pnl.empty:
        wins = (pnl > 0).sum()
        print(f"\n  Option P&L:  win={wins/len(pnl)*100:.0f}%  "
              f"avg={pnl.mean():+.1f}%  median={pnl.median():+.1f}%")
        by_exit = df.groupby("opt_exit")["opt_pnl_pct"].agg(["count","mean"])
        for reason, row in by_exit.iterrows():
            print(f"    exit={reason:<14} n={int(row['count'])}  avg={row['mean']:+.1f}%")

    # Per-symbol breakdown
    print(f"\n  By symbol:")
    for sym, grp in df.groupby("symbol"):
        w = (grp["opt_pnl_pct"] > 0).sum()
        avg = grp["opt_pnl_pct"].mean()
        print(f"    {sym:<8} n={len(grp):3d}  win={w/len(grp)*100:.0f}%  avg={avg:+.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--start",   default=START_DATE)
    args = parser.parse_args()

    print(f"Hunter backtest  symbols={args.symbols}  start={args.start}")
    print("Loading SPY regime…")
    spy_df = _load_spy()

    print("Loading sector ETFs…")
    sector_dfs = {}
    for etf in {"XLK", "XLY", "XLC", "XLP", "SPY"}:
        sector_dfs[etf] = _load_sector(etf)

    all_rows: list[dict] = []
    print("\nScanning symbols…")
    for sym in args.symbols:
        rows = _scan_symbol(sym, spy_df, sector_dfs)
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo signals found.")
        return

    df_all    = pd.DataFrame(all_rows)
    df_sector = df_all[df_all["sector_ok"]].copy()

    out_path = os.path.join(os.path.dirname(__file__), "..",
                            "data", "backtest_hunter_results.csv")
    df_all.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")

    _summarise(df_all,    "ALL signals (no sector filter)")
    _summarise(df_sector, "SECTOR-FILTERED signals (sector ETF above EMA50)")

    print(f"\nTotal signals: {len(df_all)}  sector-filtered: {len(df_sector)}")


if __name__ == "__main__":
    main()
