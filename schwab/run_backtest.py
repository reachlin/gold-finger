"""
Unified backtest runner — Scavenger + Chemist across the full watchlist.

Always produces the same table format:
  Symbol | Trades | Scavenger P&L | Chemist P&L | Combined | B&H | Edge

Usage:
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/run_backtest.py
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/run_backtest.py AAPL TSLA
"""
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from trend_scanner import compute_indicators
from backtest_scavenger import (walk_forward_scavenger, WATCHLIST,
                                 DATA_DIR, MIN_HISTORY, DTE_BARS)
from backtest_chemist import walk_forward_chemist

LINE = "─" * 95


def _bnh(df: pd.DataFrame) -> tuple[float, float, float]:
    """Buy-and-hold P&L for 100 shares starting from MIN_HISTORY bar."""
    ind = compute_indicators(df).dropna().reset_index(drop=True)
    if len(ind) <= MIN_HISTORY:
        return 0.0, 0.0, 0.0
    init_price  = float(ind.iloc[MIN_HISTORY]["close"])
    final_price = float(ind.iloc[-1]["close"])
    pnl = (final_price - init_price) * 100
    return init_price, final_price, pnl


def run(symbols: list[str]) -> list[dict]:
    spy_path = os.path.join(DATA_DIR, "spy_history.csv")
    vix_path = os.path.join(DATA_DIR, "vix_history.csv")
    spy_df = pd.read_csv(spy_path, parse_dates=["datetime"]) if os.path.exists(spy_path) else None
    vix_df = pd.read_csv(vix_path, parse_dates=["datetime"]) if os.path.exists(vix_path) else None

    rows = []
    for symbol in symbols:
        path = os.path.join(DATA_DIR, f"{symbol.lower()}_history.csv")
        if not os.path.exists(path):
            print(f"  [{symbol}] no data — skipping")
            continue
        df = pd.read_csv(path, parse_dates=["datetime"])
        if len(df) < MIN_HISTORY + DTE_BARS + 10:
            print(f"  [{symbol}] too little data — skipping")
            continue

        scav_events = walk_forward_scavenger(df, symbol, spy_df=spy_df, vix_df=vix_df)
        chem_events = walk_forward_chemist(df, symbol, spy_df=spy_df, vix_df=vix_df)

        scav_pnl  = sum(e["pnl"] for e in scav_events)
        chem_pnl  = sum(e["pnl"] for e in chem_events)
        combined  = scav_pnl + chem_pnl
        n_trades  = len(scav_events) + len(chem_events)

        _, _, bnh_pnl = _bnh(df)
        edge = combined - bnh_pnl

        rows.append({
            "symbol":   symbol,
            "trades":   n_trades,
            "scav_pnl": scav_pnl,
            "chem_pnl": chem_pnl,
            "combined": combined,
            "bnh_pnl":  bnh_pnl,
            "edge":     edge,
        })

    return sorted(rows, key=lambda r: r["combined"], reverse=True)


def print_report(rows: list[dict]) -> None:
    if not rows:
        print("No results.")
        return

    print(f"\n{'='*95}")
    print(f"  VAULT 76 BACKTEST REPORT  —  4yr (Jun 2022 → Jun 2026)  —  Scavenger + Chemist")
    print(f"{'='*95}")
    hdr = (f"  {'Symbol':<6} {'Trades':>6}   {'Scavenger':>12}   {'Chemist':>10}"
           f"   {'Combined':>12}   {'B&H':>12}   {'Edge':>10}")
    print(hdr)
    print(f"  {LINE}")

    for r in rows:
        edge_str = f"${r['edge']:>+10,.0f}"
        print(
            f"  {r['symbol']:<6} {r['trades']:>6}   "
            f"${r['scav_pnl']:>+11,.0f}   ${r['chem_pnl']:>+9,.0f}"
            f"   ${r['combined']:>+11,.0f}   ${r['bnh_pnl']:>+11,.0f}"
            f"   {edge_str}"
        )

    print(f"  {LINE}")
    tot_scav = sum(r["scav_pnl"] for r in rows)
    tot_chem = sum(r["chem_pnl"] for r in rows)
    tot_comb = sum(r["combined"] for r in rows)
    tot_bnh  = sum(r["bnh_pnl"]  for r in rows)
    tot_edge = tot_comb - tot_bnh
    print(
        f"  {'TOTAL':<6} {'':>6}   "
        f"${tot_scav:>+11,.0f}   ${tot_chem:>+9,.0f}"
        f"   ${tot_comb:>+11,.0f}   ${tot_bnh:>+11,.0f}"
        f"   ${tot_edge:>+10,.0f}"
    )
    print(f"{'='*95}\n")

    winners = [r for r in rows if r["edge"] > 0]
    losers  = [r for r in rows if r["edge"] <= 0]
    print(f"  Wheel beats B&H: {len(winners)}/{len(rows)} symbols")
    if winners:
        print(f"  Winners: {', '.join(r['symbol'] for r in winners)}")
    if losers:
        print(f"  Laggards: {', '.join(r['symbol'] for r in losers)}")
    print()


def main():
    symbols = sys.argv[1:] if len(sys.argv) > 1 else WATCHLIST
    rows = run(symbols)
    print_report(rows)


if __name__ == "__main__":
    main()
