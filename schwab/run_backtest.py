"""
Unified backtest runner — Vault 76 role system.

Usage:
  # Overseer mode (default) — all roles, Overseer gates entries by regime
  python schwab/run_backtest.py

  # Single-role mode
  python schwab/run_backtest.py --role scavenger
  python schwab/run_backtest.py --role raider
  python schwab/run_backtest.py --role chemist

  # Specific symbols (any mode)
  python schwab/run_backtest.py AAPL TSLA NVDA
  python schwab/run_backtest.py --role raider AAPL TSLA

In Overseer mode each role runs its own state machine in parallel on the same
bar stream. The Overseer classifies regime at each bar; each role's scan()
checks should_deploy(regime) internally — so they gate themselves correctly:
  RECLAMATION → Raider + Scavenger active, Chemist idle
  WASTELAND   → Scavenger + Raider active, Chemist idle
  NUKED_ZONE  → Chemist active, Raider + Scavenger idle

Capital assumption: 100 shares (1 contract) per trade, no compounding.
"""
import os, sys, argparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from trend_scanner import compute_indicators
from backtest_scavenger import walk_forward_scavenger, MIN_HISTORY, DTE_BARS, DATA_DIR
from backtest_raider   import walk_forward_raider
from backtest_chemist  import walk_forward_chemist

WATCHLIST = [
    "NVDA", "AMD",  "AAPL", "AMZN", "META", "MSFT", "GOOGL",
    "IBM",  "INTC", "IONQ", "KO",   "MMM",  "XOM",  "PG",  "TSLA",
    "UNH",  "HD",   "ABT",
]

VALID_ROLES = ("scavenger", "raider", "chemist", "overseer")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_df(symbol: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"{symbol.lower()}_history.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["datetime"])
    if len(df) < MIN_HISTORY + DTE_BARS + 10:
        return None
    return df


def _date_range(df: pd.DataFrame) -> tuple[str, str]:
    dates = pd.to_datetime(df["datetime"]).dt.date
    return str(dates.min()), str(dates.max())


def _bnh_pnl(df: pd.DataFrame) -> float:
    ind = compute_indicators(df).dropna().reset_index(drop=True)
    if len(ind) <= MIN_HISTORY:
        return 0.0
    return (float(ind.iloc[-1]["close"]) - float(ind.iloc[MIN_HISTORY]["close"])) * 100


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run(symbols: list[str], role: str,
        spy_df: pd.DataFrame | None,
        vix_df: pd.DataFrame | None) -> tuple[list[dict], str, str]:
    """
    Returns (rows, global_start, global_end).
    Each row: symbol, scav/raid/chem trades + P&L, combined, bnh, edge.
    """
    global_start = global_end = ""
    rows = []

    run_scav = role in ("scavenger", "overseer")
    run_raid = role in ("raider",    "overseer")
    run_chem = role in ("chemist",   "overseer")

    for symbol in symbols:
        df = _load_df(symbol)
        if df is None:
            print(f"  [{symbol}] no data or too short — skipping")
            continue

        start, end = _date_range(df)
        if not global_start or start < global_start:
            global_start = start
        if end > global_end:
            global_end = end

        scav_events = walk_forward_scavenger(df, symbol, spy_df, vix_df) if run_scav else []
        raid_events = walk_forward_raider(   df, symbol, spy_df, vix_df) if run_raid else []
        chem_events = walk_forward_chemist(  df, symbol, spy_df, vix_df) if run_chem else []

        scav_pnl = sum(e["pnl"] for e in scav_events)
        raid_pnl = sum(e["pnl"] for e in raid_events)
        chem_pnl = sum(e["pnl"] for e in chem_events)
        combined  = scav_pnl + raid_pnl + chem_pnl
        bnh       = _bnh_pnl(df)
        edge      = combined - bnh

        rows.append({
            "symbol":      symbol,
            "scav_trades": len(scav_events),
            "raid_trades": len(raid_events),
            "chem_trades": len(chem_events),
            "scav_pnl":    scav_pnl,
            "raid_pnl":    raid_pnl,
            "chem_pnl":    chem_pnl,
            "combined":    combined,
            "bnh":         bnh,
            "edge":        edge,
        })

    rows.sort(key=lambda r: r["edge"], reverse=True)
    return rows, global_start, global_end


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return f"${v:>+10,.0f}"


def print_report(rows: list[dict], role: str,
                 global_start: str, global_end: str) -> None:
    if not rows:
        print("No results.")
        return

    run_scav = role in ("scavenger", "overseer")
    run_raid = role in ("raider",    "overseer")
    run_chem = role in ("chemist",   "overseer")

    # ── header ───────────────────────────────────────────────────────────────
    role_label = role.upper()
    if role == "overseer":
        active = []
        if run_scav: active.append("Scavenger")
        if run_raid: active.append("Raider")
        if run_chem: active.append("Chemist")
        role_label = f"OVERSEER  ({' + '.join(active)})"

    print(f"\n{'═'*100}")
    print(f"  VAULT 76 BACKTEST  —  Role: {role_label}")
    print(f"  Date range   : {global_start} → {global_end}")
    print(f"  Capital      : 100 shares / 1 contract per trade (no compounding)")
    print(f"{'═'*100}")

    # ── column header ────────────────────────────────────────────────────────
    cols = ["Symbol", "Trades"]
    if run_scav: cols.append("Scavenger")
    if run_raid: cols.append("Raider")
    if run_chem: cols.append("Chemist")
    cols += ["Combined", "B&H", "Edge"]

    hdr_parts = [f"  {'Symbol':<6}", f"{'Trades':>6}"]
    if run_scav: hdr_parts.append(f"{'Scavenger':>12}")
    if run_raid: hdr_parts.append(f"{'Raider':>12}")
    if run_chem: hdr_parts.append(f"{'Chemist':>12}")
    hdr_parts += [f"{'Combined':>12}", f"{'B&H':>12}", f"{'Edge':>12}"]
    print("  ".join(hdr_parts) if False else "".join(hdr_parts))

    sep = "─" * 98
    print(f"  {sep}")

    # ── rows ─────────────────────────────────────────────────────────────────
    for r in rows:
        trades = r["scav_trades"] + r["raid_trades"] + r["chem_trades"]
        parts  = [f"  {r['symbol']:<6}", f"{trades:>6}"]
        if run_scav: parts.append(f"{_fmt(r['scav_pnl']):>12}")
        if run_raid: parts.append(f"{_fmt(r['raid_pnl']):>12}")
        if run_chem: parts.append(f"{_fmt(r['chem_pnl']):>12}")
        parts += [f"{_fmt(r['combined']):>12}",
                  f"{_fmt(r['bnh']):>12}",
                  f"{_fmt(r['edge']):>12}"]
        print("".join(parts))

    # ── totals ───────────────────────────────────────────────────────────────
    print(f"  {sep}")
    tot_trades = sum(r["scav_trades"] + r["raid_trades"] + r["chem_trades"] for r in rows)
    t_scav = sum(r["scav_pnl"] for r in rows)
    t_raid = sum(r["raid_pnl"] for r in rows)
    t_chem = sum(r["chem_pnl"] for r in rows)
    t_comb = sum(r["combined"] for r in rows)
    t_bnh  = sum(r["bnh"]      for r in rows)
    t_edge = t_comb - t_bnh

    parts = [f"  {'TOTAL':<6}", f"{tot_trades:>6}"]
    if run_scav: parts.append(f"{_fmt(t_scav):>12}")
    if run_raid: parts.append(f"{_fmt(t_raid):>12}")
    if run_chem: parts.append(f"{_fmt(t_chem):>12}")
    parts += [f"{_fmt(t_comb):>12}", f"{_fmt(t_bnh):>12}", f"{_fmt(t_edge):>12}"]
    print("".join(parts))
    print(f"{'═'*100}")

    # ── summary ──────────────────────────────────────────────────────────────
    winners = [r["symbol"] for r in rows if r["edge"] > 0]
    losers  = [r["symbol"] for r in rows if r["edge"] <= 0]
    print(f"\n  Beats B&H: {len(winners)}/{len(rows)} symbols"
          + (f"  →  {', '.join(winners)}" if winners else ""))
    if losers:
        print(f"  Lags  B&H: {', '.join(losers)}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vault 76 backtest runner")
    parser.add_argument("--role", choices=VALID_ROLES, default="overseer",
                        help="Role to backtest (default: overseer)")
    parser.add_argument("symbols", nargs="*",
                        help="Symbols to test (default: full watchlist)")
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols] if args.symbols else WATCHLIST

    spy_path = os.path.join(DATA_DIR, "spy_history.csv")
    vix_path = os.path.join(DATA_DIR, "vix_history.csv")
    spy_df = pd.read_csv(spy_path, parse_dates=["datetime"]) if os.path.exists(spy_path) else None
    vix_df = pd.read_csv(vix_path, parse_dates=["datetime"]) if os.path.exists(vix_path) else None

    rows, start, end = run(symbols, args.role, spy_df, vix_df)
    print_report(rows, args.role, start, end)


if __name__ == "__main__":
    main()
