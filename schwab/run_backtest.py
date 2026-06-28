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

# Fixed column widths — every cell uses these, header and data alike
_W_SYM    = 8    # "  SYMBOL" — 2-space indent + 6 chars symbol
_W_TRADES = 8    # right-align trades, 2-char gap from symbol column
_W_PNL    = 13   # "  $ +14,784" — right-aligned dollar amount


def _fmt(v: float) -> str:
    """Format a dollar P&L value into exactly _W_PNL chars."""
    raw = f"${v:>+10,.0f}"          # always 11 chars:  "$ +14,784"
    return f"{raw:>{_W_PNL}}"       # right-pad to 13:  "  $ +14,784"


def _row(sym: str, trades: int | str, cols: list[float | str]) -> str:
    """Build one table row with consistent column widths."""
    sym_cell    = f"  {sym:<{_W_SYM - 2}}"           # "  MMM   " (8 chars)
    trades_cell = f"{str(trades):>{_W_TRADES}}"       # "      28"  (8 chars)
    pnl_cells   = [f"{c:>{_W_PNL}}" if isinstance(c, str) else _fmt(float(c))
                   for c in cols]
    return sym_cell + trades_cell + "".join(pnl_cells)


def print_report(rows: list[dict], role: str,
                 global_start: str, global_end: str) -> None:
    if not rows:
        print("No results.")
        return

    run_scav = role in ("scavenger", "overseer")
    run_raid = role in ("raider",    "overseer")
    run_chem = role in ("chemist",   "overseer")

    # Ordered list of (header_label, data_key) for P&L columns
    pnl_cols: list[tuple[str, str]] = []
    if run_scav: pnl_cols.append(("Scavenger", "scav_pnl"))
    if run_raid: pnl_cols.append(("Raider",    "raid_pnl"))
    if run_chem: pnl_cols.append(("Chemist",   "chem_pnl"))
    pnl_cols += [("Combined", "combined"), ("B&H", "bnh"), ("Edge", "edge")]

    # Compute border width from actual column count
    width = _W_SYM + _W_TRADES + _W_PNL * len(pnl_cols)

    # ── header ───────────────────────────────────────────────────────────────
    role_label = role.upper()
    if role == "overseer":
        active = (["Scavenger"] if run_scav else []) + \
                 (["Raider"]    if run_raid else []) + \
                 (["Chemist"]   if run_chem else [])
        role_label = f"OVERSEER  ({' + '.join(active)})"

    print(f"\n{'═' * width}")
    print(f"  VAULT 76 BACKTEST  —  Role: {role_label}")
    print(f"  Date range   : {global_start} → {global_end}")
    print(f"  Capital      : 100 shares / 1 contract per trade (no compounding)")
    print(f"{'═' * width}")

    # ── column labels ────────────────────────────────────────────────────────
    print(_row("Symbol", "Trades", [lbl for lbl, _ in pnl_cols]))
    print("  " + "─" * (width - 2))

    # ── data rows ────────────────────────────────────────────────────────────
    for r in rows:
        trades = r["scav_trades"] + r["raid_trades"] + r["chem_trades"]
        print(_row(r["symbol"], trades, [r[key] for _, key in pnl_cols]))

    # ── totals ───────────────────────────────────────────────────────────────
    print("  " + "─" * (width - 2))
    tot_trades = sum(r["scav_trades"] + r["raid_trades"] + r["chem_trades"] for r in rows)
    totals = {
        "scav_pnl": sum(r["scav_pnl"] for r in rows),
        "raid_pnl": sum(r["raid_pnl"] for r in rows),
        "chem_pnl": sum(r["chem_pnl"] for r in rows),
        "combined": sum(r["combined"] for r in rows),
        "bnh":      sum(r["bnh"]      for r in rows),
        "edge":     sum(r["combined"] for r in rows) - sum(r["bnh"] for r in rows),
    }
    print(_row("TOTAL", tot_trades, [totals[key] for _, key in pnl_cols]))

    # COMBO: bottom-line portfolio view — per-role columns blank, only Combined/B&H/Edge
    _BOTTOM_LINE = {"combined", "bnh", "edge"}
    combo_cols = [totals[key] if key in _BOTTOM_LINE else "—" for _, key in pnl_cols]
    print(_row("COMBO", "—", combo_cols))
    print(f"{'═' * width}")

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
