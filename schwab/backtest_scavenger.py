"""
Backtest for The Scavenger perk card — wheel strategy.

Simulates cash-secured put → covered call cycles on historical daily data.

State machine per symbol:
  FLAT       → scan for SELL_PUT signal
  PUT_OPEN   → iterate bars; early-exit at 50% profit, or hold to expiry
  HOLDING    → scan for SELL_CALL signal
  CALL_OPEN  → iterate bars; early-exit at 50% profit, or hold to expiry

Early exit: buy back the short option when its B-S mark-to-market falls to the
adaptive profit target (35%–65% of premium). Target scales with entry IV and DTE
— high-IV / long-dated options wait longer; thin-premium / short-dated exit fast.

Usage:
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/backtest_scavenger.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from trend_scanner import compute_indicators
from vault76.armory.scavenger import Scavenger
from options_pricer import (black_scholes_put, black_scholes_call,
                            historical_vol, yang_zhang_vol)

DTE_BARS    = 30    # trading days to expiration
SHARES      = 100   # 1 contract
MIN_HISTORY = 60    # bars needed before scanning starts
RISK_FREE   = 0.05

# Adaptive profit target bounds
_TARGET_MIN = 0.35
_TARGET_MAX = 0.65

FLAT       = "FLAT"
PUT_OPEN   = "PUT_OPEN"
HOLDING    = "HOLDING"
CALL_OPEN  = "CALL_OPEN"


# ---------------------------------------------------------------------------
# Adaptive profit target
# ---------------------------------------------------------------------------

def adaptive_profit_target(entry_iv: float, entry_dte: int) -> float:
    """
    Self-adjusting exit threshold in [35%, 65%]:
    - Higher IV  → higher target (fat premium is worth waiting for more decay)
    - Longer DTE → higher target (more time to collect full theta decay)

    Calibration anchors:
      iv=0.20, dte=15 → ~35%  (thin premium, exit fast)
      iv=0.60, dte=45 → ~50%  (standard tastyworks rule)
      iv=1.00, dte=75 → ~65%  (high-IV, long-dated — hold for bigger capture)
    """
    iv_factor  = min(max((entry_iv - 0.20) / 0.80, 0.0), 1.0)
    dte_factor = min(max((entry_dte - 15)  / 60,   0.0), 1.0)
    score      = 0.6 * iv_factor + 0.4 * dte_factor
    return _TARGET_MIN + (_TARGET_MAX - _TARGET_MIN) * score


def _entry_iv(snapshot: "pd.DataFrame") -> float:
    """Yang-Zhang vol at entry; falls back to close-only HV."""
    iv = yang_zhang_vol(snapshot) if "open" in snapshot.columns else 0.0
    return iv if iv > 0 else (historical_vol(snapshot["close"]) or 0.30)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def walk_forward_scavenger(df: pd.DataFrame, symbol: str) -> list[dict]:
    """
    Simulate the wheel strategy on df.
    Returns list of trade events with P&L.
    """
    ind  = compute_indicators(df).dropna().reset_index(drop=True)
    scav = Scavenger()
    n    = len(ind)

    events = []
    state  = FLAT
    cycle: dict = {}   # tracks current position state

    i = MIN_HISTORY
    while i < n:
        row      = ind.iloc[i]
        snapshot = ind.iloc[: i + 1]

        # ── FLAT: look for a put to sell ─────────────────────────────────────
        if state == FLAT:
            res = scav.scan(symbol, snapshot)
            if res["signal"] == "SELL_PUT":
                iv = _entry_iv(snapshot)
                cycle = {
                    "symbol":           symbol,
                    "put_entry_i":      i,
                    "put_strike":       res["strike"],
                    "put_premium":      res["premium"],
                    "put_expiry_i":     min(i + DTE_BARS, n - 1),
                    "put_close":        float(row["close"]),
                    "all_premiums":     res["premium"],
                    "put_entry_iv":     iv,
                    "put_profit_target": adaptive_profit_target(iv, DTE_BARS),
                }
                state = PUT_OPEN
                i    += 1   # step into the holding period bar-by-bar
            else:
                i += 1

        # ── PUT_OPEN: iterate bars; early exit or hold to expiry ──────────────
        elif state == PUT_OPEN:
            close      = float(row["close"])
            days_left  = max(cycle["put_expiry_i"] - i, 0)
            T_left     = days_left / 365
            hv         = historical_vol(snapshot["close"])
            sigma      = hv if hv > 0 else 0.30
            cur_val    = black_scholes_put(close, cycle["put_strike"], T_left,
                                          RISK_FREE, sigma)
            entry_prem = cycle["put_premium"]

            # Adaptive profit target — buy back early
            if cur_val <= entry_prem * cycle["put_profit_target"] and days_left > 0:
                pnl = (entry_prem - cur_val) * SHARES
                events.append({**cycle, "event": "put_early_exit",
                               "pnl": round(pnl, 2), "exit_i": i,
                               "exit_val": round(cur_val, 4)})
                cycle = {}
                state = FLAT
                i    += 1

            elif i >= cycle["put_expiry_i"]:
                # Reached expiry
                cycle["put_expiry_close"] = close
                if close <= cycle["put_strike"]:
                    cycle["cost_basis"] = cycle["put_strike"] - entry_prem
                    cycle["assigned_i"] = i
                    events.append({**cycle, "event": "put_assigned", "pnl": 0.0})
                    state = HOLDING
                else:
                    pnl = entry_prem * SHARES
                    events.append({**cycle, "event": "put_expired",
                                   "pnl": round(pnl, 2)})
                    cycle = {}
                    state = FLAT
                i += 1
            else:
                i += 1

        # ── HOLDING: look for a call to sell ─────────────────────────────────
        elif state == HOLDING:
            res = scav.scan(symbol, snapshot, cost_basis=cycle["cost_basis"])
            if res["signal"] == "SELL_CALL":
                iv = _entry_iv(snapshot)
                cycle["call_entry_i"]        = i
                cycle["call_strike"]         = res["strike"]
                cycle["call_premium"]        = res["premium"]
                cycle["all_premiums"]       += res["premium"]
                cycle["call_expiry_i"]       = min(i + DTE_BARS, n - 1)
                cycle["call_entry_iv"]       = iv
                cycle["call_profit_target"]  = adaptive_profit_target(iv, DTE_BARS)
                state = CALL_OPEN
                i    += 1   # step bar-by-bar
            else:
                i += 1

        # ── CALL_OPEN: iterate bars; early exit or hold to expiry ─────────────
        elif state == CALL_OPEN:
            close     = float(row["close"])
            days_left = max(cycle["call_expiry_i"] - i, 0)
            T_left    = days_left / 365
            hv        = historical_vol(snapshot["close"])
            sigma     = hv if hv > 0 else 0.30
            cur_val   = black_scholes_call(close, cycle["call_strike"], T_left,
                                           RISK_FREE, sigma)
            entry_prem = cycle["call_premium"]

            # Adaptive profit target — buy back early, hold shares, sell another call
            if cur_val <= entry_prem * cycle["call_profit_target"] and days_left > 0:
                pnl = (entry_prem - cur_val) * SHARES
                cycle["all_premiums"] += entry_prem - cur_val
                events.append({**cycle, "event": "call_early_exit",
                               "pnl": round(pnl, 2), "exit_i": i,
                               "exit_val": round(cur_val, 4)})
                state = HOLDING   # back to HOLDING to sell another call
                i    += 1

            elif i >= cycle["call_expiry_i"]:
                # Reached expiry
                cycle["close_at_call_expiry"] = close
                if close >= cycle["call_strike"]:
                    # Called away — shares sold at call_strike
                    share_pnl = (cycle["call_strike"] - cycle["put_strike"]) * SHARES
                    prem_pnl  = cycle["all_premiums"] * SHARES
                    pnl       = share_pnl + prem_pnl
                    events.append({**cycle, "event": "called_away", "pnl": round(pnl, 2)})
                    cycle = {}
                    state = FLAT
                else:
                    # Call expired worthless — keep premium, sell another call
                    pnl = cycle["call_premium"] * SHARES
                    events.append({**cycle, "event": "call_expired", "pnl": round(pnl, 2)})
                    state = HOLDING
                i += 1
            else:
                i += 1

    return events


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_scavenger_report(events: list[dict], init_price: float,
                           final_price: float, symbol: str = "") -> float:
    """Print a summary of all wheel-strategy events. Returns total P&L."""
    put_expired_cnt   = sum(1 for e in events if e["event"] == "put_expired")
    put_early_cnt     = sum(1 for e in events if e["event"] == "put_early_exit")
    put_assigned_cnt  = sum(1 for e in events if e["event"] == "put_assigned")
    call_expired_cnt  = sum(1 for e in events if e["event"] == "call_expired")
    call_early_cnt    = sum(1 for e in events if e["event"] == "call_early_exit")
    called_away_cnt   = sum(1 for e in events if e["event"] == "called_away")
    total_events      = len(events)

    total_pnl = sum(e["pnl"] for e in events)
    bnh_pnl   = (final_price - init_price) * SHARES

    tag = f" [{symbol}]" if symbol else ""
    print(f"\n{'─'*55}")
    print(f"  THE SCAVENGER — Wheel Backtest{tag}")
    print(f"{'─'*55}")
    print(f"  Events:         {total_events}")
    print(f"  Puts sold:      {put_expired_cnt + put_early_cnt + put_assigned_cnt}")
    print(f"    ↳ expired worthless: {put_expired_cnt}")
    print(f"    ↳ early exit (adaptive): {put_early_cnt}")
    print(f"    ↳ assigned:          {put_assigned_cnt}")
    print(f"  Calls sold:     {call_expired_cnt + call_early_cnt + called_away_cnt}")
    print(f"    ↳ expired worthless: {call_expired_cnt}")
    print(f"    ↳ early exit (adaptive): {call_early_cnt}")
    print(f"    ↳ called away:       {called_away_cnt}")
    print(f"{'─'*55}")
    print(f"  Wheel P&L:      ${total_pnl:+.2f}")
    print(f"  B&H P&L:        ${bnh_pnl:+.2f}  (buy {init_price:.2f} → {final_price:.2f})")
    print(f"{'─'*55}")
    return total_pnl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

WATCHLIST = [
    "NVDA", "AMD", "AAPL", "AMZN",
    "META", "MSFT", "GOOGL",
    "IBM", "INTC", "IONQ", "KO",
    "MMM", "XOM", "PG", "TSLA",
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    results = []

    for symbol in WATCHLIST:
        path = os.path.join(DATA_DIR, f"{symbol.lower()}_history.csv")
        if not os.path.exists(path):
            print(f"  [{symbol}] no data file — skip")
            continue

        df = pd.read_csv(path, parse_dates=["datetime"])
        if len(df) < MIN_HISTORY + DTE_BARS + 10:
            print(f"  [{symbol}] too little data — skip")
            continue

        events = walk_forward_scavenger(df, symbol)

        ind = compute_indicators(df).dropna().reset_index(drop=True)
        init_price  = float(ind.iloc[MIN_HISTORY]["close"])
        final_price = float(ind.iloc[-1]["close"])

        total_pnl = print_scavenger_report(events, init_price, final_price, symbol)
        bnh_pnl   = (final_price - init_price) * SHARES

        results.append({
            "symbol":    symbol,
            "events":    len(events),
            "wheel_pnl": total_pnl,
            "bnh_pnl":   bnh_pnl,
            "vs_bnh":    total_pnl - bnh_pnl,
        })

    if not results:
        print("\nNo results.")
        return

    print(f"\n\n{'='*55}")
    print(f"  SCAVENGER BACKTEST — FULL WATCHLIST SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Symbol':<8} {'Events':>6} {'Wheel P&L':>12} {'B&H P&L':>12} {'vs B&H':>10}")
    print(f"  {'─'*8} {'─'*6} {'─'*12} {'─'*12} {'─'*10}")
    for r in sorted(results, key=lambda x: x["wheel_pnl"], reverse=True):
        print(f"  {r['symbol']:<8} {r['events']:>6} "
              f"  ${r['wheel_pnl']:>+9.0f}   ${r['bnh_pnl']:>+9.0f}   ${r['vs_bnh']:>+8.0f}")
    total_wheel = sum(r["wheel_pnl"] for r in results)
    total_bnh   = sum(r["bnh_pnl"]   for r in results)
    print(f"  {'─'*8} {'─'*6} {'─'*12} {'─'*12} {'─'*10}")
    print(f"  {'TOTAL':<8} {'':>6}   ${total_wheel:>+9.0f}   ${total_bnh:>+9.0f}"
          f"   ${total_wheel - total_bnh:>+8.0f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
