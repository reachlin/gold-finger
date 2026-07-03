"""
Backtest for The Scavenger role — wheel strategy.

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

import numpy as np

from trend_scanner import compute_indicators
from vault76.armory.scavenger import Scavenger
from vault76.overseer import Overseer
from options_pricer import (black_scholes_put, black_scholes_call,
                            historical_vol, yang_zhang_vol,
                            adaptive_profit_target)
from strategy_params import (
    FAST_RISKOFF_DROP, FAST_RISKOFF_LOOKBACK, FAST_RISKOFF_COOLDOWN,
    SCAV_PROFIT_TARGET_MIN, SCAV_PROFIT_TARGET_MAX,
)

DTE_BARS    = 30    # trading days to expiration
SHARES      = 100   # 1 contract
MIN_HISTORY = 60    # bars needed before scanning starts
RISK_FREE   = 0.05

# Adaptive profit target bounds — from strategy_params.py
_TARGET_MIN = SCAV_PROFIT_TARGET_MIN
_TARGET_MAX = SCAV_PROFIT_TARGET_MAX

FLAT       = "FLAT"
PUT_OPEN   = "PUT_OPEN"
HOLDING    = "HOLDING"
CALL_OPEN  = "CALL_OPEN"


def _entry_iv(snapshot: "pd.DataFrame") -> float:
    """Yang-Zhang vol at entry; falls back to close-only HV."""
    iv = yang_zhang_vol(snapshot) if "open" in snapshot.columns else 0.0
    return iv if iv > 0 else (historical_vol(snapshot["close"]) or 0.30)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _build_regime_lookup(spy_df: pd.DataFrame | None,
                         vix_df: pd.DataFrame | None) -> tuple:
    """
    Pre-process SPY and VIX into fast per-date lookups.
    Returns (overseer, spy_ind, spy_date_idx, vix_by_date).
    Any of these may be None/empty if data is missing.
    """
    overseer = Overseer()
    if spy_df is None:
        return overseer, None, {}, {}

    spy_ind = compute_indicators(spy_df).dropna().reset_index(drop=True)
    spy_dates = pd.to_datetime(spy_ind["datetime"]).dt.date
    spy_date_idx = {d: i for i, d in enumerate(spy_dates)}

    vix_by_date = {}
    if vix_df is not None:
        col = "datetime" if "datetime" in vix_df.columns else "date"
        dates = pd.to_datetime(vix_df[col]).dt.date
        vix_by_date = dict(zip(dates, vix_df["close"]))

    return overseer, spy_ind, spy_date_idx, vix_by_date


def _get_regime(overseer: Overseer, cur_date, spy_ind, spy_date_idx: dict,
                vix_by_date: dict) -> str:
    """Classify regime at cur_date using pre-built lookups."""
    vix = vix_by_date.get(cur_date, 20.0)
    if vix >= 30.0:
        return Overseer.NUKED_ZONE  # VIX alone is sufficient — no SPY needed
    if spy_ind is None:
        return Overseer.WASTELAND
    spy_i = spy_date_idx.get(cur_date)
    if spy_i is None or spy_i < MIN_HISTORY:
        return Overseer.WASTELAND
    return overseer.classify(spy_ind.iloc[:spy_i + 1], vix)


def _spy_lookback_return(spy_ind: pd.DataFrame, spy_i: int) -> float:
    """
    SPY return over FAST_RISKOFF_LOOKBACK trading days ending at spy_i.
    Used to detect fast risk-off conditions.
    Returns 0.0 if insufficient history.
    """
    if spy_i < FAST_RISKOFF_LOOKBACK:
        return 0.0
    closes = spy_ind["close"].values
    return float((closes[spy_i] - closes[spy_i - FAST_RISKOFF_LOOKBACK])
                 / closes[spy_i - FAST_RISKOFF_LOOKBACK])




def walk_forward_scavenger(df: pd.DataFrame, symbol: str,
                           spy_df: pd.DataFrame | None = None,
                           vix_df: pd.DataFrame | None = None,
                           early_exit: bool = True,
                           sell_calls: bool = True) -> list[dict]:
    """
    Simulate the wheel strategy on df.
    spy_df / vix_df: if provided, the Overseer classifies regime at each bar.
    NUKED_ZONE blocks opening new puts/calls; existing positions ride through.

    FinRL enhancement applied at each FLAT bar before put entry:
      #1 Fast risk-off: if SPY dropped FAST_RISKOFF_DROP% in FAST_RISKOFF_LOOKBACK
         days, suppress new puts for FAST_RISKOFF_COOLDOWN bars.

    early_exit / sell_calls: set both False to reproduce the pre-wheel live
    scanner behavior — puts held to expiry, assignments liquidated at the
    expiry close, no covered-call phase. Used by compare_wheel_versions.py.

    Returns list of trade events with P&L.
    """
    ind  = compute_indicators(df).dropna().reset_index(drop=True)
    ind["_date"] = pd.to_datetime(ind["datetime"]).dt.date
    scav = Scavenger()
    n    = len(ind)

    overseer, spy_ind, spy_date_idx, vix_by_date = _build_regime_lookup(spy_df, vix_df)

    events = []
    state  = FLAT
    cycle: dict = {}

    # Fast risk-off cooldown tracker: bar index until which new puts are blocked.
    # Starts at -1 (no cooldown active). Updated whenever a SPY shock is detected.
    riskoff_until_i = -1

    i = MIN_HISTORY
    while i < n:
        row      = ind.iloc[i]
        snapshot = ind.iloc[: i + 1]
        cur_date = row["_date"]
        regime   = _get_regime(overseer, cur_date, spy_ind, spy_date_idx, vix_by_date)

        # ── FLAT: look for a put to sell ─────────────────────────────────────
        if state == FLAT:
            # Overseer gates entry: route by this stock's own ADX
            if "scavenger" not in overseer.recommend_roles(regime, row):
                i += 1
                continue

            # FinRL #1 — Fast risk-off gate
            # Check if SPY just dropped hard enough to (re-)trigger a cooldown.
            # We re-check even if a cooldown is active so a deeper drop resets the clock.
            if spy_ind is not None:
                spy_i = spy_date_idx.get(cur_date)
                if spy_i is not None:
                    ret = _spy_lookback_return(spy_ind, spy_i)
                    if ret <= FAST_RISKOFF_DROP:
                        # Extend cooldown from today
                        riskoff_until_i = i + FAST_RISKOFF_COOLDOWN
            if i <= riskoff_until_i:
                i += 1
                continue

            res = scav.scan(symbol, snapshot, regime=regime)
            if res["signal"] == "SELL_PUT":
                iv = _entry_iv(snapshot)
                cycle = {
                    "symbol":            symbol,
                    "put_entry_i":       i,
                    "put_strike":        res["strike"],
                    "put_premium":       res["premium"],
                    "put_expiry_i":      min(i + DTE_BARS, n - 1),
                    "put_close":         float(row["close"]),
                    "all_premiums":      res["premium"],
                    "put_entry_iv":      iv,
                    "put_profit_target": adaptive_profit_target(iv, DTE_BARS),
                    "put_regime":        regime,
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
            if (early_exit and days_left > 0
                    and cur_val <= entry_prem * cycle["put_profit_target"]):
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
                    if not sell_calls:
                        # Pre-wheel behavior: liquidate assigned shares at the
                        # expiry close instead of selling covered calls.
                        pnl = (close - cycle["put_strike"] + entry_prem) * SHARES
                        events.append({**cycle, "event": "put_assigned_liquidated",
                                       "pnl": round(pnl, 2)})
                        cycle = {}
                        state = FLAT
                        i += 1
                        continue
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
            res = scav.scan(symbol, snapshot, cost_basis=cycle["cost_basis"],
                            regime=regime)
            if res["signal"] == "SELL_CALL":
                iv = _entry_iv(snapshot)
                cycle["call_entry_i"]        = i
                cycle["call_strike"]         = res["strike"]
                cycle["call_premium"]        = res["premium"]
                cycle["all_premiums"]       += res["premium"]
                cycle["call_expiry_i"]       = min(i + DTE_BARS, n - 1)
                cycle["call_entry_iv"]       = iv
                cycle["call_profit_target"]  = adaptive_profit_target(iv, DTE_BARS)
                cycle["call_regime"]         = regime
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
            if (early_exit and days_left > 0
                    and cur_val <= entry_prem * cycle["call_profit_target"]):
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
    nuked_put_cnt     = sum(1 for e in events if e.get("put_regime") == Overseer.NUKED_ZONE)
    nuked_call_cnt    = sum(1 for e in events if e.get("call_regime") == Overseer.NUKED_ZONE)
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
    if nuked_put_cnt or nuked_call_cnt:
        print(f"  Opened in NUKED_ZONE: {nuked_put_cnt} puts, {nuked_call_cnt} calls")
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
    "UNH", "HD", "ABT",
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    results = []

    spy_path = os.path.join(DATA_DIR, "spy_history.csv")
    vix_path = os.path.join(DATA_DIR, "vix_history.csv")
    spy_df = pd.read_csv(spy_path, parse_dates=["datetime"]) if os.path.exists(spy_path) else None
    vix_df = pd.read_csv(vix_path, parse_dates=["datetime"]) if os.path.exists(vix_path) else None

    for symbol in WATCHLIST:
        path = os.path.join(DATA_DIR, f"{symbol.lower()}_history.csv")
        if not os.path.exists(path):
            print(f"  [{symbol}] no data file — skip")
            continue

        df = pd.read_csv(path, parse_dates=["datetime"])
        if len(df) < MIN_HISTORY + DTE_BARS + 10:
            print(f"  [{symbol}] too little data — skip")
            continue

        events = walk_forward_scavenger(df, symbol, spy_df=spy_df, vix_df=vix_df)

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
