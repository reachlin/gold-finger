"""
Portfolio-level backtest — all symbols share ONE cash pool.

The per-symbol backtests (backtest_scavenger.py etc.) give every symbol
unlimited independent capital, so they cannot measure allocation policy:
which candidate gets the collateral when several signals compete. This
harness runs the same wheel mechanics across all symbols in date lockstep
under a shared cash constraint, so the question "two small caps or one
large?" produces a number.

Policies (see allocator.py):
  watchlist — first-come-first-served in watchlist order (pre-step-2 live
              behavior)
  allocator — cash-freeing first, then premium/day per collateral $ with
              a same-symbol concentration penalty (deployed step 2)

Scavenger book only (puts + covered calls). Raider/Chemist/router are out
of scope — allocation contention is a collateral problem, and the wheel
is what consumes collateral.

Usage:
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/backtest_portfolio.py
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/backtest_portfolio.py --capitals 30000,100000
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

import allocator
from trend_scanner import compute_indicators
from vault76.armory.scavenger import Scavenger
from options_pricer import black_scholes_put, black_scholes_call, historical_vol
from backtest_scavenger import (
    FLAT, PUT_OPEN, HOLDING, CALL_OPEN,
    DTE_BARS, SHARES, MIN_HISTORY, RISK_FREE,
    _build_regime_lookup, _get_regime, _entry_iv, _spy_lookback_return,
)
from options_pricer import adaptive_profit_target
from strategy_params import FAST_RISKOFF_DROP, FAST_RISKOFF_COOLDOWN

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


PRIOR_FLOOR = 0.05   # minimum edge-prior weight — unproven/negative-edge
                     # names stay tradeable, just last in line


def edge_priors(dfs: dict, cutoff, spy_df=None, vix_df=None,
                floor: float = PRIOR_FLOOR) -> dict:
    """
    Score-v2 prior: per-symbol scavenger edge per collateral dollar,
    computed ONLY from data before `cutoff` — no lookahead into the
    evaluation window. Floored so weak names are deprioritized, not banned.
    """
    from backtest_scavenger import walk_forward_scavenger
    priors = {}
    for sym, df in dfs.items():
        dates = pd.to_datetime(df["datetime"]).dt.date
        prior_df = df[dates < cutoff].reset_index(drop=True)
        if len(prior_df) < MIN_HISTORY + DTE_BARS + 10:
            continue
        frame = compute_indicators(prior_df).dropna().reset_index(drop=True)
        if len(frame) <= MIN_HISTORY:
            continue
        pnl  = sum(e["pnl"] for e in
                   walk_forward_scavenger(prior_df, sym, spy_df, vix_df))
        bnh  = (float(frame.iloc[-1]["close"])
                - float(frame.iloc[MIN_HISTORY]["close"])) * SHARES
        collateral = 0.95 * float(frame.iloc[-1]["close"]) * SHARES
        priors[sym] = max((pnl - bnh) / collateral, floor)
    return priors


def _order_candidates(candidates: list[dict], policy: str,
                      open_counts: dict, watchlist: list[str],
                      priors: "dict | None" = None) -> list[dict]:
    """Processing order for one day's SELL_PUT candidates."""
    if policy == "allocator":
        return allocator.rank_signals(candidates, open_counts)
    if policy == "allocator_v2":
        return allocator.rank_signals(candidates, open_counts, priors=priors)
    order = {s: k for k, s in enumerate(watchlist)}
    return sorted(candidates, key=lambda c: order.get(c.get("symbol"), 999))


def walk_forward_portfolio(dfs: dict, capital: float,
                           policy: str = "allocator",
                           watchlist: "list[str] | None" = None,
                           spy_df=None, vix_df=None,
                           priors: "dict | None" = None) -> dict:
    """
    Run the wheel across all symbols under one shared cash pool.
    Event pnl values are booked so that sum(event pnl) == total pnl after
    the forced end-of-data liquidation.
    """
    watchlist = watchlist or list(dfs)
    scav = Scavenger()

    ind, date_idx = {}, {}
    for sym in watchlist:
        df = dfs.get(sym)
        if df is None or len(df) < MIN_HISTORY + 10:
            continue
        frame = compute_indicators(df).dropna().reset_index(drop=True)
        frame["_date"] = pd.to_datetime(frame["datetime"]).dt.date
        ind[sym] = frame
        date_idx[sym] = {d: i for i, d in enumerate(frame["_date"])}
    watchlist = [s for s in watchlist if s in ind]

    all_dates = sorted({d for sym in watchlist for d in date_idx[sym]})
    overseer, spy_ind, spy_date_idx, vix_by_date = _build_regime_lookup(spy_df, vix_df)

    books = {sym: {"state": FLAT, "cyc": {}} for sym in watchlist}
    cash, committed = float(capital), 0.0
    events: list[dict] = []
    trades = skipped = 0
    premium_collected = 0.0
    riskoff_until_g = -1

    def _book_event(sym, kind, pnl, **extra):
        events.append({"symbol": sym, "event": kind, "pnl": round(pnl, 2),
                       **extra})

    for g, date in enumerate(all_dates):
        regime = _get_regime(overseer, date, spy_ind, spy_date_idx, vix_by_date)

        if spy_ind is not None:
            spy_i = spy_date_idx.get(date)
            if spy_i is not None and \
                    _spy_lookback_return(spy_ind, spy_i) <= FAST_RISKOFF_DROP:
                riskoff_until_g = g + FAST_RISKOFF_COOLDOWN

        candidates = []
        for sym in watchlist:
            i = date_idx[sym].get(date)
            if i is None:
                continue
            book, frame = books[sym], ind[sym]
            row, snapshot = frame.iloc[i], frame.iloc[:i + 1]
            cyc = book["cyc"]

            if book["state"] == FLAT:
                if i < MIN_HISTORY or g <= riskoff_until_g:
                    continue
                if "scavenger" not in overseer.recommend_roles(regime, row):
                    continue
                res = scav.scan(sym, snapshot, regime=regime)
                if res["signal"] == "SELL_PUT":
                    candidates.append({**res, "_i": i})

            elif book["state"] == PUT_OPEN:
                close = float(row["close"])
                strike, prem = cyc["strike"], cyc["premium"]
                days_left = max(cyc["expiry_i"] - i, 0)
                hv = historical_vol(snapshot["close"].iloc[-22:])
                val = black_scholes_put(close, strike, days_left / 365,
                                        RISK_FREE, hv if hv > 0 else 0.30)
                if days_left > 0 and val <= prem * cyc["pt"]:
                    cash -= val * SHARES
                    committed -= strike * SHARES
                    _book_event(sym, "put_early_exit", (prem - val) * SHARES)
                    book["state"], book["cyc"] = FLAT, {}
                elif i >= cyc["expiry_i"]:
                    committed -= strike * SHARES
                    if close <= strike:
                        cash -= strike * SHARES
                        _book_event(sym, "put_assigned", prem * SHARES)
                        cyc["cost_basis"] = strike - prem
                        book["state"] = HOLDING
                    else:
                        _book_event(sym, "put_expired", prem * SHARES)
                        book["state"], book["cyc"] = FLAT, {}

            elif book["state"] == HOLDING:
                res = scav.scan(sym, snapshot, regime=regime,
                                cost_basis=cyc["cost_basis"])
                if res["signal"] == "SELL_CALL":
                    cash += res["premium"] * SHARES
                    premium_collected += res["premium"] * SHARES
                    trades += 1
                    cyc.update(call_strike=res["strike"],
                               call_premium=res["premium"],
                               call_expiry_i=min(i + DTE_BARS, len(frame) - 1),
                               call_pt=adaptive_profit_target(
                                   _entry_iv(snapshot), DTE_BARS))
                    book["state"] = CALL_OPEN

            elif book["state"] == CALL_OPEN:
                close = float(row["close"])
                cstrike, cprem = cyc["call_strike"], cyc["call_premium"]
                days_left = max(cyc["call_expiry_i"] - i, 0)
                hv = historical_vol(snapshot["close"].iloc[-22:])
                val = black_scholes_call(close, cstrike, days_left / 365,
                                         RISK_FREE, hv if hv > 0 else 0.30)
                if days_left > 0 and val <= cprem * cyc["call_pt"]:
                    cash -= val * SHARES
                    _book_event(sym, "call_early_exit", (cprem - val) * SHARES)
                    book["state"] = HOLDING
                elif i >= cyc["call_expiry_i"]:
                    if close >= cstrike:
                        cash += cstrike * SHARES
                        _book_event(sym, "called_away",
                                    (cstrike - cyc["strike"] + cprem) * SHARES)
                        book["state"], book["cyc"] = FLAT, {}
                    else:
                        _book_event(sym, "call_expired", cprem * SHARES)
                        book["state"] = HOLDING

        # ── Entries: the day's candidates compete for shared cash ────────
        open_counts = {s: 1 for s, b in books.items() if b["state"] != FLAT}
        for c in _order_candidates(candidates, policy, open_counts, watchlist,
                                   priors=priors):
            sym, need = c["symbol"], float(c["strike"]) * SHARES
            if cash - committed >= need:
                i = c["_i"]
                snapshot = ind[sym].iloc[:i + 1]
                cash += c["premium"] * SHARES
                committed += need
                premium_collected += c["premium"] * SHARES
                trades += 1
                books[sym]["cyc"] = {
                    "strike":   c["strike"],
                    "premium":  c["premium"],
                    "expiry_i": min(i + DTE_BARS, len(ind[sym]) - 1),
                    "pt": adaptive_profit_target(_entry_iv(snapshot), DTE_BARS),
                }
                books[sym]["state"] = PUT_OPEN
            else:
                skipped += 1

    # ── Forced liquidation at end of data ────────────────────────────────
    for sym in watchlist:
        book, frame = books[sym], ind[sym]
        cyc, close = book["cyc"], float(frame.iloc[-1]["close"])
        hv = historical_vol(frame["close"].iloc[-22:])
        sigma = hv if hv > 0 else 0.30
        if book["state"] == PUT_OPEN:
            val = black_scholes_put(close, cyc["strike"], 1 / 365,
                                    RISK_FREE, sigma)
            cash -= val * SHARES
            committed -= cyc["strike"] * SHARES
            _book_event(sym, "end_put_buyback",
                        (cyc["premium"] - val) * SHARES)
        elif book["state"] == HOLDING:
            cash += close * SHARES
            _book_event(sym, "end_liquidate_shares",
                        (close - cyc["strike"]) * SHARES)
        elif book["state"] == CALL_OPEN:
            val = black_scholes_call(close, cyc["call_strike"], 1 / 365,
                                     RISK_FREE, sigma)
            cash += close * SHARES - val * SHARES
            _book_event(sym, "end_liquidate_shares",
                        (close - cyc["strike"]
                         + cyc["call_premium"] - val) * SHARES)

    return {
        "events":            events,
        "pnl":               round(cash - capital, 2),
        "final_value":       round(cash, 2),
        "premium_collected": round(premium_collected, 2),
        "trades":            trades,
        "skipped_for_cash":  skipped,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _bnh_equal_weight(dfs: dict, capital: float) -> float:
    """Equal-weight buy & hold P&L over the same frames (fractional shares)."""
    pnl, n = 0.0, len(dfs)
    for df in dfs.values():
        frame = compute_indicators(df).dropna().reset_index(drop=True)
        first = float(frame.iloc[MIN_HISTORY]["close"])
        last  = float(frame.iloc[-1]["close"])
        pnl  += (capital / n) * (last / first - 1)
    return round(pnl, 2)


def main():
    from run_backtest import WATCHLIST, _load_df

    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", help="default: full watchlist")
    ap.add_argument("--capitals", default="30000,100000")
    ap.add_argument("--start", default=None,
                    help="truncate data to start here (YYYY-MM-DD) — with "
                         "--prior-cutoff before it, the v2 prior is fully "
                         "out-of-sample for the evaluation window")
    ap.add_argument("--prior-cutoff", default="2024-01-01",
                    help="v2 edge priors use only data before this date")
    args = ap.parse_args()

    symbols  = [s.upper() for s in args.symbols] or WATCHLIST
    capitals = [float(c) for c in args.capitals.split(",")]
    cutoff   = pd.to_datetime(args.prior_cutoff).date()

    full_dfs = {s: d for s in symbols if (d := _load_df(s)) is not None}
    dfs = full_dfs
    if args.start:
        start = pd.to_datetime(args.start).date()
        dfs = {s: d[pd.to_datetime(d["datetime"]).dt.date >= start]
               .reset_index(drop=True) for s, d in full_dfs.items()}
        dfs = {s: d for s, d in dfs.items()
               if len(d) >= MIN_HISTORY + DTE_BARS + 10}

    spy_path = os.path.join(DATA_DIR, "spy_history.csv")
    vix_path = os.path.join(DATA_DIR, "vix_history.csv")
    spy_df = pd.read_csv(spy_path, parse_dates=["datetime"]) if os.path.exists(spy_path) else None
    vix_df = pd.read_csv(vix_path, parse_dates=["datetime"]) if os.path.exists(vix_path) else None

    t0 = time.time()
    # v2 priors always come from FULL history strictly before the cutoff —
    # when --start >= --prior-cutoff the evaluation is fully out-of-sample
    priors = edge_priors(full_dfs, cutoff, spy_df, vix_df)

    window = f"from {args.start}" if args.start else "full history"
    print("═" * 96)
    print("  PORTFOLIO BACKTEST — shared cash pool, wheel (Scavenger) book")
    print(f"  Symbols: {len(dfs)}   Window: {window}   "
          f"v2 priors: data < {cutoff}")
    print(f"  Priors: " + "  ".join(f"{s}:{p:.2f}"
          for s, p in sorted(priors.items(), key=lambda kv: -kv[1])))
    print("═" * 96)
    print(f"  {'Capital':>10s}  {'Policy':13s}{'P&L':>14s}{'Premiums':>13s}"
          f"{'Trades':>8s}{'Skipped':>9s}{'B&H eq-w':>13s}{'Edge':>13s}")
    print("  " + "─" * 92)
    for capital in capitals:
        bnh = _bnh_equal_weight(dfs, capital)
        for policy in ("watchlist", "allocator", "allocator_v2"):
            r = walk_forward_portfolio(dfs, capital, policy=policy,
                                       watchlist=list(dfs),
                                       spy_df=spy_df, vix_df=vix_df,
                                       priors=priors)
            print(f"  {capital:>10,.0f}  {policy:13s}{r['pnl']:>+14,.0f}"
                  f"{r['premium_collected']:>13,.0f}{r['trades']:>8d}"
                  f"{r['skipped_for_cash']:>9d}{bnh:>+13,.0f}"
                  f"{r['pnl'] - bnh:>+13,.0f}")
        print("  " + "─" * 92)
    print(f"  Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
