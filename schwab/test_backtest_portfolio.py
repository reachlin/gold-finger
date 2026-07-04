"""
Tests for backtest_portfolio.py — all symbols under ONE shared cash pool,
so allocation policy (watchlist order vs allocator ranking) becomes
measurable. The per-symbol backtests give every symbol unlimited capital
and cannot see allocation effects by construction.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import backtest_portfolio as bp


def _wavy(n=300, base=100.0, amp=1.0, seed=5):
    """Sideways series; amp scales volatility (HV) without moving RSI/ADX
    much, so two symbols fire SELL_PUT on the same days with different
    premiums — exactly the allocation dilemma."""
    rng = np.random.default_rng(seed)
    w = np.cumsum(rng.normal(0, 1.0, n))
    w = w - np.linspace(w[0], w[-1], n)          # pin ends → sideways
    closes = base + amp * w
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open": closes * 0.999, "high": closes * 1.012,
        "low": closes * 0.988, "close": closes,
        "volume": np.full(n, 3e6),
    })


def _dfs(amp_a=1.6, amp_b=3.2, n=300):
    """Same seed/shape (identical RSI/ADX → same signal days), different
    volatility scale (different HV → different premiums)."""
    return {"AAA": _wavy(n=n, amp=amp_a), "BBB": _wavy(n=n, amp=amp_b)}


class TestOrderCandidates:
    def test_watchlist_policy_uses_list_position(self):
        cands = [{"symbol": "BBB", "signal": "SELL_PUT", "strike": 95.0,
                  "premium": 3.0, "dte": 30},
                 {"symbol": "AAA", "signal": "SELL_PUT", "strike": 95.0,
                  "premium": 1.0, "dte": 30}]
        out = bp._order_candidates(cands, "watchlist", {}, ["AAA", "BBB"])
        assert [c["symbol"] for c in out] == ["AAA", "BBB"]

    def test_allocator_policy_ranks_by_efficiency(self):
        cands = [{"symbol": "AAA", "signal": "SELL_PUT", "strike": 95.0,
                  "premium": 1.0, "dte": 30},
                 {"symbol": "BBB", "signal": "SELL_PUT", "strike": 95.0,
                  "premium": 3.0, "dte": 30}]
        out = bp._order_candidates(cands, "allocator", {}, ["AAA", "BBB"])
        assert [c["symbol"] for c in out] == ["BBB", "AAA"]


class TestWalkForwardPortfolio:
    def test_runs_and_reports(self):
        res = bp.walk_forward_portfolio(_dfs(), capital=100_000)
        assert set(res) >= {"events", "pnl", "premium_collected",
                            "trades", "skipped_for_cash", "final_value"}
        assert res["final_value"] == pytest.approx(100_000 + res["pnl"])
        assert res["skipped_for_cash"] >= 0

    def test_ample_capital_policies_identical(self):
        a = bp.walk_forward_portfolio(_dfs(), capital=10_000_000,
                                      policy="watchlist")
        b = bp.walk_forward_portfolio(_dfs(), capital=10_000_000,
                                      policy="allocator")
        assert a["skipped_for_cash"] == 0 and b["skipped_for_cash"] == 0
        assert a["pnl"] == pytest.approx(b["pnl"])

    def test_tight_capital_forces_skips(self):
        res = bp.walk_forward_portfolio(_dfs(), capital=10_000)
        assert res["skipped_for_cash"] > 0

    def test_tight_capital_allocator_takes_higher_premium_first(self):
        """One contract of cash; AAA is first in watchlist but BBB pays
        more premium for the same collateral."""
        dfs = _dfs()
        wl = bp.walk_forward_portfolio(dfs, capital=10_000,
                                       policy="watchlist",
                                       watchlist=["AAA", "BBB"])
        al = bp.walk_forward_portfolio(dfs, capital=10_000,
                                       policy="allocator",
                                       watchlist=["AAA", "BBB"])
        first_put = lambda r: next(e["symbol"] for e in r["events"]
                                   if e["event"].startswith("put_"))
        assert first_put(wl) == "AAA"
        assert first_put(al) == "BBB"
        # Which choice earns more is path-dependent (assignments trap
        # capital) — that outcome is what the real-data run measures, so
        # only the allocation *behavior* is asserted here.
        assert wl["trades"] > 0 and al["trades"] > 0

    def test_cash_accounting_is_consistent(self):
        """Everything is liquidated at the end: pnl equals sum of event pnl."""
        res = bp.walk_forward_portfolio(_dfs(), capital=50_000)
        assert res["pnl"] == pytest.approx(
            sum(e["pnl"] for e in res["events"]), abs=0.05)
