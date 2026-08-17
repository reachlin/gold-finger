"""Tests for overseer_status pure-logic helpers.

The bug these guard against: the operator (Claude) read only `tail -n30` of
overseer.log, missed an NVDA close + IBM open that had scrolled past, and
guessed the committed collateral belonged to the wrong position. These tests
lock in the two facts that must come from GROUND TRUTH (live positions), not a
log tail: (1) committed collateral is derived from actual short puts, and
(2) state-change events are extracted from the WHOLE log, not just its end.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from overseer_status import (
    parse_occ, committed_from_positions, short_options, recent_events,
    last_scan,
)


def test_parse_occ():
    sym, exp, typ, strike = parse_occ("IBM   260918P00220000")
    assert sym == "IBM" and typ == "P" and strike == 220.0 and exp == "2026-09-18"
    sym, exp, typ, strike = parse_occ("NVDA  260911P00205000")
    assert sym == "NVDA" and strike == 205.0 and typ == "P"


def test_committed_counts_short_puts_only():
    # A short put ties up strike*100 (cash-secured). A short call covered by
    # stock ties up no cash. This is exactly the NVDA-vs-IBM confusion.
    positions = [
        {"symbol": "IBM   260918P00220000", "shortQuantity": 1.0, "longQuantity": 0.0},
        {"symbol": "INTC  260918C00110000", "shortQuantity": 5.0, "longQuantity": 0.0},
    ]
    # Only the IBM put counts: 220 * 100 = 22,000 (call is covered → 0)
    assert committed_from_positions(positions, covered_calls=True) == 22000.0


def test_committed_two_puts():
    positions = [
        {"symbol": "IBM   260918P00220000", "shortQuantity": 1.0, "longQuantity": 0.0},
        {"symbol": "NVDA  260911P00205000", "shortQuantity": 1.0, "longQuantity": 0.0},
    ]
    assert committed_from_positions(positions) == 22000.0 + 20500.0


def test_short_options_lists_open_shorts():
    positions = [
        {"symbol": "IBM   260918P00220000", "shortQuantity": 1.0, "longQuantity": 0.0,
         "marketValue": -467.5},
        {"symbol": "AAPL", "shortQuantity": 0.0, "longQuantity": 100.0},
    ]
    shorts = short_options(positions)
    assert len(shorts) == 1 and shorts[0]["symbol"].startswith("IBM")


def test_recent_events_scans_whole_log_not_tail():
    # The open/close lines are near the TOP; a tail-based reader would miss them.
    lines = ["  [Reconcile] NVDA $205.0 closed on Schwab (fill $2.12, P&L $+279.00) — ledger updated"]
    lines += [f"  filler scan line {i}" for i in range(200)]
    lines += ["  [Overseer] ✅ Fill booked: IBM SELL_PUT $220.0 @ $4.37"]
    lines += [f"  more filler {i}" for i in range(200)]
    # Recurring per-scan advisory boilerplate must NOT drown out real events.
    lines += ["MAX LOSS:$21547.5/contract if assigned"] * 50
    lines += ["  ⚠  Strike above Kronos support floor — assignment risk elevated."] * 50
    ev = recent_events(lines)
    joined = " ".join(ev)
    assert "NVDA" in joined and "closed" in joined
    assert "IBM SELL_PUT $220.0" in joined
    assert "MAX LOSS" not in joined and "assignment risk" not in joined


def test_last_scan_parses_number():
    lines = ["  ■ END SCAN #14 [G:1959]  —  next in 5 min.",
             "  ■ END SCAN #15 [G:1960]  —  next in 5 min."]
    assert last_scan(lines)["num"] == 15


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"  ✓ {fn.__name__}")
        except Exception:
            print(f"  ✗ {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
