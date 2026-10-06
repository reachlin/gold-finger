"""One open position per underlying — the hard rule in _pre_trade_check.

Changed 2026-09-29 at the user's request. The old rule allowed stacking
different strikes on the same ticker and only blocked an exact symbol+strike
duplicate; the LLM prompt merely asked for one NEW entry per symbol per day.
That is how the book ended up 99% concentrated in AMZN on 2026-09-18 — every
individual entry was legal.

Now any open short option on an underlying blocks a new entry on that
underlying, regardless of strike or expiry. The check lives in code rather
than the prompt on purpose: the prompt is advice a model may talk itself out
of, this is a gate it cannot reach.

Also pinned here: the matcher must compare the OCC *root*, not do a substring
test. `"V" in "AVGO  261023C00200000"` is True, so a Visa signal used to be
blocked by an unrelated Broadcom position.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import real_overseer as ro


def option(symbol_occ, put_call="PUT", strike=245.0, short=1.0):
    return {
        "instrument": {
            "assetType": "OPTION",
            "symbol": symbol_occ,
            "putCall": put_call,
            "strikePrice": strike,
        },
        "shortQuantity": short,
    }


def equity(symbol):
    return {"instrument": {"assetType": "EQUITY", "symbol": symbol}, "shortQuantity": 0.0}


def check(positions, signal, monkeypatch, avail=100000.0):
    """Run _pre_trade_check against a faked account."""
    monkeypatch.setattr(ro, "fetch_account", lambda c, h: {
        "currentBalances": {"availableFunds": avail, "pendingDeposits": 0},
        "positions": positions,
    })
    ov = ro.RealOverseer.__new__(ro.RealOverseer)   # no __init__: no LLM, no network
    return ro.RealOverseer._pre_trade_check(ov, None, "hash", signal)


SIG = {"symbol": "AMZN", "strike": 240.0, "signal": "SELL_PUT"}


# --- the new rule ----------------------------------------------------------

def test_a_different_strike_on_the_same_ticker_is_now_blocked(monkeypatch):
    """The case that built the AMZN concentration: every entry was a different
    strike, so nothing ever tripped the old duplicate check."""
    ok, why = check([option("AMZN  261009P00245000", strike=245.0)], SIG, monkeypatch)
    assert not ok
    assert "AMZN" in why


def test_a_different_expiry_on_the_same_ticker_is_blocked(monkeypatch):
    ok, why = check([option("AMZN  261023P00240000", strike=240.0)], SIG, monkeypatch)
    assert not ok


def test_the_exact_same_contract_is_still_blocked(monkeypatch):
    ok, _ = check([option("AMZN  261009P00240000", strike=240.0)], SIG, monkeypatch)
    assert not ok


def test_a_short_call_on_the_same_ticker_also_blocks(monkeypatch):
    """One position per underlying means per underlying, not per option type."""
    ok, _ = check([option("AMZN  261009C00260000", put_call="CALL", strike=260.0)],
                  SIG, monkeypatch)
    assert not ok


# --- what must still be allowed -------------------------------------------

def test_a_different_ticker_is_allowed(monkeypatch):
    ok, why = check([option("IBM   261023P00220000", strike=220.0)], SIG, monkeypatch)
    assert ok, why


def test_an_empty_account_is_allowed(monkeypatch):
    ok, why = check([], SIG, monkeypatch)
    assert ok, why


def test_a_long_option_does_not_block(monkeypatch):
    """Only SHORT positions tie up collateral and create assignment risk."""
    ok, why = check([option("AMZN  261009P00245000", strike=245.0, short=0.0)],
                    SIG, monkeypatch)
    assert ok, why


def test_a_stock_position_does_not_block(monkeypatch):
    ok, why = check([equity("AMZN")], SIG, monkeypatch)
    assert ok, why


# --- the substring bug -----------------------------------------------------

def test_a_ticker_that_is_a_substring_of_another_is_not_blocked(monkeypatch):
    """`"V" in "AVGO  261023P00200000"` is True. A Visa signal must not be
    blocked by a Broadcom position — match the OCC root, not a substring."""
    ok, why = check([option("AVGO  261023P00200000", strike=200.0)],
                    {"symbol": "V", "strike": 300.0, "signal": "SELL_PUT"},
                    monkeypatch)
    assert ok, f"V was blocked by an AVGO position: {why}"


def test_the_root_still_matches_its_own_ticker(monkeypatch):
    ok, _ = check([option("V     261023P00300000", strike=300.0)],
                  {"symbol": "V", "strike": 280.0, "signal": "SELL_PUT"},
                  monkeypatch)
    assert not ok, "a real V position must still block a V signal"


# --- the budget gate is unaffected ----------------------------------------

def test_insufficient_funds_still_blocks_first(monkeypatch):
    ok, why = check([], SIG, monkeypatch, avail=1000.0)
    assert not ok
    assert "collateral" in why.lower()


# ===========================================================================
# A RESTING order counts as an open position (2026-10-06)
# ===========================================================================
#
# The gate above reads filled positions only. That was sufficient while order
# limits crossed the bid and filled within seconds, so "placed but not yet a
# position" lasted no time at all. It stopped being sufficient the moment limits
# moved to the mid (2026-10-06): an unfilled DAY order now rests for the WHOLE
# session, and every one of the ~80 scans in that session sees a clean slate for
# the ticker and may place another.
#
# This is not hypothetical. Before the one-per-underlying rule existed, the order
# history shows exactly this stacking:
#   2026-08-27  IBM  -> 3 SELL_TO_OPEN, all filled ($220, $225, $225)
#   2026-08-07  XOM  -> 2 SELL_TO_OPEN, both filled ($144, $145)
#
# The budget gate does NOT cover this. It counts a resting order's collateral via
# _pending_collateral, so cash cannot be overdrawn -- but with enough free cash,
# two orders on one underlying both pass it. That protection is incidental to
# being capital-starved, not structural.
#
# Nor can the LLM cover it: build_prompt is given open positions and the peer
# signals of the CURRENT scan, never orders left resting by earlier scans.

def _order(instruction, occ, status="WORKING", price=1.50):
    return {
        "status": status,
        "price": price,
        "orderLegCollection": [
            {"instruction": instruction, "instrument": {"symbol": occ}}
        ],
    }


def check_with_orders(positions, orders, signal, monkeypatch, avail=100000.0,
                      pending=None, raise_on_fetch=False):
    monkeypatch.setattr(ro, "fetch_account", lambda c, h: {
        "currentBalances": {"availableFunds": avail, "pendingDeposits": 0},
        "positions": positions,
    })

    def _fetch(client, account_hash, days_back=90):
        if raise_on_fetch:
            raise RuntimeError("Schwab orders endpoint down")
        return orders

    monkeypatch.setattr(ro, "fetch_orders", _fetch)
    monkeypatch.setattr(ro, "_load_pending", lambda: pending or [])
    ov = ro.RealOverseer.__new__(ro.RealOverseer)
    return ro.RealOverseer._pre_trade_check(ov, None, "hash", signal)


def test_a_resting_sell_to_open_blocks_a_second_entry(monkeypatch):
    """The gap this closes: no position yet, but an order is already working."""
    ok, msg = check_with_orders(
        [], [_order("SELL_TO_OPEN", "AMZN  261023P00245000")], SIG, monkeypatch)
    assert not ok
    assert "resting" in msg.lower() or "working" in msg.lower(), msg


def test_a_resting_order_on_another_ticker_does_not_block(monkeypatch):
    ok, _ = check_with_orders(
        [], [_order("SELL_TO_OPEN", "GOOGL 261030P00330000")], SIG, monkeypatch)
    assert ok


def test_the_occ_root_is_matched_not_a_substring(monkeypatch):
    """Same trap as the position check: "V" is inside "AVGO  ...".""" 
    ok, _ = check_with_orders(
        [], [_order("SELL_TO_OPEN", "AVGO  261023P00200000")],
        {"symbol": "V", "strike": 300.0, "signal": "SELL_PUT"}, monkeypatch)
    assert ok, "a Broadcom order must not block Visa"


def test_a_dead_order_does_not_block(monkeypatch):
    """Only live orders matter -- a cancelled or expired one frees the ticker."""
    for status in ("CANCELED", "EXPIRED", "REJECTED", "REPLACED", "FILLED"):
        ok, _ = check_with_orders(
            [], [_order("SELL_TO_OPEN", "AMZN  261023P00245000", status=status)],
            SIG, monkeypatch)
        assert ok, f"{status} order must not block a new entry"


def test_a_resting_cover_does_not_block(monkeypatch):
    """A BUY_TO_CLOSE rests for the life of every position by design. Blocking on
    it would mean one assignment-free month per ticker, not one position."""
    ok, _ = check_with_orders(
        [], [_order("BUY_TO_CLOSE", "AMZN  261023P00245000")], SIG, monkeypatch)
    assert ok


def test_falls_back_to_pending_file_when_schwab_orders_fail(monkeypatch):
    """The orders endpoint is not always reachable, and failing OPEN here would
    reinstate the exact hole being closed. pending_orders.json is written on
    every placement, so it answers the question without the network."""
    ok, msg = check_with_orders(
        [], [], SIG, monkeypatch, raise_on_fetch=True,
        pending=[{"signal": "SELL_PUT", "symbol": "AMZN", "strike": 245.0,
                  "duration": "DAY"}])
    assert not ok, "a locally-tracked resting order must still block"
    assert "AMZN" in msg


def test_pending_cover_entries_do_not_block_via_the_fallback(monkeypatch):
    """The pending file is mostly GTC covers; they must not look like entries."""
    ok, _ = check_with_orders(
        [], [], SIG, monkeypatch, raise_on_fetch=True,
        pending=[{"signal": "BUY_TO_CLOSE", "symbol": "AMZN", "strike": 245.0,
                  "duration": "GTC"}])
    assert ok


def test_a_filled_position_still_blocks_with_no_orders(monkeypatch):
    """Regression: the original rule must keep working."""
    ok, msg = check_with_orders(
        [option("AMZN  261023P00245000")], [], SIG, monkeypatch)
    assert not ok
    assert "Already short" in msg


def test_clean_slate_still_passes(monkeypatch):
    ok, msg = check_with_orders([], [], SIG, monkeypatch)
    assert ok, msg
    assert "Pre-check OK" in msg
