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
