"""The per-symbol cap must refuse a signal BEFORE the expensive path.

Found live 2026-10-14, within an hour of arming MULTI_CONTRACT_SIZING.

Arming the cap retired the one-position-per-underlying check in
_pre_trade_check. That check used to return False from the positions loop,
BEFORE the method reached fetch_orders. With it gone, every scan holding an
at-cap symbol runs the whole path: one LLM call, one fetch_account, and one
fetch_orders — and fetch_orders reads /accounts/{h}/orders, the same endpoint
order WRITES use. That is the path whose rate limit killed the NVDA GTC cover
on 2026-10-13.

Observed cost: NVDA sat at $21,500 against a $9,168 cap and was refused in
four consecutive scans, each burning an orders read. At a 5-minute interval
that is ~78 wasted reads a day on the endpoint that must stay free for covers.

It also read wrong. Sizing refuses after the scanner has already printed its
APPROVED verdict block, so the log said:

    🤖 Overseer: ✅ Y
    [Pre-trade] Pre-check OK
    [Sizing] ⊘ NVDA SELL_PUT — per-symbol cap full
    ===APPROVED===

No order was placed and no ledger row was written — real mode books on fill
confirmation, and _log_option_trade is paper-only — so nothing was corrupted.
But a log that says APPROVED when nothing was sent is how a human or a future
session misreads the book.

So the cap now gates in the scan loop beside the earnings gate, before the LLM
sees the signal. The sizing call in _place_order stays as the authoritative
check: the scanner's cash figure is a cached sync, so the cap is re-applied
against live balances at order time.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import live_scanner as ls  # noqa: E402


NVDA = {"signal": "SELL_PUT", "symbol": "NVDA", "strike": 215.0}


def test_an_at_cap_symbol_is_blocked(monkeypatch):
    monkeypatch.setattr(ls, "MULTI_CONTRACT_SIZING", True)
    monkeypatch.setattr(ls, "_symbol_exposure", lambda sym: 21_500.0)
    why = ls.symbol_cap_block(NVDA, cash=45_838)
    assert why is not None
    assert "21,500" in why and "9,16" in why, why


def test_a_symbol_with_room_is_not_blocked(monkeypatch):
    monkeypatch.setattr(ls, "MULTI_CONTRACT_SIZING", True)
    monkeypatch.setattr(ls, "_symbol_exposure", lambda sym: 0.0)
    assert ls.symbol_cap_block(NVDA, cash=45_838) is None


def test_an_unheld_symbol_is_never_blocked_by_the_cap(monkeypatch):
    """
    The floor applies to a name we do not hold: 20% of $45,838 is under one
    contract, but a first contract is always allowed. The cap gate must not
    pre-empt the floor.
    """
    monkeypatch.setattr(ls, "MULTI_CONTRACT_SIZING", True)
    monkeypatch.setattr(ls, "_symbol_exposure", lambda sym: 0.0)
    assert ls.symbol_cap_block(
        {"signal": "SELL_PUT", "symbol": "META", "strike": 725.0},
        cash=45_838) is None


def test_partial_room_is_not_blocked(monkeypatch):
    """$600k: cap $120,000, $21,500 held — room for more, so let it through."""
    monkeypatch.setattr(ls, "MULTI_CONTRACT_SIZING", True)
    monkeypatch.setattr(ls, "_symbol_exposure", lambda sym: 21_500.0)
    assert ls.symbol_cap_block(NVDA, cash=600_000) is None


def test_disarmed_never_blocks(monkeypatch):
    """With sizing off the old gate in _pre_trade_check is back in force."""
    monkeypatch.setattr(ls, "MULTI_CONTRACT_SIZING", False)
    monkeypatch.setattr(ls, "_symbol_exposure", lambda sym: 21_500.0)
    assert ls.symbol_cap_block(NVDA, cash=45_838) is None


def test_only_short_puts_are_gated(monkeypatch):
    monkeypatch.setattr(ls, "MULTI_CONTRACT_SIZING", True)
    monkeypatch.setattr(ls, "_symbol_exposure", lambda sym: 21_500.0)
    for sig in ("SELL_CALL", "BUY_CALL", "BUY", "SELL", "BUY_TO_CLOSE"):
        s = dict(NVDA, signal=sig)
        assert ls.symbol_cap_block(s, cash=45_838) is None, sig


def test_it_never_raises(monkeypatch):
    """A gate that can throw would stop the scan. Fail open, like annotate."""
    monkeypatch.setattr(ls, "MULTI_CONTRACT_SIZING", True)
    def _boom(sym):
        raise RuntimeError("ledger unreadable")
    monkeypatch.setattr(ls, "_symbol_exposure", _boom)
    assert ls.symbol_cap_block(NVDA, cash=45_838) is None


def test_the_gate_runs_before_the_llm_in_the_scan_loop():
    """
    Source-level ordering check. The point of the gate is to skip the LLM call
    and the two account fetches, so it has to sit above them.
    """
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "live_scanner.py")).read()
    body = src.split("budget_ok, budget_msg = _budget_check", 1)[1]
    gate = body.index("symbol_cap_block(")
    llm  = body.index("_close_signals")
    assert gate < llm, "cap gate must precede the approval path"
