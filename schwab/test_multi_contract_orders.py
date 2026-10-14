"""Wiring sizing into the order paths — and the cover that must match it.

The dangerous half of multi-contract sizing is not the sell order, it is the
cover. Four places carried a literal `1`, and one comparison was worse than a
wrong quantity:

    short_qty[n]        += 1                                 # per ledger ROW
    schwab_cover_qty[n] += int(leg.get("quantity", 1) or 1)   # real quantity
    ...
    if schwab_cover_qty.get(occ_norm, 0) >= short_qty.get(occ_norm, 0):
        continue    # "broker already shows enough covers for this contract"

A 3-contract short with a 1-contract cover reads 1 >= 1, so the overseer
concludes the position is fully covered and never places another. That is not
a mis-sized cover, it is two permanently naked short puts that look protected
in the log and in overseer_status.py. It breaks the invariant the book rests
on: every open short ALWAYS carries a resting GTC buy-to-close.

So the cover's quantity is never "sized" — it is read from the position being
covered. Sizing decides how much to sell; the cover copies it.

This ships with MULTI_CONTRACT_SIZING = False, so every assertion about the
flag being off describes live behaviour after this commit: quantity 1, the
one-position-per-underlying gate intact, the prompt unchanged. The flag-on
tests prove the wiring is right so arming it later is a one-line change, not
a leap.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import live_scanner as ls  # noqa: E402
import options_ledger as ol  # noqa: E402
import real_overseer as ro  # noqa: E402


# ---------------------------------------------------------------------------
# The flag ships OFF
# ---------------------------------------------------------------------------

def test_multi_contract_sizing_is_off_by_default():
    """
    launchd runs the overseer with KeepAlive, so a crash restarts it from this
    source with no decision from anyone. Landing this armed would change live
    sizing on a crash.
    """
    assert ls.MULTI_CONTRACT_SIZING is False


def test_with_the_flag_off_quantity_is_one_whatever_the_capital():
    sig = {"signal": "SELL_PUT", "symbol": "NVDA", "strike": 215.0}
    for cash, avail in ((45_838, 24_338), (400_000, 400_000),
                        (5_000_000, 5_000_000)):
        assert ro._sized_contracts(ls, sig, cash, 0, avail) == 1, cash


def test_with_the_flag_on_quantity_comes_from_size_put(monkeypatch):
    monkeypatch.setattr(ls, "MULTI_CONTRACT_SIZING", True)
    sig = {"signal": "SELL_PUT", "symbol": "NVDA", "strike": 215.0}
    assert ro._sized_contracts(ls, sig, 400_000, 0, 400_000) == 3
    assert ro._sized_contracts(ls, sig, 45_838, 0, 45_838) == 1
    assert ro._sized_contracts(ls, sig, 400_000, 64_500, 400_000) == 0


def test_a_refusal_is_zero_not_one(monkeypatch):
    """0 must mean "do not place", never silently fall back to one contract."""
    monkeypatch.setattr(ls, "MULTI_CONTRACT_SIZING", True)
    sig = {"signal": "SELL_PUT", "symbol": "NVDA", "strike": 215.0}
    assert ro._sized_contracts(ls, sig, 400_000, 80_000, 400_000) == 0


# ---------------------------------------------------------------------------
# short_qty must count CONTRACTS, not ledger rows
# ---------------------------------------------------------------------------

OCC = "NVDA  261113P00215000"


def _open_row(symbol="NVDA", strike="215", date="2026-10-13 10:43:30", **kw):
    row = {"date": date, "symbol": symbol, "signal": "SELL_PUT",
           "strike": strike, "verdict": "APPROVED"}
    row.update(kw)
    return row


def test_short_quantities_counts_contracts():
    row = _open_row(contracts=3)
    occ_map = {ol.row_key(row): OCC}
    assert ro._short_quantities([row], occ_map) == {OCC.replace(" ", ""): 3}


def test_short_quantities_treats_a_legacy_row_as_one():
    row = _open_row()
    occ_map = {ol.row_key(row): OCC}
    assert ro._short_quantities([row], occ_map) == {OCC.replace(" ", ""): 1}


def test_short_quantities_sums_separate_opens_on_one_contract():
    """Two tranches of the same strike and expiry are one OCC symbol."""
    a = _open_row(date="2026-10-13 10:00:00", contracts=2)
    b = _open_row(date="2026-10-14 10:00:00", contracts=3)
    occ_map = {ol.row_key(a): OCC, ol.row_key(b): OCC}
    assert ro._short_quantities([a, b], occ_map) == {OCC.replace(" ", ""): 5}


def test_short_quantities_skips_opens_not_confirmed_on_schwab():
    row = _open_row(contracts=3)
    assert ro._short_quantities([row], {}) == {}


def test_cover_quantities_reads_the_leg_quantity():
    orders = [{"status": "WORKING", "orderLegCollection": [
        {"instruction": "BUY_TO_CLOSE", "quantity": 3,
         "instrument": {"symbol": OCC}}]}]
    assert ro._cover_quantities(orders) == {OCC.replace(" ", ""): 3}


def test_cover_quantities_ignores_non_working_orders():
    orders = [{"status": "CANCELED", "orderLegCollection": [
        {"instruction": "BUY_TO_CLOSE", "quantity": 3,
         "instrument": {"symbol": OCC}}]}]
    assert ro._cover_quantities(orders) == {}


def test_a_one_contract_cover_does_not_satisfy_a_three_contract_short():
    """
    The bug this commit exists to prevent. Before: short_qty 1 vs cover 1 →
    "enough covers" → two naked shorts, silently.
    """
    row = _open_row(contracts=3)
    occ_map = {ol.row_key(row): OCC}
    shorts = ro._short_quantities([row], occ_map)
    covers = ro._cover_quantities([{"status": "WORKING", "orderLegCollection": [
        {"instruction": "BUY_TO_CLOSE", "quantity": 1,
         "instrument": {"symbol": OCC}}]}])
    key = OCC.replace(" ", "")
    assert covers.get(key, 0) < shorts.get(key, 0), \
        "a partial cover must never read as sufficient"


def test_a_matching_cover_does_satisfy_the_short():
    row = _open_row(contracts=3)
    occ_map = {ol.row_key(row): OCC}
    shorts = ro._short_quantities([row], occ_map)
    covers = ro._cover_quantities([{"status": "WORKING", "orderLegCollection": [
        {"instruction": "BUY_TO_CLOSE", "quantity": 3,
         "instrument": {"symbol": OCC}}]}])
    key = OCC.replace(" ", "")
    assert covers.get(key, 0) >= shorts.get(key, 0)


# ---------------------------------------------------------------------------
# The cover order carries the short's quantity
# ---------------------------------------------------------------------------

def test_cover_order_is_built_with_the_requested_contract_count():
    """Build the order the way _submit_close_order does and read it back."""
    from schwab.orders.options import option_buy_to_close_limit
    from schwab.orders.common import Duration, Session
    order = (option_buy_to_close_limit(OCC, 3, "1.49")
             .set_duration(Duration.GOOD_TILL_CANCEL)
             .set_session(Session.NORMAL)
             .build())
    legs = order["orderLegCollection"]
    assert len(legs) == 1
    assert legs[0]["quantity"] == 3
    assert legs[0]["instruction"] == "BUY_TO_CLOSE"


def test_cover_contracts_helper_reads_the_opening_row():
    assert ro._cover_contracts(_open_row(contracts=4)) == 4
    assert ro._cover_contracts(_open_row()) == 1
    assert ro._cover_contracts({}) == 1
    assert ro._cover_contracts(None) == 1


# ---------------------------------------------------------------------------
# one-position-per-underlying: intact with the flag off, relaxed with it on
# ---------------------------------------------------------------------------

def _check(positions, signal, monkeypatch, avail=400_000.0, orders=None):
    monkeypatch.setattr(ro, "fetch_account", lambda c, h: {
        "currentBalances": {"availableFunds": avail, "pendingDeposits": 0},
        "positions": positions,
    })
    ov = ro.RealOverseer.__new__(ro.RealOverseer)
    return ro.RealOverseer._pre_trade_check(ov, None, "hash", signal)


def _short_pos(occ="NVDA  261113P00215000", qty=1.0):
    return {"instrument": {"assetType": "OPTION", "symbol": occ},
            "shortQuantity": qty}


SIG = {"symbol": "NVDA", "strike": 215.0, "signal": "SELL_PUT"}


def test_flag_off_keeps_the_one_position_per_underlying_gate(monkeypatch):
    """
    Added 2026-09-29 after the book went 99% AMZN, in code rather than the
    prompt on purpose. With the flag off it must still be there.
    """
    ok, why = _check([_short_pos()], SIG, monkeypatch)
    assert ok is False
    assert "one open position per underlying" in why


def test_flag_on_allows_a_second_position_on_the_same_underlying(monkeypatch):
    """
    The cap replaces the blanket gate: exposure, not presence, decides.
    """
    monkeypatch.setattr(ls, "MULTI_CONTRACT_SIZING", True)
    ok, why = _check([_short_pos()], SIG, monkeypatch)
    assert ok is True, why


def test_flag_on_still_blocks_a_different_underlyings_lookalike(monkeypatch):
    """
    The OCC root matcher must stay exact. `"V" in "AVGO  261023C00200000"` is
    True, which once blocked a Visa signal behind a Broadcom position.
    """
    monkeypatch.setattr(ls, "MULTI_CONTRACT_SIZING", True)
    ok, why = _check([_short_pos("AVGO  261023C00200000")],
                     {"symbol": "V", "strike": 300.0, "signal": "SELL_PUT"},
                     monkeypatch)
    assert ok is True, why


def test_a_resting_order_still_blocks_even_with_the_flag_on(monkeypatch):
    """
    Two in-flight orders on one symbol is a race the cap cannot see: neither
    has filled, so neither is exposure yet. One in-flight order per underlying
    stays the rule whatever the flag says.
    """
    monkeypatch.setattr(ls, "MULTI_CONTRACT_SIZING", True)
    monkeypatch.setattr(ro, "fetch_account", lambda c, h: {
        "currentBalances": {"availableFunds": 400_000.0, "pendingDeposits": 0},
        "positions": [],
    })
    monkeypatch.setattr(ro, "_load_pending", lambda: [
        {"signal": "SELL_PUT", "symbol": "NVDA", "strike": 215.0}])
    ov = ro.RealOverseer.__new__(ro.RealOverseer)
    ok, why = ro.RealOverseer._pre_trade_check(ov, None, "hash", SIG)
    assert ok is False
    assert "resting" in why.lower()


# ---------------------------------------------------------------------------
# The prompt must agree with whichever rule is in force
# ---------------------------------------------------------------------------

def test_the_prompts_symbol_rule_follows_the_flag(monkeypatch):
    """
    The model is told "one open short per underlying is the rule". With the cap
    armed that is no longer true, and a stale instruction makes the model veto
    top-ups the code now permits — the prompt would be fighting the allocator.
    """
    off = ro._symbol_rule_text(ls)
    monkeypatch.setattr(ls, "MULTI_CONTRACT_SIZING", True)
    on = ro._symbol_rule_text(ls)
    assert "one open short per underlying" in off
    assert "one open short per underlying" not in on
    assert "20%" in on or "per-symbol cap" in on


# ---------------------------------------------------------------------------
# place_order.py CLI takes the quantity from the pending entry
# ---------------------------------------------------------------------------

def test_manual_order_path_uses_the_entrys_contract_count():
    """
    The CLI places an order the overseer already sized and recorded. It must
    not re-size: the decision was made when the entry was written.
    """
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "place_order.py")).read()
    assert "option_sell_to_open_limit(occ_sym, 1," not in src, \
        "manual path still hardcodes one contract"
    assert "contracts_of(entry)" in src or "_entry_contracts" in src


# ---------------------------------------------------------------------------
# "could not ask" must not read as "nothing resting"
# ---------------------------------------------------------------------------

def test_fetch_orders_still_returns_empty_for_ordinary_callers(capsys):
    """Reconcile and the cover loop tolerate an empty list; keep that."""
    assert ro.fetch_orders(None, "hash", days_back=5) == []
    assert "Could not fetch" in capsys.readouterr().out


def test_fetch_orders_raises_for_callers_that_must_know(capsys):
    with pytest.raises(Exception):
        ro.fetch_orders(None, "hash", days_back=5, strict=True)


def test_a_failed_order_fetch_falls_back_to_the_pending_file(monkeypatch):
    """
    The fallback was unreachable: fetch_orders swallowed every exception and
    returned [], so _pre_trade_check's except branch never ran and a failed
    fetch looked like a clean book. A 429 here would let a second order go out
    on a symbol that already has one resting.
    """
    monkeypatch.setattr(ro, "fetch_account", lambda c, h: {
        "currentBalances": {"availableFunds": 400_000.0, "pendingDeposits": 0},
        "positions": [],
    })
    monkeypatch.setattr(ro, "_load_pending", lambda: [
        {"signal": "SELL_PUT", "symbol": "NVDA", "strike": 215.0}])
    ov = ro.RealOverseer.__new__(ro.RealOverseer)
    ok, why = ro.RealOverseer._pre_trade_check(ov, None, "hash", SIG)
    assert ok is False, "a failed fetch must not read as nothing resting"
    assert "resting" in why.lower()
