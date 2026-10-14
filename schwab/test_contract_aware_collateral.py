"""Collateral arithmetic must honour a contract count.

Today every order is one contract: `option_sell_to_open_limit(occ, 1, ...)` is
a literal in both order paths, and `_collateral_required` does not even look at
a `contracts` field. So `strike * 100` happens to be right everywhere.

The moment sizing exists, that stops being true, and the failure is silent in
the dangerous direction: `committed_collateral()` feeds `_budget_check()`, so a
quantity-blind collateral number would let the scanner approve a put the
account cannot cash-secure — defeating "no margin / no naked contracts", which
is the promise the whole book rests on.

This commit makes the three collateral functions quantity-aware while every
caller still passes one contract, so the live numbers are unchanged. The tests
below pin BOTH halves of that: the new arithmetic, and the fact that a row
without a `contracts` column still values exactly as it does today.

Deliberately NOT in this commit: P&L (`premium_ct`, assignment, the wheel) and
the order quantities themselves. P&L is reporting, not gating — it cannot
approve a trade the account cannot cover — and it only becomes wrong once
quantities can vary, which is the commit that wires sizing in.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import options_ledger as ol  # noqa: E402


# ---------------------------------------------------------------------------
# contracts_of — the single place the field is interpreted
# ---------------------------------------------------------------------------

def test_missing_contracts_means_one():
    """Every row written before this commit lacks the column entirely."""
    assert ol.contracts_of({}) == 1
    assert ol.contracts_of({"strike": "215"}) == 1


def test_empty_string_means_one():
    """read_rows backfills absent columns with "", not None."""
    assert ol.contracts_of({"contracts": ""}) == 1


def test_reads_an_explicit_count():
    assert ol.contracts_of({"contracts": 3}) == 3
    assert ol.contracts_of({"contracts": "3"}) == 3
    assert ol.contracts_of({"contracts": 3.0}) == 3
    assert ol.contracts_of({"contracts": "3.0"}) == 3


def test_garbage_falls_back_to_one_not_zero():
    """
    Fails toward OVER-counting collateral. A zero here would free collateral
    the account has actually committed, which is the direction that overdraws.
    """
    for bad in (0, -2, "", "abc", None, [], {}):
        assert ol.contracts_of({"contracts": bad}) == 1, bad


def test_a_fractional_count_rounds_down_but_never_below_one():
    """Options are not fractional; 2.7 contracts is 2, 0.4 is still 1."""
    assert ol.contracts_of({"contracts": 2.7}) == 2
    assert ol.contracts_of({"contracts": 0.4}) == 1


# ---------------------------------------------------------------------------
# committed_collateral
# ---------------------------------------------------------------------------

def _open_put(symbol="NVDA", strike="215", date="2026-10-13 10:43:30", **kw):
    row = {"date": date, "symbol": symbol, "signal": "SELL_PUT",
           "strike": strike, "verdict": "APPROVED"}
    row.update(kw)
    return row


def test_one_contract_is_strike_times_one_hundred():
    """The legacy number, unchanged."""
    assert ol.committed_collateral([_open_put()]) == 21_500.0


def test_absent_column_values_exactly_as_today():
    """A pre-migration row must not change value."""
    row = _open_put()
    assert "contracts" not in row
    assert ol.committed_collateral([row]) == 21_500.0


def test_three_contracts_locks_three_times_the_cash():
    assert ol.committed_collateral([_open_put(contracts=3)]) == 64_500.0


def test_a_closed_put_locks_nothing_regardless_of_count():
    rows = [
        _open_put(contracts=4),
        {"date": "2026-10-20 09:30:00", "symbol": "NVDA", "signal": "SELL_PUT",
         "strike": "215", "verdict": "CLOSED", "contracts": 4,
         "ref": "NVDA_2026-10-13 10:43:30"},
    ]
    assert ol.committed_collateral(rows) == 0.0


def test_an_assigned_put_still_locks_its_full_count():
    """Assignment turns cash into shares; the cash is not free."""
    rows = [
        _open_put(contracts=2),
        {"date": "2026-11-13 16:00:00", "symbol": "NVDA", "signal": "SELL_PUT",
         "strike": "215", "verdict": "ASSIGNED", "contracts": 2,
         "ref": "NVDA_2026-10-13 10:43:30"},
    ]
    assert ol.committed_collateral(rows) == 43_000.0


def test_mixed_counts_sum_independently():
    rows = [
        _open_put("NVDA", "215", "2026-10-13 10:43:30", contracts=2),
        _open_put("AMZN", "245", "2026-10-13 11:00:00"),            # legacy
        _open_put("GOOGL", "335", "2026-10-13 12:00:00", contracts=3),
    ]
    assert ol.committed_collateral(rows) == 43_000 + 24_500 + 100_500


def test_a_covered_call_locks_no_cash_at_any_count():
    rows = [{"date": "2026-10-13 10:00:00", "symbol": "INTC",
             "signal": "SELL_CALL", "strike": "90", "verdict": "APPROVED",
             "contracts": 15}]
    assert ol.committed_collateral(rows) == 0.0


# ---------------------------------------------------------------------------
# The zero-behaviour-change proof, against the real production ledger
# ---------------------------------------------------------------------------

def test_production_ledger_value_is_unchanged_by_this_commit():
    """
    The live ledger has 60 rows and no `contracts` column. Its committed
    collateral must equal the legacy formula exactly — if this drifts, the
    refactor changed live numbers and must not ship.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "paper_options_ledger.csv")
    if not os.path.exists(path):
        return                      # not checked out; nothing to prove
    rows = ol.read_rows(path)
    legacy = 0.0
    for s in ol._states(rows).values():
        r = s["row"]
        if r.get("signal") != "SELL_PUT" or s["status"] == "closed":
            continue
        try:
            legacy += float(r["strike"]) * ol.SHARES_PER_CONTRACT
        except (ValueError, KeyError):
            pass
    assert ol.committed_collateral(rows) == legacy


# ---------------------------------------------------------------------------
# CSV round-trip: the column must persist, and legacy files must still load
# ---------------------------------------------------------------------------

def test_contracts_is_a_ledger_column():
    assert "contracts" in ol.FIELDNAMES


def test_append_and_read_preserves_the_count(tmp_path):
    p = str(tmp_path / "led.csv")
    ol.append_row(p, _open_put(contracts=7))
    rows = ol.read_rows(p)
    assert len(rows) == 1
    assert ol.contracts_of(rows[0]) == 7


def test_a_legacy_file_without_the_column_still_reads_as_one(tmp_path):
    """
    Simulates the file on disk right now. read_rows backfills the column and
    contracts_of must read the blank as one contract, not zero.
    """
    p = str(tmp_path / "legacy.csv")
    legacy_cols = [c for c in ol.FIELDNAMES if c != "contracts"]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=legacy_cols)
        w.writeheader()
        w.writerow({"date": "2026-10-13 10:43:30", "symbol": "NVDA",
                    "signal": "SELL_PUT", "strike": "215",
                    "verdict": "APPROVED"})
    rows = ol.read_rows(p)
    assert ol.contracts_of(rows[0]) == 1
    assert ol.committed_collateral(rows) == 21_500.0


def test_appending_to_a_legacy_file_migrates_the_header(tmp_path):
    p = str(tmp_path / "legacy.csv")
    legacy_cols = [c for c in ol.FIELDNAMES if c != "contracts"]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=legacy_cols)
        w.writeheader()
        w.writerow({"date": "2026-10-01 10:00:00", "symbol": "KO",
                    "signal": "SELL_PUT", "strike": "86",
                    "verdict": "APPROVED"})
    ol.append_row(p, _open_put(contracts=2))
    with open(p, newline="") as f:
        header = next(csv.reader(f))
    assert header == ol.FIELDNAMES
    rows = ol.read_rows(p)
    assert ol.contracts_of(rows[0]) == 1      # the migrated legacy row
    assert ol.contracts_of(rows[1]) == 2


# ---------------------------------------------------------------------------
# live_scanner._collateral_required / _pending_collateral
# ---------------------------------------------------------------------------

def _scanner():
    import live_scanner
    return live_scanner


def test_collateral_required_is_unchanged_for_one_contract():
    ls = _scanner()
    assert ls._collateral_required({"signal": "SELL_PUT", "strike": 215}) == 21_500.0


def test_collateral_required_scales_with_contracts():
    ls = _scanner()
    assert ls._collateral_required(
        {"signal": "SELL_PUT", "strike": 215, "contracts": 4}) == 86_000.0


def test_collateral_required_scales_a_long_call_debit():
    ls = _scanner()
    assert ls._collateral_required(
        {"signal": "BUY_CALL", "cost_per_ct": 3_241, "contracts": 15}) == 48_615.0


def test_collateral_required_ignores_contracts_on_signals_that_lock_nothing():
    ls = _scanner()
    for sig in ("SELL_CALL", "BUY", "SELL", "BUY_TO_CLOSE", "NONE"):
        assert ls._collateral_required({"signal": sig, "contracts": 9}) == 0.0


def test_pending_collateral_scales_with_contracts(tmp_path, monkeypatch):
    ls = _scanner()
    p = str(tmp_path / "pending.json")
    with open(p, "w") as f:
        json.dump([
            {"signal": "SELL_PUT", "strike": 215, "contracts": 2},
            {"signal": "SELL_PUT", "strike": 245},                 # legacy
            {"signal": "BUY_CALL", "limit": 32.41, "contracts": 15},
            {"signal": "SELL_CALL", "strike": 90, "contracts": 15},
        ], f)
    monkeypatch.setattr(ls, "_PENDING_ORDERS_PATH", p)
    assert ls._pending_collateral() == 43_000 + 24_500 + 48_615


def test_pending_collateral_unchanged_when_no_counts_present(tmp_path, monkeypatch):
    ls = _scanner()
    p = str(tmp_path / "pending.json")
    with open(p, "w") as f:
        json.dump([{"signal": "SELL_PUT", "strike": 215}], f)
    monkeypatch.setattr(ls, "_PENDING_ORDERS_PATH", p)
    assert ls._pending_collateral() == 21_500.0


def test_budget_check_sees_the_full_multi_contract_requirement(monkeypatch):
    """
    The point of the whole commit: _budget_check must refuse a multi-contract
    put the cash cannot secure. With a quantity-blind _collateral_required it
    would have approved this.
    """
    ls = _scanner()
    monkeypatch.setattr(ls, "_committed_collateral", lambda: 0.0)
    monkeypatch.setattr(ls, "_pending_collateral", lambda: 0.0)
    monkeypatch.setattr(ls, "_schwab_pending_deposits", 0)
    sig = {"signal": "SELL_PUT", "symbol": "NVDA", "strike": 215, "contracts": 3}
    ok, msg = ls._budget_check(sig, 45_837.81)
    assert ok is False
    assert "64,500" in msg, msg
    ok1, _ = ls._budget_check(
        {"signal": "SELL_PUT", "symbol": "NVDA", "strike": 215}, 45_837.81)
    assert ok1 is True, "one contract must still pass, exactly as today"


# ---------------------------------------------------------------------------
# real_overseer._pre_trade_check — the LAST gate before a real order
# ---------------------------------------------------------------------------

def _check(positions, signal, monkeypatch, avail=100_000.0):
    """Run _pre_trade_check against a faked account (pattern borrowed from
    test_one_position_per_symbol.check)."""
    import real_overseer as ro
    monkeypatch.setattr(ro, "fetch_account", lambda c, h: {
        "currentBalances": {"availableFunds": avail, "pendingDeposits": 0},
        "positions": positions,
    })
    ov = ro.RealOverseer.__new__(ro.RealOverseer)
    return ro.RealOverseer._pre_trade_check(ov, None, "hash", signal)


def test_pre_trade_check_is_unchanged_for_one_contract(monkeypatch):
    ok, why = _check([], {"symbol": "NVDA", "strike": 215.0,
                          "signal": "SELL_PUT"}, monkeypatch, avail=30_000.0)
    assert ok is True, why


def test_pre_trade_check_aborts_when_cash_cannot_secure_every_contract(monkeypatch):
    """
    The broker-side gate computes its own collateral figure, independent of
    live_scanner._collateral_required. If only that one were made
    quantity-aware, a 3-contract order would still clear this gate on the cash
    for one — and this is the last check before the order is sent.
    """
    ok, why = _check([], {"symbol": "NVDA", "strike": 215.0, "contracts": 3,
                          "signal": "SELL_PUT"}, monkeypatch, avail=30_000.0)
    assert ok is False
    assert "64,500" in why, why


def test_pre_trade_check_allows_multi_contract_when_cash_covers_it(monkeypatch):
    ok, why = _check([], {"symbol": "NVDA", "strike": 215.0, "contracts": 3,
                          "signal": "SELL_PUT"}, monkeypatch, avail=70_000.0)
    assert ok is True, why
