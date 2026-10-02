"""Tests for real_overseer.available_funds — settled-cash gating.

Context (2026-08-18): user initiated a $20K ACH bank->Schwab deposit. Schwab
grants provisional buying power instantly — `availableFunds` jumps immediately
and includes the still-pending amount (`pendingDeposits`). The overseer must
trade only SETTLED cash, so available_funds() subtracts pendingDeposits. When
the ACH lands, pendingDeposits -> 0 and the funds become usable automatically.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from real_overseer import available_funds

import pytest


@pytest.fixture(autouse=True)
def _no_ambient_unsettled_flag(monkeypatch):
    """Clear ALLOW_UNSETTLED_CASH before every test in this module.

    The flag lives in .env, which real_overseer loads on import, so without
    this the developer's own configuration decides what the tests assert —
    turning it on silently broke five tests that verify the DEFAULT settled-cash
    behaviour. A test must describe the code, not the machine it runs on.
    Tests that want the flag set it themselves with monkeypatch.setenv.
    """
    monkeypatch.delenv("ALLOW_UNSETTLED_CASH", raising=False)




def test_excludes_pending_deposit():
    # Real 2026-08-18 snapshot: availableFunds already includes the pending $20K.
    bal = {"availableFundsNonMarginableTrade": 0.0,
           "availableFunds": 29697.44, "pendingDeposits": 20000.0}
    # Only settled cash is usable → 29697.44 - 20000 = 9697.44
    assert abs(available_funds(bal) - 9697.44) < 0.01


def test_no_pending_returns_full():
    # After the ACH settles, pendingDeposits is 0 → full amount usable.
    bal = {"availableFundsNonMarginableTrade": 0.0,
           "availableFunds": 29697.44, "pendingDeposits": 0.0}
    assert abs(available_funds(bal) - 29697.44) < 0.01


def test_missing_pending_key_treated_as_zero():
    bal = {"availableFunds": 15000.0}
    assert abs(available_funds(bal) - 15000.0) < 0.01


def test_no_funds_keys_returns_none():
    assert available_funds({"pendingDeposits": 5000.0}) is None


def test_settled_matches_schwab_nonmarginable_bp():
    # Sanity: our computed settled figure equals Schwab's own
    # buyingPowerNonMarginableTrade in the same snapshot ($9,697.44).
    bal = {"availableFunds": 29697.44, "pendingDeposits": 20000.0,
           "buyingPowerNonMarginableTrade": 9697.44}
    assert abs(available_funds(bal) - bal["buyingPowerNonMarginableTrade"]) < 0.01


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


class TestConsumePosition:
    """Quantity-aware matching so stacked identical contracts reconcile right."""

    def test_stacked_partial_close(self):
        from real_overseer import consume_position
        # two ledger opens of IBM 225P, but Schwab shows qty 1
        pos = {"IBM   260925P00225000".replace(" ", ""): ["IBM   260925P00225000", 1]}
        assert consume_position(pos, "IBM", "P", 225.0) == "IBM   260925P00225000"
        assert consume_position(pos, "IBM", "P", 225.0) is None   # 2nd → closed

    def test_qty_two_both_match_then_exhaust(self):
        from real_overseer import consume_position
        pos = {"IBM260925P00225000": ["IBM   260925P00225000", 2]}
        assert consume_position(pos, "IBM", "P", 225.0) is not None
        assert consume_position(pos, "IBM", "P", 225.0) is not None
        assert consume_position(pos, "IBM", "P", 225.0) is None   # exhausted

    def test_no_cross_match(self):
        from real_overseer import consume_position
        pos = {"IBM260925P00225000": ["IBM   260925P00225000", 1]}
        assert consume_position(pos, "IBM", "P", 220.0) is None    # wrong strike
        assert consume_position(pos, "AAPL", "P", 225.0) is None   # wrong symbol
        assert consume_position(pos, "IBM", "C", 225.0) is None    # wrong type


class TestFindOrderStatus:
    def test_status_filter_finds_the_filled_close(self):
        from real_overseer import find_order
        occ = "IBM   260925P00225000"
        orders = [
            {"status": "WORKING", "orderId": "1",
             "orderLegCollection": [{"instruction": "BUY_TO_CLOSE",
                                     "instrument": {"symbol": occ}}]},
            {"status": "FILLED", "orderId": "2", "price": 2.5,
             "orderLegCollection": [{"instruction": "BUY_TO_CLOSE",
                                     "instrument": {"symbol": occ}}]},
        ]
        n = occ.replace(" ", "")
        # no status → grabs the first (WORKING) — the old bug that missed closes
        assert find_order(orders, occ_norm=n, instruction="BUY_TO_CLOSE")["orderId"] == "1"
        # status="FILLED" → finds the real close
        assert find_order(orders, occ_norm=n, instruction="BUY_TO_CLOSE",
                          status="FILLED")["orderId"] == "2"


# --- ALLOW_UNSETTLED_CASH override (added 2026-10-02) ----------------------
#
# The $20K ACH of 2026-09-24 sat pending for three-plus business days while the
# book earned nothing (free collateral $3,308, 11,418 budget blocks, zero
# closes in ten days). ACH returns almost always arrive inside 2-5 business
# days, so by then the reversal risk was largely spent and the user chose to
# deploy it. The guard stays in place and off by default; this flag makes the
# exception explicit, visible in .env, and revertible without a code change.

def _bal(pending=20000.0):
    return {"availableFundsNonMarginableTrade": 0.0,
            "availableFunds": 29697.44, "pendingDeposits": pending}


def test_flag_absent_still_excludes_pending(monkeypatch):
    """Default must not change: settled cash only."""
    monkeypatch.delenv("ALLOW_UNSETTLED_CASH", raising=False)
    assert abs(available_funds(_bal()) - 9697.44) < 0.01


def test_flag_true_counts_the_pending_deposit(monkeypatch):
    monkeypatch.setenv("ALLOW_UNSETTLED_CASH", "true")
    assert abs(available_funds(_bal()) - 29697.44) < 0.01


def test_flag_is_case_insensitive(monkeypatch):
    for v in ("TRUE", "True", "yes", "1"):
        monkeypatch.setenv("ALLOW_UNSETTLED_CASH", v)
        assert abs(available_funds(_bal()) - 29697.44) < 0.01, v


def test_anything_else_is_treated_as_off(monkeypatch):
    """A typo must fail CLOSED, not silently unlock unsettled cash."""
    for v in ("false", "no", "0", "", "maybe", "ture"):
        monkeypatch.setenv("ALLOW_UNSETTLED_CASH", v)
        assert abs(available_funds(_bal()) - 9697.44) < 0.01, v


def test_flag_is_a_no_op_once_the_deposit_settles(monkeypatch):
    """When the ACH lands pendingDeposits -> 0, so the flag stops mattering and
    can be left on harmlessly — though it should still be turned off."""
    monkeypatch.setenv("ALLOW_UNSETTLED_CASH", "true")
    on = available_funds(_bal(pending=0.0))
    monkeypatch.delenv("ALLOW_UNSETTLED_CASH", raising=False)
    off = available_funds(_bal(pending=0.0))
    assert on == off == 29697.44


def test_flag_cannot_conjure_funds_that_are_not_there(monkeypatch):
    """It only stops the subtraction; it never invents buying power."""
    monkeypatch.setenv("ALLOW_UNSETTLED_CASH", "true")
    assert available_funds({"pendingDeposits": 5000.0}) is None
