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
