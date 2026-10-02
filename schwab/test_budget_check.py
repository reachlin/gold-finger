"""Tests for live_scanner._budget_check — unsettled-ACH fencing.

Regression guard: the budget message used to show cashBalance (ungated), so it
reported $51,391 free while the real _pre_trade_check gate only allowed $31,391
(it subtracts pendingDeposits). Now _budget_check subtracts unsettled ACH too,
so the message and the gate agree.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import live_scanner as ls

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




def _setup(committed=0.0, order_pending=0.0, ach=None):
    ls._committed_collateral = lambda: committed
    ls._pending_collateral   = lambda: order_pending
    ls._collateral_required  = lambda s: s["strike"] * 100
    ls._schwab_pending_deposits = ach


def test_unsettled_ach_is_fenced_from_budget():
    _setup(ach=20000.0)
    # $520 strike = $52k needed; only $31,391 truly free ($51,391 - $20k ACH)
    ok, msg = ls._budget_check({"strike": 520.0}, 51391.0)
    assert ok is False
    assert "31,391 free" in msg and "20,000 unsettled-ACH" in msg


def test_trade_under_gated_limit_allowed():
    _setup(ach=20000.0)
    # $300 strike = $30k <= $31,391 gated → allowed
    ok, msg = ls._budget_check({"strike": 300.0}, 51391.0)
    assert ok is True and "held back" in msg


def test_no_ach_unchanged():
    _setup(ach=None)
    ok, msg = ls._budget_check({"strike": 300.0}, 51391.0)
    assert ok is True and "unsettled-ACH" not in msg


def test_ach_pushes_a_borderline_trade_over():
    _setup(ach=20000.0)
    # $400 strike = $40k: fits ungated $51,391, but the ACH gate ($31,391) blocks it.
    ok, msg = ls._budget_check({"strike": 400.0}, 51391.0)
    assert ok is False and "unsettled-ACH" in msg
    # same trade with no ACH fence → allowed
    _setup(ach=None)
    ok2, _ = ls._budget_check({"strike": 400.0}, 51391.0)
    assert ok2 is True


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


# --- ALLOW_UNSETTLED_CASH (added 2026-10-02) -------------------------------
#
# The flag must be honoured HERE too, not just in real_overseer.available_funds.
# This is the gate that actually fires: it produced 11,418 BUDGET BLOCK lines
# and short-circuits before _pre_trade_check ever runs. Setting the flag in only
# one of the two places would look like a working change and deploy nothing.

def test_flag_lets_the_scanner_use_unsettled_ach(monkeypatch):
    monkeypatch.setenv("ALLOW_UNSETTLED_CASH", "true")
    _setup(ach=20000.0)
    # $400 strike = $40k. Blocked at the gated $31,391, allowed at the full $51,391.
    ok, msg = ls._budget_check({"strike": 400.0}, 51391.0)
    assert ok is True, msg


def test_default_still_fences_the_ach(monkeypatch):
    monkeypatch.delenv("ALLOW_UNSETTLED_CASH", raising=False)
    _setup(ach=20000.0)
    ok, _ = ls._budget_check({"strike": 400.0}, 51391.0)
    assert ok is False


def test_a_typo_fails_closed(monkeypatch):
    monkeypatch.setenv("ALLOW_UNSETTLED_CASH", "ture")
    _setup(ach=20000.0)
    ok, _ = ls._budget_check({"strike": 400.0}, 51391.0)
    assert ok is False


def test_the_message_says_the_ach_is_being_counted(monkeypatch):
    """Whoever reads the log must see that unsettled money is in play."""
    monkeypatch.setenv("ALLOW_UNSETTLED_CASH", "true")
    _setup(ach=20000.0)
    ok, msg = ls._budget_check({"strike": 300.0}, 51391.0)
    assert ok is True
    assert "unsettled" in msg.lower()
    assert "held back" not in msg, "it is no longer being held back"


def test_committed_and_order_pending_are_still_subtracted(monkeypatch):
    """The flag releases ONLY the ACH fence. Collateral already locked by open
    and in-flight orders must still reduce what is spendable."""
    monkeypatch.setenv("ALLOW_UNSETTLED_CASH", "true")
    _setup(committed=40000.0, order_pending=5000.0, ach=20000.0)
    # 51,391 - 40,000 - 5,000 = 6,391 spendable; a $10k strike must not pass.
    ok, _ = ls._budget_check({"strike": 100.0}, 51391.0)
    assert ok is False
