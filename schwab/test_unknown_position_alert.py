"""An unrecognised Schwab position must alert ONCE, not once per scan.

Vault 76 bought 15x INTC 280121C00090000 by hand on 2026-10-13. The overseer's
ledger knows nothing about it, so _reconcile's "Schwab has option X not in
local ledger" branch fires on every cycle. Reconcile runs each scan, ~78 times
a trading day, and each one @-mentions the user in Slack — so a single
deliberate manual trade would page the user 78 times a day for as long as it
is held.

The alert still has to fire: a genuinely unexpected position is exactly what it
exists to catch. It just has to fire once per process, per contract.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import real_overseer as ro  # noqa: E402


def setup_function():
    ro._reset_unknown_position_alerts()


def test_first_sighting_alerts():
    assert ro._should_alert_unknown_position("INTC  280121C00090000") is True


def test_second_sighting_of_the_same_contract_is_silent():
    occ = "INTC  280121C00090000"
    assert ro._should_alert_unknown_position(occ) is True
    assert ro._should_alert_unknown_position(occ) is False
    assert ro._should_alert_unknown_position(occ) is False


def test_a_different_contract_still_alerts():
    """Suppression is per contract, not a global mute."""
    assert ro._should_alert_unknown_position("INTC  280121C00090000") is True
    assert ro._should_alert_unknown_position("AMZN  261106P00245000") is True


def test_spacing_does_not_create_a_second_identity():
    """
    Schwab pads OCC roots to six characters; code elsewhere strips spaces.
    If the two spellings counted separately the alert would fire twice.
    """
    assert ro._should_alert_unknown_position("INTC  280121C00090000") is True
    assert ro._should_alert_unknown_position("INTC280121C00090000") is False


def test_reset_restores_alerting():
    """A restart is a fresh process, and the user should be told again then."""
    occ = "INTC  280121C00090000"
    assert ro._should_alert_unknown_position(occ) is True
    assert ro._should_alert_unknown_position(occ) is False
    ro._reset_unknown_position_alerts()
    assert ro._should_alert_unknown_position(occ) is True


def test_suppression_is_memory_only_not_persisted():
    """
    Deliberately not written to disk. The ledger is the place for positions
    the system knows about; a file here would quietly hide a position that
    reappears after a crash, which is the opposite of the alert's purpose.
    """
    ro._should_alert_unknown_position("INTC  280121C00090000")
    assert not os.path.exists(
        os.path.join(getattr(ro, "_DATA_DIR", "."), "unknown_positions.json"))
