"""A rejected GTC cover must not be re-placed forever.

The storm this prevents, from the live order history on 2026-09-04: the same
IBM 260925P00225000 BUY_TO_CLOSE at $1.44 was sent 45 times between 16:04 and
19:56 UTC, each one rejected with "This order may result in an
oversold/overbought position in your account." Each rejection also fired its own
Slack alert.

The loop had two halves, each correct alone:
  1. place_gtc_close is idempotent against data/pending_orders.json -- it skips
     if a BUY_TO_CLOSE for this opening_ref is already pending.
  2. _reconcile drops any order in DEAD_STATUSES from pending, since a rejected
     order will never fill.
Together: place -> Schwab rejects asynchronously at its risk check -> reconcile
prunes the pending entry -> next cycle sees no cover -> place again, every ~5
minutes until something changes.

So rejections need their own memory, outside pending_orders.json.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import real_overseer as ro  # noqa: E402


def _fresh_state(tmp):
    ro._DATA_DIR = tmp
    ro.PENDING_ORDERS_PATH = os.path.join(tmp, "pending_orders.json")
    ro._TRADE_COUNTER_PATH = os.path.join(tmp, "trade_counter.json")
    ro._COVER_REJECTS_PATH = os.path.join(tmp, "cover_rejects.json")


# --- the key ---------------------------------------------------------------

def test_key_prefers_opening_ref_so_stacked_opens_track_separately():
    a = ro._cover_key("T0101", "IBM   260925P00225000")
    b = ro._cover_key("T0102", "IBM   260925P00225000")
    assert a != b, "two opens of the same contract must not share a counter"


def test_key_falls_back_to_the_contract_ignoring_occ_padding():
    assert (ro._cover_key(None, "IBM   260925P00225000")
            == ro._cover_key(None, "IBM260925P00225000"))


# --- counting --------------------------------------------------------------

def test_rejections_accumulate_and_then_block():
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_state(tmp)
        k = ro._cover_key("T0101", "IBM   260925P00225000")
        assert not ro._cover_is_blocked(k)
        for i in range(1, ro.MAX_COVER_ATTEMPTS):
            ro._record_cover_reject(k, "oversold")
            assert not ro._cover_is_blocked(k), \
                f"must still retry after {i} rejection(s)"
        ro._record_cover_reject(k, "oversold")
        assert ro._cover_is_blocked(k), "must stop at MAX_COVER_ATTEMPTS"


def test_blocking_is_per_contract_not_global():
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_state(tmp)
        bad = ro._cover_key("T0101", "IBM   260925P00225000")
        ok = ro._cover_key("T0102", "AMZN  261023P00245000")
        for _ in range(ro.MAX_COVER_ATTEMPTS):
            ro._record_cover_reject(bad, "oversold")
        assert ro._cover_is_blocked(bad)
        assert not ro._cover_is_blocked(ok), \
            "one bad contract must not freeze covers on every other position"


def test_a_successful_cover_clears_the_counter():
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_state(tmp)
        k = ro._cover_key("T0101", "IBM   260925P00225000")
        ro._record_cover_reject(k, "oversold")
        ro._clear_cover_reject(k)
        assert not ro._cover_is_blocked(k)
        # and the slate is genuinely clean, not merely under the cap
        for i in range(1, ro.MAX_COVER_ATTEMPTS):
            ro._record_cover_reject(k, "oversold")
            assert not ro._cover_is_blocked(k)


def test_state_survives_a_restart():
    """The overseer restarts often; an in-memory counter would reset the storm."""
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_state(tmp)
        k = ro._cover_key("T0101", "IBM   260925P00225000")
        for _ in range(ro.MAX_COVER_ATTEMPTS):
            ro._record_cover_reject(k, "oversold")
        assert ro._cover_is_blocked(k)
        _fresh_state(tmp)          # same files, fresh read
        assert ro._cover_is_blocked(k), "must persist across process restarts"


def test_missing_or_corrupt_state_file_never_blocks_a_cover():
    """Failing open matters: a cover protects real money. A damaged counter file
    must not be able to stop covers being placed."""
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_state(tmp)
        k = ro._cover_key("T0101", "IBM   260925P00225000")
        assert not ro._cover_is_blocked(k)          # file absent
        with open(ro._COVER_REJECTS_PATH, "w") as f:
            f.write("{not json at all")
        assert not ro._cover_is_blocked(k)          # file garbage
        ro._record_cover_reject(k, "oversold")      # and still writable after


def test_recording_reports_the_running_count():
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_state(tmp)
        k = ro._cover_key("T0101", "IBM   260925P00225000")
        assert ro._record_cover_reject(k, "oversold") == 1
        assert ro._record_cover_reject(k, "oversold") == 2


def test_reason_is_retained_for_the_alert():
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_state(tmp)
        k = ro._cover_key("T0101", "IBM   260925P00225000")
        ro._record_cover_reject(k, "oversold/overbought position")
        rec = ro._load_cover_rejects()[k]
        assert "oversold" in rec["reason"]
        assert rec["last"], "needs a timestamp so a human can see when it began"


# --- wiring: the two halves of the loop ------------------------------------

class _FakeScanner:
    def __init__(self):
        self.slack = []

    def _send_slack(self, msg):
        self.slack.append(msg)


class _Resp:
    headers = {"Location": "https://api/orders/12345"}


def test_submit_close_order_stops_after_the_cap(monkeypatch):
    """The first half: once blocked, no further order reaches Schwab."""
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_state(tmp)
        ov = ro.RealOverseer.__new__(ro.RealOverseer)
        sc = _FakeScanner()
        sent = []
        monkeypatch.setattr(ro, "place_order_with_retry",
                            lambda *a, **k: (sent.append(1), _Resp())[1])

        k = ro._cover_key("T0101", "IBM   260925P00225000")
        for _ in range(ro.MAX_COVER_ATTEMPTS):
            ro._record_cover_reject(k, "oversold")

        out = ov._submit_close_order(
            sc, None, "h", symbol="IBM", strike=225.0, expiry="2026-09-25",
            days_left=19, occ_sym="IBM   260925P00225000", entry_prem=2.00,
            target_price=1.44, opening_ref="T0101")
        assert out is None, "a blocked cover must not be placed"
        assert sent == [], "no order should reach Schwab once blocked"
        # Note this path returns BEFORE _submit_close_order imports the
        # schwab-py order builders, which is why this file runs under pytest
        # while test_gtc_close.py (which needs them) must run as a script.


def test_a_rejection_is_recorded_so_the_next_cycle_sees_it():
    """The second half: reconcile must remember the rejection it prunes."""
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_state(tmp)
        entry = {"signal": "BUY_TO_CLOSE", "symbol": "IBM", "strike": 225.0,
                 "occ_sym": "IBM   260925P00225000", "opening_ref": "T0101",
                 "trade_id": "T0200"}
        k = ro._cover_key("T0101", "IBM   260925P00225000")
        for i in range(ro.MAX_COVER_ATTEMPTS):
            ro._note_dead_order(entry, "REJECTED", "oversold/overbought")
        assert ro._cover_is_blocked(k)


def test_a_cancelled_order_is_not_counted_as_a_rejection():
    """Cancelling a GTC cover by hand is not a systematic failure."""
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_state(tmp)
        entry = {"signal": "BUY_TO_CLOSE", "occ_sym": "IBM   260925P00225000",
                 "opening_ref": "T0101"}
        for _ in range(ro.MAX_COVER_ATTEMPTS + 2):
            ro._note_dead_order(entry, "CANCELED", "cancelled by user")
        assert not ro._cover_is_blocked(ro._cover_key("T0101", ""))


def test_a_rejected_open_order_is_not_counted():
    """Only covers are tracked; a rejected SELL_PUT is a different problem."""
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_state(tmp)
        entry = {"signal": "SELL_PUT", "occ_sym": "IBM   260925P00225000",
                 "opening_ref": "T0101"}
        for _ in range(ro.MAX_COVER_ATTEMPTS + 2):
            ro._note_dead_order(entry, "REJECTED", "no buying power")
        assert not ro._cover_is_blocked(ro._cover_key("T0101", ""))
