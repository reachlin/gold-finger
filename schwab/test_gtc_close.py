"""
Tests for GTC buy-to-close behaviour in real_overseer.

Run (production-style, from repo root so the schwab-py library resolves
ahead of the local schwab/ package dir):

    /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/test_gtc_close.py

Covers:
  1. _submit_close_order places a GOOD_TILL_CANCEL (not DAY) BUY_TO_CLOSE,
     tracks it in pending with duration="GTC", and announces it as GTC.
  2. It is idempotent — a second call for the same contract places nothing.
  3. _process_pending does NOT day-expire a resting GTC close order after
     the 4am ET cutoff (the DAY-order eviction must not touch GTC orders).
"""
import os
import sys
import json
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pre-import the real schwab-py library so the local schwab/ dir can't shadow
# it once real_overseer inserts the repo root onto sys.path.
import schwab.orders.options  # noqa: F401

import real_overseer as ro


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeResp:
    def __init__(self, order_id="12345"):
        self.headers = {"Location": f"https://api/accounts/abc/orders/{order_id}"}

    def raise_for_status(self):
        pass


class FakeClient:
    def __init__(self):
        self.placed = []          # list of built order dicts

    def place_order(self, account_hash, order):
        self.placed.append(order)
        return FakeResp(str(1000 + len(self.placed)))


class FakeScanner:
    def __init__(self):
        self.slack = []

    def _send_slack(self, msg):
        self.slack.append(msg)


def _fresh_state(tmp):
    """Point the module's pending + counter files at a temp dir."""
    ro._DATA_DIR = tmp
    ro.PENDING_ORDERS_PATH = os.path.join(tmp, "pending_orders.json")
    ro._TRADE_COUNTER_PATH = os.path.join(tmp, "trade_counter.json")


def _bare_overseer():
    return ro.RealOverseer.__new__(ro.RealOverseer)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_places_gtc_close():
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_state(tmp)
        ov, client, scanner = _bare_overseer(), FakeClient(), FakeScanner()

        tid = ov._submit_close_order(
            scanner, client, "acct",
            symbol="NVDA", strike=200.0, expiry="2026-09-04", days_left=29,
            occ_sym="NVDA  260904P00200000", entry_prem=6.07,
            target_price=2.65, mark=3.95, opening_signal="SELL_PUT",
            opening_ref="T0018", trigger="open",
        )

        assert tid == "T0001", tid
        assert len(client.placed) == 1, "exactly one order placed"
        order = client.placed[0]
        assert order["duration"] == "GOOD_TILL_CANCEL", order["duration"]
        assert order["session"] == "NORMAL", order["session"]
        leg = order["orderLegCollection"][0]
        assert leg["instruction"] == "BUY_TO_CLOSE", leg["instruction"]

        pend = ro._load_pending()
        assert len(pend) == 1, pend
        e = pend[0]
        assert e["signal"] == "BUY_TO_CLOSE", e
        assert e["duration"] == "GTC", e
        assert e["occ_sym"] == "NVDA  260904P00200000", e
        assert e["limit"] == 2.65, e

        assert any("GTC" in m for m in scanner.slack), scanner.slack
    print("ok  test_places_gtc_close")


def test_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_state(tmp)
        ov, client, scanner = _bare_overseer(), FakeClient(), FakeScanner()
        kw = dict(
            symbol="NVDA", strike=200.0, expiry="2026-09-04", days_left=29,
            occ_sym="NVDA  260904P00200000", entry_prem=6.07,
            target_price=2.65, mark=3.95, opening_signal="SELL_PUT",
            opening_ref="T0018", trigger="open",
        )
        tid1 = ov._submit_close_order(scanner, client, "acct", **kw)
        tid2 = ov._submit_close_order(scanner, client, "acct", **kw)

        assert tid1 == "T0001", tid1
        assert tid2 is None, "second call must be a no-op"
        assert len(client.placed) == 1, "no duplicate order at the broker"
        assert len(ro._load_pending()) == 1, "no duplicate pending entry"
    print("ok  test_idempotent")


def test_gtc_not_day_expired():
    """A GTC close placed yesterday must survive today's 4am cutoff."""
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_state(tmp)
        ov, scanner = _bare_overseer(), FakeScanner()

        yesterday = (ro._now_et() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        # One GTC close (should survive) + one DAY open from yesterday
        # (should be day-expired and dropped).
        ro._save_pending([
            {"trade_id": "T0018", "symbol": "NVDA", "signal": "BUY_TO_CLOSE",
             "strike": 200.0, "expiry": "2026-09-04", "dte": 29, "limit": 2.65,
             "occ_sym": "NVDA  260904P00200000", "schwab_order_id": "555",
             "notified_at": yesterday, "duration": "GTC"},
            {"trade_id": "T0019", "symbol": "KO", "signal": "SELL_PUT",
             "strike": 83.0, "expiry": "2026-08-28", "dte": 22, "limit": 0.44,
             "occ_sym": "KO  260828P00083000", "schwab_order_id": "556",
             "notified_at": yesterday, "duration": "DAY"},
        ])

        # No matching orders returned by Schwab -> neither is FILLED/DEAD, so
        # both reach the cutoff logic. GTC stays; DAY gets evicted.
        ov._process_pending(scanner, client=FakeClient(),
                            account_hash="acct", orders=[])

        pend = ro._load_pending()
        ids = {e["trade_id"] for e in pend}
        assert ids == {"T0018"}, f"only the GTC close should remain, got {ids}"
    print("ok  test_gtc_not_day_expired")


if __name__ == "__main__":
    test_places_gtc_close()
    test_idempotent()
    test_gtc_not_day_expired()
    print("\nALL PASS")
