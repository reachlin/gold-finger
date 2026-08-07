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


class _HttpErr(Exception):
    """Mimics httpx.HTTPStatusError — carries .response.status_code."""
    def __init__(self, status):
        self.response = type("R", (), {"status_code": status})()
        super().__init__(f"HTTP {status}")


class FakeResp429:
    def raise_for_status(self):
        raise _HttpErr(429)


class FlakyClient:
    """Returns 429 for the first `fail_times` placements, then succeeds."""
    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def place_order(self, account_hash, order):
        self.calls += 1
        if self.calls <= self.fail_times:
            return FakeResp429()
        return FakeResp(str(9000 + self.calls))


def test_retry_succeeds_after_429(monkeypatch_sleep=None):
    """Two 429s then success → order lands, backoff was used."""
    slept = []
    orig_sleep = ro.time.sleep
    ro.time.sleep = lambda s: slept.append(s)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _fresh_state(tmp)
            ov, scanner = _bare_overseer(), FakeScanner()
            client = FlakyClient(fail_times=2)
            tid = ov._submit_close_order(
                scanner, client, "acct",
                symbol="XOM", strike=144.0, expiry="2026-09-04", days_left=28,
                occ_sym="XOM  260904P00144000", entry_prem=2.03,
                target_price=1.02, trigger="open",
            )
            assert tid == "T0001", tid
            assert client.calls == 3, f"1 fail+1 fail+1 ok = 3 calls, got {client.calls}"
            assert len(slept) == 2, f"backoff slept twice, got {slept}"
            assert slept == [1.5, 3.0], slept        # exponential
            assert len(ro._load_pending()) == 1, "order tracked after retry"
    finally:
        ro.time.sleep = orig_sleep
    print("ok  test_retry_succeeds_after_429")


def test_retry_gives_up():
    """Persistent 429 → helper exhausts attempts, close is a clean no-op."""
    orig_sleep = ro.time.sleep
    ro.time.sleep = lambda s: None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _fresh_state(tmp)
            ov, scanner = _bare_overseer(), FakeScanner()
            client = FlakyClient(fail_times=99)
            tid = ov._submit_close_order(
                scanner, client, "acct",
                symbol="XOM", strike=144.0, expiry="2026-09-04", days_left=28,
                occ_sym="XOM  260904P00144000", entry_prem=2.03,
                target_price=1.02, trigger="open",
            )
            assert tid is None, "no trade_id when all attempts 429"
            assert client.calls == 4, f"default 4 attempts, got {client.calls}"
            assert ro._load_pending() == [], "nothing tracked on failure"
            assert any("failed" in m.lower() for m in scanner.slack), scanner.slack
    finally:
        ro.time.sleep = orig_sleep
    print("ok  test_retry_gives_up")


if __name__ == "__main__":
    test_places_gtc_close()
    test_idempotent()
    test_gtc_not_day_expired()
    test_retry_succeeds_after_429()
    test_retry_gives_up()
    print("\nALL PASS")
