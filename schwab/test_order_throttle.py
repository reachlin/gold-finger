"""
Rate-limit handling for Schwab's /orders endpoint.

What we actually observed, which is why these tests exist:

1. 2026-10-05 — an XOM open was followed ~2s later by its GTC cover; the cover
   got a 429 and was lost. A burst of our own making.
2. 2026-10-08 09:48-09:49 — an EarlyTP replace (POST /orders), then fill
   detection (GET /orders), then a placement, all inside ~40 seconds. The
   placement 429'd through all three retries and the signal was dropped. Note
   the calls were TENS of seconds apart, so plain spacing would not have saved
   it; the budget looks like a rolling count per window, not a gap rule.
3. The backoff was 1.5 + 3 + 6 = 10.5s total, which cannot outlast a
   rolling-minute window. All three retries failed every time it fired (5
   separate events in data/overseer.log) -- a burst limit would have cleared.
4. `Retry-After`, if Schwab sends it, was ignored.
5. The EarlyTP replace did not go through the retry helper at all, so one
   transient 429 lost the attempt outright.

The delays here are a judgement call -- Schwab does not document the quota and
we cannot measure it without deliberately tripping it. What is NOT judgement is
that 10.5s was demonstrably too short, so the schedule now reaches past a minute.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import schwab.orders.options  # noqa: F401  (grafted by conftest.py)

import real_overseer as ro


class Resp:
    def __init__(self, status=201, retry_after=None):
        self.status_code = status
        self.headers = {"Location": "https://api/accounts/a/orders/7"}
        if retry_after is not None:
            self.headers["Retry-After"] = str(retry_after)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise FakeHTTP(self)


class FakeHTTP(Exception):
    def __init__(self, resp):
        super().__init__(f"Client error '{resp.status_code} Too Many Requests'")
        self.response = resp


class Client:
    """Fails with 429 for the first `fail_times` calls, then succeeds."""

    def __init__(self, fail_times=0, retry_after=None):
        self.fail_times, self.retry_after = fail_times, retry_after
        self.calls = 0

    def place_order(self, account_hash, order):
        self.calls += 1
        if self.calls <= self.fail_times:
            return Resp(429, retry_after=self.retry_after)
        return Resp(201)

    def replace_order(self, account_hash, order_id, order):
        return self.place_order(account_hash, order)


def _no_sleep(monkey):
    """Record sleeps instead of taking them."""
    slept = []
    monkey["orig"] = ro.time.sleep
    ro.time.sleep = lambda s: slept.append(s)
    return slept


def _restore(monkey):
    ro.time.sleep = monkey["orig"]


# --------------------------------------------------------------------------- #
# Backoff schedule
# --------------------------------------------------------------------------- #
def test_backoff_outlasts_a_rolling_minute():
    """10.5s of retries never cleared a single real 429. Reach past 60s."""
    m = {}
    slept = _no_sleep(m)
    try:
        client = Client(fail_times=99)
        try:
            ro.place_order_with_retry(client, "acct", {"price": "1.00"})
        except Exception:
            pass
        total = sum(slept)
        assert total >= 60, (
            f"total backoff {total:.1f}s cannot outlast a rolling-minute "
            f"quota (schedule was {slept})")
        assert max(slept) <= 30, f"no single sleep should stall a scan: {slept}"
    finally:
        _restore(m)


def test_a_transient_429_eventually_succeeds():
    m = {}
    _no_sleep(m)
    try:
        client = Client(fail_times=2)
        resp = ro.place_order_with_retry(client, "acct", {"price": "1.00"})
        assert resp.status_code == 201
        assert client.calls == 3, client.calls
    finally:
        _restore(m)


def test_retry_after_is_honored():
    """If Schwab tells us how long to wait, wait that long."""
    m = {}
    slept = _no_sleep(m)
    try:
        client = Client(fail_times=1, retry_after=17)
        ro.place_order_with_retry(client, "acct", {"price": "1.00"})
        assert 17 in slept, f"Retry-After: 17 ignored; slept {slept}"
    finally:
        _restore(m)


def test_a_non_429_error_is_not_retried():
    """A rejected order must fail fast, not be resubmitted five times."""
    m = {}
    _no_sleep(m)
    try:
        class Rejecting(Client):
            def place_order(self, account_hash, order):
                self.calls += 1
                return Resp(400)

        client = Rejecting()
        try:
            ro.place_order_with_retry(client, "acct", {"price": "1.00"})
            assert False, "a 400 must propagate"
        except Exception:
            pass
        assert client.calls == 1, f"retried a hard rejection {client.calls}x"
    finally:
        _restore(m)


# --------------------------------------------------------------------------- #
# Spacing between order-endpoint writes
# --------------------------------------------------------------------------- #
def test_consecutive_order_writes_are_spaced():
    """The 10-05 case: an open and its cover fired ~2s apart and the cover 429'd."""
    m = {}
    slept = _no_sleep(m)
    try:
        ro._reset_order_throttle()
        client = Client()
        ro.place_order_with_retry(client, "acct", {"price": "1.00"})
        slept.clear()
        ro.place_order_with_retry(client, "acct", {"price": "2.00"})
        assert sum(slept) > 0, (
            "second order write went out with no spacing at all")
        assert sum(slept) <= ro.MIN_ORDER_GAP_S + 0.01, slept
    finally:
        _restore(m)


def test_the_first_order_write_is_not_delayed():
    """Spacing must never add latency to an isolated order."""
    m = {}
    slept = _no_sleep(m)
    try:
        ro._reset_order_throttle()
        ro.place_order_with_retry(Client(), "acct", {"price": "1.00"})
        assert sum(slept) == 0, f"delayed the first order by {sum(slept)}s"
    finally:
        _restore(m)


def test_spacing_elapses_naturally():
    """A write long after the previous one waits for nothing."""
    m = {}
    slept = _no_sleep(m)
    try:
        ro._reset_order_throttle()
        ro.place_order_with_retry(Client(), "acct", {"price": "1.00"})
        ro._LAST_ORDER_CALL -= (ro.MIN_ORDER_GAP_S + 5)   # pretend time passed
        slept.clear()
        ro.place_order_with_retry(Client(), "acct", {"price": "2.00"})
        assert sum(slept) == 0, f"waited {sum(slept)}s when the gap had elapsed"
    finally:
        _restore(m)


# --------------------------------------------------------------------------- #
# The EarlyTP replace must use the same path
# --------------------------------------------------------------------------- #
def test_replace_order_goes_through_the_retry_helper():
    """A transient 429 on a replace must be retried, not silently lost."""
    m = {}
    _no_sleep(m)
    try:
        ro._reset_order_throttle()
        client = Client(fail_times=2)
        resp = ro.replace_order_with_retry(client, "acct", "oldid",
                                           {"price": "1.21"})
        assert resp.status_code == 201
        assert client.calls == 3, client.calls
    finally:
        _restore(m)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall order-throttle tests passed")
