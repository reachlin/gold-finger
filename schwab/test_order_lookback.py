"""The order-lookback window must outlive a resting GTC cover.

Failure of 2026-10-02. `fetch_orders` searched by `from_entered_datetime` over a
7-day window. Every GTC buy-to-close cover is placed the moment a position
OPENS and then deliberately rests for the option's whole life — that is the
entire point of it being GTC (weekend- and token-expiry-safe). With ~30-day
DTE, a cover routinely rests 3-5x longer than the lookback.

So T0089, entered 2026-09-08 and filled 2026-10-02, was invisible to the
reconciler. It saw the position gone from Schwab with no matching closing order
and logged:

    [Reconcile] AMZN $245.0 gone from Schwab before expiry with no
                closing order — manual review

Nothing had gone wrong at the broker: the order filled normally at $1.45
against its $1.69 limit. The consequence was a ledger row that never got
written, so `_committed_collateral()` kept counting $24,500 of freed collateral
and the scanner blocked AAPL ($31,500) and GOOGL ($32,500) against a stale
free-cash figure — on the first day there was real capital to deploy.

This was guaranteed to recur. Every earlier close happened within days of its
cover being placed, which is the only reason it had not surfaced.
"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

import real_overseer as ro


class FakeClient:
    """Records the datetime range fetch_orders asks for."""

    def __init__(self):
        self.kw = None

    def get_orders_for_account(self, account_hash, **kw):
        self.kw = kw

        class R:
            status_code = 200

            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return []
        return R()


def _window_days(client) -> float:
    span = client.kw["to_entered_datetime"] - client.kw["from_entered_datetime"]
    return span.total_seconds() / 86400


def test_default_window_covers_a_full_option_lifetime():
    """A cover placed at open on a ~30-DTE position must still be found when it
    fills near expiry. 7 days was not enough; the default must comfortably
    exceed the longest DTE the scanner trades (~35 days)."""
    c = FakeClient()
    ro.fetch_orders(c, "hash")
    assert _window_days(c) >= 60, (
        f"lookback is only {_window_days(c):.0f} days — a resting GTC cover "
        f"that fills later becomes invisible and its close is never booked"
    )


def test_the_exact_2026_10_02_order_would_be_found():
    """T0089: entered 2026-09-08, filled 2026-10-02 — 24 days apart."""
    c = FakeClient()
    ro.fetch_orders(c, "hash")
    entered = datetime(2026, 9, 8, 13, 32)
    filled = datetime(2026, 10, 2, 13, 30)
    window_start = filled - timedelta(days=_window_days(c))
    assert entered >= window_start, (
        "the order that caused the 2026-10-02 manual-review flag would still "
        "fall outside the lookback"
    )


def test_window_is_still_explicitly_overridable():
    c = FakeClient()
    ro.fetch_orders(c, "hash", days_back=3)
    assert abs(_window_days(c) - 3) < 0.1


def test_a_fetch_failure_still_returns_empty_not_raises():
    """The reconciler must degrade, not crash, if the orders call fails."""
    class Boom:
        def get_orders_for_account(self, *a, **k):
            raise RuntimeError("schwab down")

    assert ro.fetch_orders(Boom(), "hash") == []
