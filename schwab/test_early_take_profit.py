"""
_maybe_early_take_profit must not claim a tighten it did not achieve.

The defect this pins, observed live on 2026-10-08. Two EarlyTP replaces fired in
one scan. GOOGL's succeeded (old order REPLACED, new order FILLED at $4.33,
+$256 booked). XOM's did NOT -- Schwab still had order 1008209230096 WORKING at
$0.77, the same id, never replaced -- yet the overseer printed

    [EarlyTP] ⚡ fast winner: XOM 30% in 1d — tightened GTC $0.77 → $1.40 (bank +$60)

and wrote limit=1.40, early_tp=True into pending_orders.json.

Two causes, both fixed here:
  1. replace_order's response was never status-checked. A non-2xx returns
     normally, so nothing raised.
  2. `new_id = ... if loc else old_order_id` then reused the OLD id, making a
     failed replace indistinguishable from a successful one.

The compounding harm is `early_tp`, a one-shot flag nothing ever clears: a
failure that sets it is never retried, so the position's cover stays at the deep
target forever while local state lies about it.

The protective cover being intact on failure is the one thing that already
worked, and these tests keep it that way.
"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import schwab.orders.options  # noqa: F401  (grafted by conftest.py)

import real_overseer as ro


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeHTTPError(Exception):
    pass


class FakeResp:
    """A replace_order response. `location=None` means Schwab returned no id."""

    def __init__(self, status=201, location="https://api/accounts/abc/orders/222"):
        self.status_code = status
        self.headers = {"Location": location} if location else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise FakeHTTPError(
                f"Client error '{self.status_code} Too Many Requests' for url "
                f"'https://api.schwabapi.com/trader/v1/accounts/abc/orders'")


class QuoteResp:
    def __init__(self, mark, iv=None):
        self._mark, self._iv = mark, iv

    def raise_for_status(self):
        pass

    def json(self):
        return {OCC: {"quote": {"mark": self._mark, "volatility": self._iv}}}


class FakeClient:
    def __init__(self, mark, replace_resp):
        self._mark, self._replace_resp = mark, replace_resp
        self.replace_calls = []

    def get_quotes(self, syms):
        return QuoteResp(self._mark, 22.0)

    def replace_order(self, account_hash, order_id, order):
        self.replace_calls.append((order_id, order))
        if isinstance(self._replace_resp, Exception):
            raise self._replace_resp
        return self._replace_resp


class FakeScanner:
    def __init__(self):
        self.slack = []

    def _send_slack(self, msg):
        self.slack.append(msg)


OCC      = "XOM   261106P00155000"
OPEN_REF = "XOM_2026-10-07 12:06:34"
OLD_ID   = "1008209230096"


def _state(tmp, limit=0.77, early_tp=None, order_id=OLD_ID):
    ro.PENDING_ORDERS_PATH = os.path.join(tmp, "pending_orders.json")
    ro._TRADE_COUNTER_PATH = os.path.join(tmp, "trade_counter.json")
    with open(ro.PENDING_ORDERS_PATH, "w") as f:
        json.dump([{
            "trade_id": "T0109", "symbol": "XOM", "signal": "BUY_TO_CLOSE",
            "strike": 155.0, "limit": limit, "occ_sym": OCC,
            "schwab_order_id": order_id, "opening_ref": OPEN_REF,
            "duration": "GTC", "early_tp": early_tp,
        }], f)


def _opening(premium_sh=2.00, date="2026-10-07 12:06:34"):
    return {"symbol": "XOM", "strike": 155.0, "premium_sh": premium_sh,
            "date": date, "signal": "SELL_PUT"}


def _run(tmp, mark, replace_resp, **state):
    _state(tmp, **state)
    ov      = ro.RealOverseer.__new__(ro.RealOverseer)
    client  = FakeClient(mark, replace_resp)
    scanner = FakeScanner()
    ov._maybe_early_take_profit(scanner, client, "acct", _opening(), OCC, OPEN_REF)
    return json.load(open(ro.PENDING_ORDERS_PATH))[0], client, scanner


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_rejected_replace_leaves_state_untouched():
    """The live XOM case: a 429 must not look like a banked profit."""
    with tempfile.TemporaryDirectory() as tmp:
        entry, client, scanner = _run(tmp, 1.40, FakeResp(status=429))

        assert len(client.replace_calls) == 1, "the replace must be attempted"
        assert entry["limit"] == 0.77, f"limit was changed to {entry['limit']}"
        assert entry["schwab_order_id"] == OLD_ID, entry["schwab_order_id"]
        assert not entry.get("early_tp"), (
            "early_tp set on a FAILED replace — the one-shot flag would block "
            "every future retry for this position")
        assert not scanner.slack, f"announced a tighten that did not happen: {scanner.slack}"
    print("ok  test_rejected_replace_leaves_state_untouched")


def test_success_without_order_id_is_not_success():
    """A 2xx with no Location header leaves us unable to track the new order."""
    with tempfile.TemporaryDirectory() as tmp:
        entry, client, scanner = _run(tmp, 1.40, FakeResp(status=201, location=None))

        assert entry["limit"] == 0.77, entry["limit"]
        assert entry["schwab_order_id"] == OLD_ID, entry["schwab_order_id"]
        assert not entry.get("early_tp"), "early_tp set with no new order id"
        assert not scanner.slack, scanner.slack
    print("ok  test_success_without_order_id_is_not_success")


def test_network_exception_leaves_state_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        entry, client, scanner = _run(tmp, 1.40, FakeHTTPError("connection reset"))

        assert entry["limit"] == 0.77, entry["limit"]
        assert not entry.get("early_tp"), entry
        assert not scanner.slack, scanner.slack
    print("ok  test_network_exception_leaves_state_untouched")


def test_successful_replace_records_the_new_order():
    """The GOOGL case: a real replace updates limit, id and the one-shot flag."""
    with tempfile.TemporaryDirectory() as tmp:
        entry, client, scanner = _run(tmp, 1.40, FakeResp(status=201))

        assert len(client.replace_calls) == 1, client.replace_calls
        assert client.replace_calls[0][0] == OLD_ID, "must replace the OLD order"
        assert client.replace_calls[0][1]["price"] == "1.40", client.replace_calls[0][1]
        assert entry["limit"] == 1.40, entry["limit"]
        assert entry["schwab_order_id"] == "222", entry["schwab_order_id"]
        assert entry["early_tp"] is True, entry
        assert any("Early take-profit" in m for m in scanner.slack), scanner.slack
    print("ok  test_successful_replace_records_the_new_order")


def test_one_shot_per_position():
    """Once genuinely tightened, never churn the same position again."""
    with tempfile.TemporaryDirectory() as tmp:
        entry, client, scanner = _run(tmp, 1.80, FakeResp(status=201),
                                      limit=1.40, early_tp=True)
        assert not client.replace_calls, "already tightened — must not churn"
        assert entry["limit"] == 1.40, entry["limit"]
    print("ok  test_one_shot_per_position")


def test_loser_is_never_tightened():
    """A position under water must not be 'taken profit' on at any price."""
    with tempfile.TemporaryDirectory() as tmp:
        # XOM 2026-10-05: entry $0.79, mark $1.61 -- a 104% LOSS
        _state(tmp, limit=0.31)
        ov      = ro.RealOverseer.__new__(ro.RealOverseer)
        client  = FakeClient(1.61, FakeResp(status=201))
        scanner = FakeScanner()
        ov._maybe_early_take_profit(scanner, client, "acct",
                                    _opening(premium_sh=0.79, date="2026-10-05 09:45:14"),
                                    OCC, OPEN_REF)
        assert not client.replace_calls, "tightened a LOSING position"
        entry = json.load(open(ro.PENDING_ORDERS_PATH))[0]
        assert entry["limit"] == 0.31, entry["limit"]
    print("ok  test_loser_is_never_tightened")


if __name__ == "__main__":
    test_rejected_replace_leaves_state_untouched()
    test_success_without_order_id_is_not_success()
    test_network_exception_leaves_state_untouched()
    test_successful_replace_records_the_new_order()
    test_one_shot_per_position()
    test_loser_is_never_tightened()
    print("\nall early-take-profit tests passed")
