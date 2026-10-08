"""
The limit price we compute must be the limit price the broker receives.

Background. schwab-py's set_price accepts a float but converts it lossily, and
warns about it: "passing floats to set_price and set_stop_price is deprecated ...
Please update your code to pass prices as strings instead." That warning is not
cosmetic -- the float path TRUNCATES. Measured across all 640 tradable tick
values from $0.01 to $20.00, 47 of them (7.3%) reach the broker a cent BELOW
what was asked, always downward:

    float 2.01  -> '2.00'     a cent given away on every sell
    float 0.77  -> '0.7700'   harmless, but shows the conversion is ad hoc
    str  '2.01' -> '2.01'     correct

This reached production. XOM 261106P00155000 on 2026-10-07: bid 1.87 / ask 2.14,
mid 2.005, round_up_to_tick -> 2.01, logged as "limit mid-on-tick $2.01" -- and
Schwab's own order record says price 2.00, executed 2.00. Only $1, but it
silently defeats the round-UP-to-tick rule that chain_quotes documents as
load-bearing, and it leans the wrong way on a sell.

Both BUY_TO_CLOSE paths already pass f"{x:.2f}" strings; only the SELL_TO_OPEN
path was left on the float form. These tests pin all three.
"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Resolved by conftest.py under pytest; needed for a direct script run too.
import schwab.orders.options  # noqa: F401
from schwab.orders.options import option_sell_to_open_limit
from schwab.orders.common import Duration, Session

import real_overseer as ro
from chain_quotes import round_up_to_tick


# --------------------------------------------------------------------------- #
# The tick grid the market actually displays: 1c below $3, 5c at or above.
# --------------------------------------------------------------------------- #
def _tick_grid():
    vals = [round(i * 0.01, 2) for i in range(1, 300)]           # 0.01 .. 2.99
    vals += [round(3.00 + i * 0.05, 2) for i in range(0, 341)]   # 3.00 .. 20.00
    return vals


def _built_price(limit):
    """The price string schwab-py would send for this limit argument."""
    return (option_sell_to_open_limit("XOM   261106P00155000", 1, limit)
            .set_duration(Duration.DAY)
            .set_session(Session.NORMAL)
            .build())["price"]


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeResp:
    headers = {"Location": "https://api/accounts/abc/orders/999"}

    def raise_for_status(self):
        pass


class FakeScanner:
    _current_client = object()

    def __init__(self):
        self.slack = []

    def _send_slack(self, msg):
        self.slack.append(msg)


def _place_and_capture(monkey_state, signal):
    """Run the REAL _place_order and return the order dict handed to Schwab.

    Everything outside the price path is stubbed: the broker call, the account
    hash, the pre-trade check, and the fill poll (left unfilled on purpose so
    _confirm_open_fill never runs and the order dict is all we observe).
    """
    captured = {}

    def fake_place(client, account_hash, order, **kw):
        captured["order"] = order
        return FakeResp()

    scanner_stub = FakeScanner()
    sys.modules["live_scanner"] = scanner_stub          # the function imports it

    saved = dict(place=ro.place_order_with_retry, hash=ro.get_account_hash,
                 fetch=ro.fetch_orders, find=ro.find_order, sleep=ro.time.sleep,
                 real=os.environ.get("REALLY_REAL"))
    try:
        ro.place_order_with_retry = fake_place
        ro.get_account_hash      = lambda c: "acct"
        ro.fetch_orders          = lambda *a, **k: []
        ro.find_order            = lambda *a, **k: None   # never fills
        ro.time.sleep            = lambda s: None         # don't wait 30s
        os.environ["REALLY_REAL"] = "true"

        ov = ro.RealOverseer.__new__(ro.RealOverseer)
        ov._pre_trade_check = lambda client, account_hash, s: (True, "stubbed OK")
        ov._place_order(signal)
    finally:
        ro.place_order_with_retry = saved["place"]
        ro.get_account_hash      = saved["hash"]
        ro.fetch_orders          = saved["fetch"]
        ro.find_order            = saved["find"]
        ro.time.sleep            = saved["sleep"]
        if saved["real"] is None:
            os.environ.pop("REALLY_REAL", None)
        else:
            os.environ["REALLY_REAL"] = saved["real"]
        sys.modules.pop("live_scanner", None)

    return captured.get("order"), scanner_stub


def _signal(order_limit):
    return dict(symbol="XOM", signal="SELL_PUT", strike=155.0,
                expiry="2026-11-06", dte=30, premium=order_limit,
                order_limit=order_limit, ask=2.14, confidence=72,
                reason="test")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_float_prices_are_lossy():
    """Document WHY the string form is required, so nobody 'simplifies' it back."""
    mangled = [(v, _built_price(v)) for v in _tick_grid()]
    mangled = [(v, s) for v, s in mangled if float(s) != v]

    assert mangled, ("schwab-py no longer mangles float prices -- if this "
                     "library was upgraded, re-check whether the string "
                     "conversion in _place_order is still needed")
    assert (2.01, "2.00") in mangled, mangled[:10]
    # every error is downward: never pay more, but never collect less either
    assert all(float(s) - v < 0 for v, s in mangled), mangled[:10]
    print(f"ok  test_float_prices_are_lossy  ({len(mangled)} of "
          f"{len(_tick_grid())} tick values truncate)")


def test_sell_to_open_sends_the_asked_limit():
    """The XOM case that reached production: ask 2.01, the broker must get 2.01."""
    with tempfile.TemporaryDirectory() as tmp:
        ro.PENDING_ORDERS_PATH = os.path.join(tmp, "pending_orders.json")
        ro._TRADE_COUNTER_PATH = os.path.join(tmp, "trade_counter.json")

        order, scanner = _place_and_capture(tmp, _signal(2.01))

        assert order is not None, "no order was placed"
        assert order["price"] == "2.01", (
            f"broker received {order['price']!r}, we asked for 2.01")
        assert order["duration"] == "DAY", order["duration"]
        leg = order["orderLegCollection"][0]
        assert leg["instruction"] == "SELL_TO_OPEN", leg["instruction"]
        # the pending record must agree with what the broker got
        assert ro._load_pending()[0]["limit"] == 2.01, ro._load_pending()
        assert any("2.01" in m for m in scanner.slack), scanner.slack
    print("ok  test_sell_to_open_sends_the_asked_limit")


def test_sell_to_open_exact_across_the_tick_grid():
    """No tradable tick value may arrive at the broker altered."""
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        ro.PENDING_ORDERS_PATH = os.path.join(tmp, "pending_orders.json")
        ro._TRADE_COUNTER_PATH = os.path.join(tmp, "trade_counter.json")
        for v in _tick_grid():
            order, _ = _place_and_capture(tmp, _signal(v))
            if order is None or float(order["price"]) != v:
                bad.append((v, order and order["price"]))
    assert not bad, f"{len(bad)} tick value(s) altered in transit: {bad[:10]}"
    print(f"ok  test_sell_to_open_exact_across_the_tick_grid "
          f"({len(_tick_grid())} values)")


def test_model_priced_fallback_also_exact():
    """A signal with no chain quote falls back to round_up_to_tick(premium)."""
    with tempfile.TemporaryDirectory() as tmp:
        ro.PENDING_ORDERS_PATH = os.path.join(tmp, "pending_orders.json")
        ro._TRADE_COUNTER_PATH = os.path.join(tmp, "trade_counter.json")
        s = _signal(2.005)
        s.pop("order_limit")                 # model-priced: no live quote
        order, _ = _place_and_capture(tmp, s)
        expected = round_up_to_tick(2.005)   # 2.01
        assert float(order["price"]) == expected, (
            f"broker got {order['price']!r}, expected {expected}")
    print("ok  test_model_priced_fallback_also_exact")


def test_close_paths_still_pass_strings():
    """Regression guard: the two BUY_TO_CLOSE call sites must stay string-formatted."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "real_overseer.py")).read()
    calls = [ln.strip() for ln in src.splitlines()
             if "option_buy_to_close_limit(" in ln and "import" not in ln]
    assert len(calls) == 2, f"expected 2 close call sites, found {len(calls)}: {calls}"
    for c in calls:
        assert ':.2f}"' in c, f"close order price is not a formatted string: {c}"
    print("ok  test_close_paths_still_pass_strings")


if __name__ == "__main__":
    test_float_prices_are_lossy()
    test_sell_to_open_sends_the_asked_limit()
    test_sell_to_open_exact_across_the_tick_grid()
    test_model_priced_fallback_also_exact()
    test_close_paths_still_pass_strings()
    print("\nall order-limit-price tests passed")
