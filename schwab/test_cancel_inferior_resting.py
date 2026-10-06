"""Cancel a resting STO when a materially better candidate appears.

The design, chosen deliberately over an atomic replace_order swap: CANCEL ONLY.
Nothing is placed in the same breath. The cancelled order's collateral frees, the
one-position-per-underlying gate stops seeing a resting order, and the NEXT scan
places whatever wins on its own merits through the normal, already-approved path
-- budget check, LLM review, pre-trade gate, all of it. No second placement path
exists to get wrong, and the better candidate gets no special privilege: it has to
win again next scan, or lose to something better still.

That also removes the cancel-then-place race entirely, because there is no
"then place". If the cancel fails, the original order simply stays live and the
gate keeps blocking, which is the safe direction.

Why this is needed at all: limits moved to the mid on 2026-10-06, so an unfilled
DAY order now rests for the whole session instead of filling in seconds. The gate
added the same day then blocks that underlying for the rest of the day -- so
without this, a stale order beats a better one purely by arriving first.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_overseer as ro


def _fresh(tmp):
    ro._DATA_DIR = tmp
    ro.PENDING_ORDERS_PATH = os.path.join(tmp, "pending_orders.json")
    ro._CANCEL_REQUESTS_PATH = os.path.join(tmp, "cancel_requests.json")


class _Client:
    """Records cancel calls; _pre_trade_check needs nothing else from it."""
    def __init__(self):
        self.cancelled = []

    def cancel_order(self, order_id, account_hash):
        self.cancelled.append((order_id, account_hash))
        class _R:
            status_code = 200
        return _R()


def _resting(price, strike, occ="AMZN  261023P00245000", order_id="999",
             age_min=120, status="WORKING"):
    """A working SELL_TO_OPEN as Schwab reports it."""
    from datetime import datetime, timedelta, timezone
    entered = datetime.now(timezone.utc) - timedelta(minutes=age_min)
    return {
        "orderId": order_id,
        "status": status,
        "price": price,
        "enteredTime": entered.strftime("%Y-%m-%dT%H:%M:%S+0000"),
        "orderLegCollection": [
            {"instruction": "SELL_TO_OPEN", "instrument": {"symbol": occ}}
        ],
    }


def _check(client, orders, signal, monkeypatch, avail=100000.0):
    monkeypatch.setattr(ro, "fetch_account", lambda c, h: {
        "currentBalances": {"availableFunds": avail, "pendingDeposits": 0},
        "positions": [],
    })
    monkeypatch.setattr(ro, "fetch_orders",
                        lambda c, h, days_back=90: orders)
    monkeypatch.setattr(ro, "_load_pending", lambda: [])
    ov = ro.RealOverseer.__new__(ro.RealOverseer)
    return ro.RealOverseer._pre_trade_check(ov, client, "hash", signal)


# resting: $2.45 on a $245 strike = 1.00% of collateral
RESTING = dict(price=2.45, strike=245.0)


def test_a_materially_better_candidate_cancels_the_resting_order(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        c = _Client()
        # 1.50% vs 1.00% — half again as much premium per dollar of collateral
        sig = {"symbol": "AMZN", "strike": 240.0, "signal": "SELL_PUT",
               "order_limit": 3.60}
        ok, msg = _check(c, [_resting(**RESTING)], sig, monkeypatch)
        assert not ok, "this scan must still skip — the cancel may not have landed"
        assert c.cancelled == [("999", "hash")]
        assert "cancel" in msg.lower(), msg


def test_a_marginally_better_candidate_does_not_cancel(monkeypatch):
    """Quote noise must not cause churn."""
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        c = _Client()
        sig = {"symbol": "AMZN", "strike": 240.0, "signal": "SELL_PUT",
               "order_limit": 2.45}          # ~1.02%, barely above 1.00%
        ok, msg = _check(c, [_resting(**RESTING)], sig, monkeypatch)
        assert not ok
        assert c.cancelled == [], "a rounding-error improvement must not cancel"


def test_a_worse_candidate_does_not_cancel(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        c = _Client()
        sig = {"symbol": "AMZN", "strike": 240.0, "signal": "SELL_PUT",
               "order_limit": 1.20}
        ok, _ = _check(c, [_resting(**RESTING)], sig, monkeypatch)
        assert not ok
        assert c.cancelled == []


def test_a_freshly_placed_order_is_left_alone(monkeypatch):
    """An order needs a fair chance to fill before we judge it stale."""
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        c = _Client()
        sig = {"symbol": "AMZN", "strike": 240.0, "signal": "SELL_PUT",
               "order_limit": 3.60}
        ok, _ = _check(c, [_resting(**RESTING, age_min=5)], sig, monkeypatch)
        assert not ok
        assert c.cancelled == [], "must respect MIN_REST_MINUTES"


def test_the_same_order_is_not_cancelled_twice(monkeypatch):
    """Schwab takes time to report CANCELED. Until it does, the order still looks
    WORKING, and re-issuing the cancel every 5 minutes is how a retry storm
    starts -- see the 45 rejected IBM covers on 2026-09-04."""
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        c = _Client()
        sig = {"symbol": "AMZN", "strike": 240.0, "signal": "SELL_PUT",
               "order_limit": 3.60}
        orders = [_resting(**RESTING)]
        _check(c, orders, sig, monkeypatch)
        _check(c, orders, sig, monkeypatch)
        _check(c, orders, sig, monkeypatch)
        assert c.cancelled == [("999", "hash")], \
            f"cancel must be issued once, got {len(c.cancelled)}"


def test_a_failed_cancel_leaves_the_gate_closed(monkeypatch):
    """The safe direction: original order stays live, ticker stays blocked."""
    class _Boom(_Client):
        def cancel_order(self, order_id, account_hash):
            raise RuntimeError("Schwab 503")
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        c = _Boom()
        sig = {"symbol": "AMZN", "strike": 240.0, "signal": "SELL_PUT",
               "order_limit": 3.60}
        ok, msg = _check(c, [_resting(**RESTING)], sig, monkeypatch)
        assert not ok, "a failed cancel must never open the gate"


def test_a_cancel_is_retried_after_a_failure(monkeypatch):
    """A failed attempt must not be recorded as done, or one 503 would strand
    the order for the rest of the session."""
    calls = []

    class _FlakeyThenOk(_Client):
        def cancel_order(self, order_id, account_hash):
            calls.append(order_id)
            if len(calls) == 1:
                raise RuntimeError("Schwab 503")
            return super().cancel_order(order_id, account_hash)

    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        c = _FlakeyThenOk()
        sig = {"symbol": "AMZN", "strike": 240.0, "signal": "SELL_PUT",
               "order_limit": 3.60}
        orders = [_resting(**RESTING)]
        _check(c, orders, sig, monkeypatch)      # fails
        _check(c, orders, sig, monkeypatch)      # succeeds
        assert len(calls) == 2
        assert c.cancelled == [("999", "hash")]


def test_once_cancelled_the_next_scan_is_free_to_place(monkeypatch):
    """The whole point: after Schwab reports CANCELED the order leaves
    WORKING_STATUSES, so the gate opens and the normal path takes over."""
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        c = _Client()
        sig = {"symbol": "AMZN", "strike": 240.0, "signal": "SELL_PUT",
               "order_limit": 3.60}
        ok, msg = _check(c, [_resting(**RESTING, status="CANCELED")],
                         sig, monkeypatch)
        assert ok, msg
        assert c.cancelled == [], "nothing left to cancel"


def test_a_resting_order_on_another_ticker_is_untouched(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        c = _Client()
        sig = {"symbol": "AMZN", "strike": 240.0, "signal": "SELL_PUT",
               "order_limit": 3.60}
        ok, _ = _check(c, [_resting(**RESTING, occ="GOOGL 261030P00330000")],
                       sig, monkeypatch)
        assert ok, "a GOOGL order neither blocks nor is cancelled by an AMZN signal"
        assert c.cancelled == []


def test_a_resting_cover_is_never_cancelled(monkeypatch):
    """BUY_TO_CLOSE covers protect open positions. Never touch them."""
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        c = _Client()
        cover = _resting(**RESTING)
        cover["orderLegCollection"][0]["instruction"] = "BUY_TO_CLOSE"
        sig = {"symbol": "AMZN", "strike": 240.0, "signal": "SELL_PUT",
               "order_limit": 3.60}
        ok, _ = _check(c, [cover], sig, monkeypatch)
        assert ok
        assert c.cancelled == []


# --- the user must be told ---------------------------------------------------

def test_a_cancel_notifies_slack(monkeypatch):
    """Blocks are routine and stay out of Slack — 80 scans a day would spam it.
    Cancelling a live order is not routine: real money was committed to that
    order and the overseer withdrew it, so it belongs in the feed."""
    sent = []
    monkeypatch.setattr(ro, "_send_slack_safe", lambda m: sent.append(m))
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        c = _Client()
        sig = {"symbol": "AMZN", "strike": 240.0, "signal": "SELL_PUT",
               "order_limit": 3.60}
        _check(c, [_resting(**RESTING)], sig, monkeypatch)
        assert c.cancelled == [("999", "hash")]
        assert len(sent) == 1, "exactly one Slack message per cancel"
        body = sent[0]
        assert "AMZN" in body
        assert "261023P00245000" in body, "name the contract withdrawn"
        assert "1.00" in body and "1.50" in body, "show both yields"


def test_a_block_without_a_cancel_is_silent(monkeypatch):
    sent = []
    monkeypatch.setattr(ro, "_send_slack_safe", lambda m: sent.append(m))
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        c = _Client()
        sig = {"symbol": "AMZN", "strike": 240.0, "signal": "SELL_PUT",
               "order_limit": 1.20}          # worse — no cancel
        _check(c, [_resting(**RESTING)], sig, monkeypatch)
        assert sent == [], "a routine block must not reach Slack"


def test_a_slack_failure_does_not_undo_the_cancel(monkeypatch):
    """The order is already cancelled at the broker; a notification problem must
    not make the code behave as though it were not."""
    def _boom(_m):
        raise RuntimeError("slack down")
    monkeypatch.setattr(ro, "_send_slack_safe", _boom)
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        c = _Client()
        sig = {"symbol": "AMZN", "strike": 240.0, "signal": "SELL_PUT",
               "order_limit": 3.60}
        ok, msg = _check(c, [_resting(**RESTING)], sig, monkeypatch)
        assert c.cancelled == [("999", "hash")]
        assert str(c.cancelled[0][0]) in ro._load_cancel_requests(), \
            "the cancel must still be recorded, or it will be re-issued"
        assert not ok
