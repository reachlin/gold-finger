"""Tests for RealOverseer._readopt_untracked_covers — restart/crash recovery.

A cover order can be live at Schwab yet missing from pending_orders.json if the
process dies between placing the order and persisting it. Reconcile must adopt
those back so the position regains early-TP eligibility and local visibility.
"""
import os, sys, json, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import real_overseer as ro


class StubScanner:
    OPTION_LEDGER_PATH = "/nonexistent/ledger.csv"   # → open_rows == []
    def _send_slack(self, msg): pass


def _order(status, instr, occ_spaced, oid, price):
    return {"status": status, "orderId": oid, "price": price,
            "orderLegCollection": [{"instruction": instr,
                                    "instrument": {"symbol": occ_spaced}}]}


def _setup():
    tmp = tempfile.mktemp(suffix=".json")
    ro.PENDING_ORDERS_PATH = tmp
    with open(tmp, "w") as f:
        json.dump([], f)
    ro._next_trade_id = lambda: "TEST"          # don't touch the real counter
    return tmp, ro.RealOverseer.__new__(ro.RealOverseer)


def test_readopts_untracked_working_cover():
    tmp, ov = _setup()
    orders = [_order("PENDING_ACTIVATION", "BUY_TO_CLOSE",
                     "NVDA  260918P00205000", "111", 2.02)]
    assert ov._readopt_untracked_covers(StubScanner(), orders) == 1
    e = json.load(open(tmp))[0]
    assert e["symbol"] == "NVDA" and e["strike"] == 205.0
    assert e["signal"] == "BUY_TO_CLOSE" and e["duration"] == "GTC"
    assert e["occ_sym"] == "NVDA  260918P00205000"   # spaced, quote-ready
    assert e["limit"] == 2.02 and e["schwab_order_id"] == "111"
    assert e["readopted"] is True and e["entry_iv"] is None


def test_idempotent():
    tmp, ov = _setup()
    orders = [_order("WORKING", "BUY_TO_CLOSE", "NVDA  260918P00205000", "111", 2.02)]
    ov._readopt_untracked_covers(StubScanner(), orders)
    assert ov._readopt_untracked_covers(StubScanner(), orders) == 0   # no dupe
    assert len(json.load(open(tmp))) == 1


def test_dedup_by_order_id_not_contract():
    # Dedup is by ORDER ID, not contract: the same order id is skipped, but a
    # DIFFERENT order id for the same contract (a stacked 2nd cover) is adopted.
    tmp, ov = _setup()
    json.dump([{"occ_sym": "NVDA  260918P00205000", "signal": "BUY_TO_CLOSE",
                "schwab_order_id": "111", "opening_ref": "NVDA_A"}], open(tmp, "w"))
    same = [_order("WORKING", "BUY_TO_CLOSE", "NVDA  260918P00205000", "111", 2.02)]
    assert ov._readopt_untracked_covers(StubScanner(), same) == 0   # already tracked
    diff = [_order("WORKING", "BUY_TO_CLOSE", "NVDA  260918P00205000", "222", 2.02)]
    assert ov._readopt_untracked_covers(StubScanner(), diff) == 1   # 2nd cover adopted


def test_ignores_non_working_and_non_btc():
    tmp, ov = _setup()
    orders = [
        _order("FILLED", "BUY_TO_CLOSE", "NVDA  260918P00205000", "1", 2.0),   # not working
        _order("WORKING", "SELL_TO_OPEN", "AAPL  260918P00200000", "2", 3.0),  # not a cover
        _order("CANCELED", "BUY_TO_CLOSE", "IBM   260918P00220000", "3", 1.0), # dead
    ]
    assert ov._readopt_untracked_covers(StubScanner(), orders) == 0


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
