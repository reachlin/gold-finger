"""The booking layer must record, and spend, every contract it sold.

Sizing (2026-10-14, `MULTI_CONTRACT_SIZING`) can return more than one
contract, and the order paths place that quantity. But the quantity died at
the ledger door: `_confirm_open_fill` kept `contracts` in the pending entry
and handed `book_open_fill` a dict built from `signal_data | {symbol, signal,
strike, dte}`, which never carried it, and `book_open_fill`'s row dict never
wrote the column. Proven by booking a 3-contract fill through the real
function on 2026-10-21: `row['contracts'] == ''`, so `contracts_of` read 1,
`committed_collateral` reported $8,800 of a real $26,400, and
`_cover_contracts` wanted to buy back 1 of 3.

Three live consequences, all from that one dropped field:

  * `_symbol_exposure` under-reports, so the 20% per-symbol cap stops binding
    and the book can stack past it — the concentration limit that exists
    because of the 99%-AMZN incident.
  * the GTC cover buys back 1 of N and leaves N-1 naked while the safety
    ceiling, comparing short contracts against resting BUY_TO_CLOSE
    quantity, reads "covered".
  * cash and P&L book one contract's money for an N-contract trade.

So the invariant under test is not just "write the column". It is that every
place a ledger row turns into money or shares multiplies by the count:
premium credited, buyback cost, expiry P&L, assignment cash, and the share
count that lands in wheel holdings.

`premium_sh` and `premium_ct` stay PER CONTRACT. That keeps the 61 rows
written before this commit meaningful and keeps `contracts_of`'s default of 1
correct for them; the multiplication happens where the row is spent, not
where it is stored.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import options_ledger as ol  # noqa: E402
import real_overseer as ro  # noqa: E402


LIVE_LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "data", "paper_options_ledger.csv")


class Recorder:
    """Stands in for the cash ledger; keeps what it was told to record."""

    def __init__(self):
        self.entries = []

    def record(self, kind, symbol, amount, detail):
        self.entries.append({"kind": kind, "symbol": symbol,
                             "amount": amount, "detail": detail})

    @property
    def amount(self):
        assert len(self.entries) == 1, self.entries
        return self.entries[0]["amount"]


def _signal(symbol="KO", strike=88.0, contracts=None, **kw):
    s = {"symbol": symbol, "signal": "SELL_PUT", "strike": strike,
         "close": strike * 1.05, "dte": 30, "hv": 24.0}
    if contracts is not None:
        s["contracts"] = contracts
    s.update(kw)
    return s


def _opening(tmp_path, contracts=None, fill=2.50, **kw):
    """Book an opening fill and return (ledger_path, the row as read back)."""
    led = str(tmp_path / "led.csv")
    ro.book_open_fill(led, None, _signal(contracts=contracts, **kw), fill)
    return led, ol.read_rows(led)[-1]


# ---------------------------------------------------------------------------
# The dropped field
# ---------------------------------------------------------------------------

def test_book_open_fill_records_the_contract_count(tmp_path):
    _, row = _opening(tmp_path, contracts=3)
    assert row["contracts"] == "3"
    assert ol.contracts_of(row) == 3


def test_book_open_fill_defaults_to_one_contract(tmp_path):
    """A signal that does not size itself is one contract, as it always was."""
    _, row = _opening(tmp_path)
    assert ol.contracts_of(row) == 1


def test_the_fill_path_carries_the_count_from_the_pending_entry():
    """
    The bug lived in the dict literal inside `_confirm_open_fill`, which no
    test could reach. The merge is now a named function, and this is the
    assertion that would have failed.
    """
    entry = {"symbol": "KO", "signal": "SELL_PUT", "strike": 88.0, "dte": 30,
             "contracts": 3,
             "signal_data": {"symbol": "KO", "signal": "SELL_PUT",
                             "strike": 88.0, "close": 92.6, "dte": 30}}
    handed = ro._fill_signal(entry)
    assert ol.contracts_of(handed) == 3
    for field in ("symbol", "signal", "strike", "dte"):
        assert handed[field] == entry[field]


def test_the_fill_path_defaults_to_one_when_the_entry_predates_sizing():
    """A pending entry written before sizing existed has no contracts key."""
    entry = {"symbol": "KO", "signal": "SELL_PUT", "strike": 88.0, "dte": 30,
             "signal_data": {"symbol": "KO", "signal": "SELL_PUT",
                             "strike": 88.0, "dte": 30}}
    assert ol.contracts_of(ro._fill_signal(entry)) == 1


# ---------------------------------------------------------------------------
# What the dropped field broke
# ---------------------------------------------------------------------------

def test_collateral_reflects_every_contract_sold(tmp_path):
    led, _ = _opening(tmp_path, contracts=3)
    rows = ol.read_rows(led)
    assert ol.committed_collateral(rows) == 88.0 * 100 * 3
    assert ol.symbol_collateral(rows, "KO") == 26_400.0


def test_a_cover_for_a_three_contract_short_buys_back_three(tmp_path):
    """The naked-put hole: a 1-contract cover on a 3-contract short."""
    _, row = _opening(tmp_path, contracts=3)
    assert ro._cover_contracts(row) == 3


def test_the_short_quantity_map_counts_contracts_not_rows(tmp_path):
    """
    `_short_quantities` feeds the safety ceiling that decides whether a
    position is covered. One row holding three contracts is three shorts.
    """
    led, row = _opening(tmp_path, contracts=3)
    occ_map = {ol.row_key(row): "KO    261120P00088000"}
    assert ro._short_quantities([row], occ_map) == {"KO261120P00088000": 3}


# ---------------------------------------------------------------------------
# Money: open
# ---------------------------------------------------------------------------

def test_the_premium_credited_scales_with_contracts(tmp_path):
    rec = Recorder()
    ro.book_open_fill(str(tmp_path / "l.csv"), rec,
                      _signal(contracts=3), 2.50)
    assert rec.amount == 750.0            # $2.50 x 100 x 3
    assert rec.entries[0]["kind"] == "OPTION_SELL"


def test_a_one_contract_fill_credits_exactly_one_contract(tmp_path):
    rec = Recorder()
    ro.book_open_fill(str(tmp_path / "l.csv"), rec, _signal(), 2.50)
    assert rec.amount == 250.0


def test_the_stored_premium_stays_per_contract(tmp_path):
    """
    Legacy rows depend on this: premium_ct has meant one contract for 61
    rows, and contracts_of defaults them to 1.
    """
    _, row = _opening(tmp_path, contracts=3, fill=2.50)
    assert float(row["premium_sh"]) == 2.50
    assert float(row["premium_ct"]) == 250.0


# ---------------------------------------------------------------------------
# Money: close
# ---------------------------------------------------------------------------

def _booked(tmp_path, contracts=3, fill=2.50):
    led, row = _opening(tmp_path, contracts=contracts, fill=fill)
    return led, row


def test_buyback_pnl_and_cost_scale_with_contracts(tmp_path):
    led, opening = _booked(tmp_path)
    rec = Recorder()
    pnl = ro.book_buyback(led, rec, opening, 1.00)
    assert pnl == (2.50 - 1.00) * 100 * 3          # $450
    assert rec.amount == -300.0                    # $1.00 x 100 x 3 paid out
    assert float(ol.read_rows(led)[-1]["pnl"]) == 450.0


def test_a_one_contract_buyback_is_unchanged(tmp_path):
    led, opening = _booked(tmp_path, contracts=1)
    rec = Recorder()
    assert ro.book_buyback(led, rec, opening, 1.00) == 150.0
    assert rec.amount == -100.0


def test_expiry_pnl_scales_with_contracts(tmp_path):
    led, opening = _booked(tmp_path)
    rec = Recorder()
    assert ro.book_expired(led, rec, opening) == 750.0
    assert rec.amount == 0.0


def test_assignment_pays_for_every_contract(tmp_path):
    led, opening = _booked(tmp_path)
    rec = Recorder()
    hold = str(tmp_path / "hold.json")
    ro.book_assigned(led, rec, opening, hold)
    assert rec.amount == -26_400.0                 # 88 x 100 x 3
    assert json.load(open(hold))["KO"]["contracts"] == 3


def test_assignment_stacks_contract_counts_and_averages_the_basis(tmp_path):
    led, first = _booked(tmp_path, contracts=1, fill=2.50)
    hold = str(tmp_path / "hold.json")
    ro.book_assigned(led, None, first, hold)
    # a second assignment of three contracts at a different premium
    second = dict(first, premium_sh=4.50, contracts=3)
    ro.book_assigned(led, None, second, hold)
    h = json.load(open(hold))["KO"]
    assert h["contracts"] == 4
    # (85.50 x 1 + 83.50 x 3) / 4
    assert h["cost_basis"] == 84.0


def test_close_rows_carry_the_count_forward(tmp_path):
    """
    A CLOSED row that reads as one contract would make any per-row P&L or
    collateral walk over history wrong.
    """
    led, opening = _booked(tmp_path)
    ro.book_buyback(led, None, opening, 1.00)
    assert ol.contracts_of(ol.read_rows(led)[-1]) == 3


# ---------------------------------------------------------------------------
# The settlement sweep in options_ledger
# ---------------------------------------------------------------------------

def _expired_put(tmp_path, contracts=3, strike=88.0, premium=2.50):
    import pathlib
    led = str(pathlib.Path(tmp_path) / "led.csv")
    ol.append_row(led, {"date": "2026-01-02 10:00:00", "symbol": "KO",
                        "signal": "SELL_PUT", "strike": strike,
                        "premium_sh": premium, "premium_ct": premium * 100,
                        "dte": 30, "hv": 24.0, "verdict": "APPROVED",
                        "contracts": contracts})
    return led


def test_worthless_expiry_keeps_the_premium_on_every_contract(tmp_path):
    from datetime import date
    led = _expired_put(tmp_path)
    events = ol.process_expirations(
        led, str(tmp_path / "h.json"),
        lambda sym: {"close": 95.0, "hv": 0.24},
        today=date(2026, 2, 2))
    assert [e["action"] for e in events] == ["put_expired"]
    assert events[0]["pnl"] == 750.0
    assert float(ol.read_rows(led)[-1]["pnl"]) == 750.0


def test_expiry_assignment_records_shares_for_every_contract(tmp_path):
    from datetime import date
    led = _expired_put(tmp_path)
    hold = str(tmp_path / "h.json")
    events = ol.process_expirations(
        led, hold,
        lambda sym: {"close": 80.0, "hv": 0.24},
        today=date(2026, 2, 2))
    assert [e["action"] for e in events] == ["put_assigned"]
    assert json.load(open(hold))["KO"]["contracts"] == 3
    assert "300 sh" in events[0]["detail"]


def _early_exit_pnl(tmp_path, contracts):
    from datetime import date
    led = _expired_put(tmp_path / str(contracts), contracts=contracts,
                       premium=5.00)
    events = ol.process_expirations(
        led, str(tmp_path / f"h{contracts}.json"),
        lambda sym: {"close": 120.0, "hv": 0.10},   # deep OTM, near worthless
        today=date(2026, 1, 20))
    assert [e["action"] for e in events] == ["early_exit"], events
    return events[0]["pnl"]


def test_early_exit_pnl_scales_with_contracts(tmp_path):
    """Three contracts earn three times what one earns, same scenario."""
    (tmp_path / "1").mkdir()
    (tmp_path / "3").mkdir()
    one   = _early_exit_pnl(tmp_path, 1)
    three = _early_exit_pnl(tmp_path, 3)
    assert one > 0
    assert three == round(one * 3, 10)


# ---------------------------------------------------------------------------
# Nothing written before this commit changes meaning
# ---------------------------------------------------------------------------

def test_legacy_rows_with_an_empty_contracts_column_book_as_one(tmp_path):
    led = str(tmp_path / "led.csv")
    ol.append_row(led, {"date": "2026-10-13 10:43:30", "symbol": "NVDA",
                        "signal": "SELL_PUT", "strike": 215.0,
                        "premium_sh": 3.85, "premium_ct": 385.0, "dte": 31,
                        "verdict": "APPROVED"})
    row = ol.read_rows(led)[-1]
    assert row["contracts"] == ""
    assert ol.contracts_of(row) == 1
    assert ol.committed_collateral([row]) == 21_500.0
    assert ro._cover_contracts(row) == 1


def test_the_live_ledger_still_reconciles_to_the_account():
    """
    The real book on 2026-10-21: NVDA $215 and IBM $227.50, one contract
    each, against Schwab's committed $44,250. Every row predates the
    contracts column or has it blank, so this passes only while
    contracts_of's default of 1 holds.
    """
    rows = ol.read_rows(LIVE_LEDGER)
    assert ol.committed_collateral(rows) == 44_250.0
    assert ol.symbol_collateral(rows, "NVDA") == 21_500.0
    assert ol.symbol_collateral(rows, "IBM") == 22_750.0
