"""Per-symbol position sizing — the 20% cap and the new-symbol floor.

Replaces two accidental limits with one deliberate one.

The accidental limits: order quantity is the literal `1` in both order paths,
and `_pre_trade_check` refuses any new entry on an underlying it is already
short. Together those capped a symbol at `strike * 100` and capped the whole
book at one contract per watchlist name — measured, that saturates around
$9,000/month of premium no matter how much cash is added, because the earnings
gate leaves a mean of 9.7 of 14 names sellable and nothing can hold more.

The deliberate limit, as specified by the user:

  new symbol (no exposure yet) : at least 1 contract, up to 20% of capital
  symbol already held          : only what keeps combined exposure under 20%

The asymmetry is the point. A floor on an unheld name forces breadth; a
top-up has to earn its room under the cap. Simulated over the trailing year
that holds ~6.2 names with the largest at 32% of the book, against 3.4 names
at 44% for a flat 1/3 cap.

Denominator decisions pinned below, both of which have a wrong answer that
looks reasonable:

  * TOTAL cash, not free cash. A cap on what is *left* compounds: take 20% of
    free, then 20% of the remainder, and repeated entries converge on 100% of
    the book in one name.
  * Cash, not account equity. Equity includes positions that cannot be sold to
    secure a put -- Vault 76 holds a $51k INTC LEAP -- so an equity
    denominator would authorise collateral the account cannot post.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import live_scanner as ls  # noqa: E402
import options_ledger as ol  # noqa: E402


NVDA = {"signal": "SELL_PUT", "symbol": "NVDA", "strike": 215.0}   # $21,500/ct
META = {"signal": "SELL_PUT", "symbol": "META", "strike": 725.0}   # $72,500/ct


# ---------------------------------------------------------------------------
# The cap constant
# ---------------------------------------------------------------------------

def test_the_cap_is_twenty_percent():
    assert ls.MAX_SYMBOL_FRACTION == 0.20


# ---------------------------------------------------------------------------
# A symbol we do not hold: floor of one, up to the cap
# ---------------------------------------------------------------------------

def test_new_symbol_gets_the_cap_when_capital_is_large():
    """20% of $400k = $80,000 room; $21,500 a contract = 3."""
    assert ls.size_put(NVDA, cash=400_000, exposure=0, available=400_000) == 3


def test_new_symbol_gets_at_least_one_even_though_the_cap_forbids_it():
    """
    Today's book: 20% of $45,838 is $9,167, less than a single $21,500
    contract. Without the floor the cap blocks 12 of 14 symbols outright and
    the overseer stops trading -- measured at $416/mo of premium against
    $1,145/mo with the floor.
    """
    assert ls.size_put(NVDA, cash=45_838, exposure=0, available=45_838) == 1


def test_the_floor_is_one_contract_not_one_dollar_of_cap():
    """100k: cap room $20,000, just under one contract. Still one."""
    assert ls.size_put(NVDA, cash=100_000, exposure=0, available=100_000) == 1


def test_the_floor_never_overrides_the_cash_budget():
    """
    The floor relaxes the concentration cap, never the no-margin rule. With
    $10,000 free there is no way to cash-secure a $21,500 put.
    """
    assert ls.size_put(NVDA, cash=45_838, exposure=0, available=10_000) == 0


def test_an_expensive_name_still_gets_its_single_contract():
    """META at $72,500 a contract, on a book whose 20% is $9,167."""
    assert ls.size_put(META, cash=45_838, exposure=0, available=80_000) == 1
    assert ls.size_put(META, cash=45_838, exposure=0, available=50_000) == 0


# ---------------------------------------------------------------------------
# A symbol we already hold: the cap governs, with no floor
# ---------------------------------------------------------------------------

def test_held_symbol_may_top_up_within_the_cap():
    """$600k: cap $120,000, already $21,500 in → $98,500 room → 4 contracts."""
    assert ls.size_put(NVDA, cash=600_000, exposure=21_500, available=600_000) == 4


def test_held_symbol_is_refused_when_the_cap_is_reached():
    """
    No floor here. This is the gate that replaces "one open position per
    underlying" -- the rule added 2026-09-29 after the book went 99% AMZN.
    """
    assert ls.size_put(NVDA, cash=400_000, exposure=64_500, available=300_000) == 0


def test_held_symbol_gets_only_the_remaining_room():
    """$400k: cap $80,000, $21,500 in → $58,500 → 2 contracts, not 3."""
    assert ls.size_put(NVDA, cash=400_000, exposure=21_500, available=400_000) == 2


def test_exposure_at_or_over_the_cap_yields_nothing():
    for exposure in (80_000, 80_001, 200_000):
        assert ls.size_put(NVDA, cash=400_000, exposure=exposure,
                           available=400_000) == 0, exposure


# ---------------------------------------------------------------------------
# The cash budget always clamps
# ---------------------------------------------------------------------------

def test_available_cash_clamps_below_the_cap():
    """Cap allows 3; only one contract's cash is free."""
    assert ls.size_put(NVDA, cash=400_000, exposure=0, available=30_000) == 1


def test_no_free_cash_means_no_contracts():
    assert ls.size_put(NVDA, cash=400_000, exposure=0, available=0) == 0


def test_negative_available_is_treated_as_none():
    """availableFunds can go negative after an assignment."""
    assert ls.size_put(NVDA, cash=400_000, exposure=0, available=-5_000) == 0


# ---------------------------------------------------------------------------
# Degenerate inputs must not size a trade
# ---------------------------------------------------------------------------

def test_a_missing_or_zero_strike_sizes_nothing():
    for strike in (0, 0.0, None, "", "abc"):
        s = {"signal": "SELL_PUT", "symbol": "NVDA", "strike": strike}
        assert ls.size_put(s, cash=400_000, exposure=0, available=400_000) == 0, strike


def test_zero_or_negative_cash_sizes_nothing():
    """No capital means no floor either -- the floor is not a licence to trade."""
    for cash in (0, -1_000):
        assert ls.size_put(NVDA, cash=cash, exposure=0, available=400_000) == 0, cash


def test_a_non_put_signal_is_left_at_one_contract():
    """
    Sizing covers short puts only. Anything else must come back as exactly
    today's behaviour so wiring this in cannot silently resize a covered call
    or an equity order.
    """
    for sig in ("SELL_CALL", "BUY_CALL", "BUY", "SELL", "BUY_TO_CLOSE", "NONE"):
        s = {"signal": sig, "symbol": "NVDA", "strike": 215.0}
        assert ls.size_put(s, cash=400_000, exposure=0, available=400_000) == 1, sig


# ---------------------------------------------------------------------------
# Per-symbol exposure: ledger positions plus orders not yet filled
# ---------------------------------------------------------------------------

def _open_put(symbol, strike, date, **kw):
    row = {"date": date, "symbol": symbol, "signal": "SELL_PUT",
           "strike": strike, "verdict": "APPROVED"}
    row.update(kw)
    return row


def test_symbol_collateral_counts_only_the_named_underlying():
    rows = [
        _open_put("NVDA", "215", "2026-10-13 10:00:00", contracts=2),
        _open_put("AMZN", "245", "2026-10-13 11:00:00"),
    ]
    assert ol.symbol_collateral(rows, "NVDA") == 43_000.0
    assert ol.symbol_collateral(rows, "AMZN") == 24_500.0
    assert ol.symbol_collateral(rows, "GOOGL") == 0.0


def test_symbol_collateral_ignores_closed_positions():
    rows = [
        _open_put("NVDA", "215", "2026-10-13 10:00:00", contracts=2),
        {"date": "2026-10-20 09:30:00", "symbol": "NVDA", "signal": "SELL_PUT",
         "strike": "215", "verdict": "CLOSED", "contracts": 2,
         "ref": "NVDA_2026-10-13 10:00:00"},
    ]
    assert ol.symbol_collateral(rows, "NVDA") == 0.0


def test_symbol_collateral_still_counts_an_assigned_put():
    rows = [
        _open_put("NVDA", "215", "2026-10-13 10:00:00"),
        {"date": "2026-11-13 16:00:00", "symbol": "NVDA", "signal": "SELL_PUT",
         "strike": "215", "verdict": "ASSIGNED",
         "ref": "NVDA_2026-10-13 10:00:00"},
    ]
    assert ol.symbol_collateral(rows, "NVDA") == 21_500.0


def test_symbol_collateral_is_case_insensitive():
    rows = [_open_put("nvda", "215", "2026-10-13 10:00:00")]
    assert ol.symbol_collateral(rows, "NVDA") == 21_500.0


def test_exposure_includes_orders_placed_but_not_yet_filled(tmp_path, monkeypatch):
    """
    A resting SELL_TO_OPEN is exposure. Omitting it would let one scan size to
    the cap and the next scan size to the cap again before the first fills --
    exactly the double-order hole _pre_trade_check closes for the 1-contract
    case.
    """
    import json
    led = str(tmp_path / "led.csv")
    ol.append_row(led, _open_put("NVDA", "215", "2026-10-13 10:00:00"))
    pend = str(tmp_path / "pending.json")
    with open(pend, "w") as f:
        json.dump([{"signal": "SELL_PUT", "symbol": "NVDA", "strike": 215,
                    "contracts": 2},
                   {"signal": "SELL_PUT", "symbol": "AMZN", "strike": 245}], f)
    monkeypatch.setattr(ls, "OPTION_LEDGER_PATH", led)
    monkeypatch.setattr(ls, "_PENDING_ORDERS_PATH", pend)
    assert ls._symbol_exposure("NVDA") == 21_500 + 43_000
    assert ls._symbol_exposure("AMZN") == 24_500


def test_exposure_is_zero_for_an_untouched_symbol(tmp_path, monkeypatch):
    led = str(tmp_path / "led.csv")
    ol.append_row(led, _open_put("NVDA", "215", "2026-10-13 10:00:00"))
    monkeypatch.setattr(ls, "OPTION_LEDGER_PATH", led)
    monkeypatch.setattr(ls, "_PENDING_ORDERS_PATH", str(tmp_path / "none.json"))
    assert ls._symbol_exposure("GOOGL") == 0.0


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------
# The guards that pinned "nothing calls size_put yet" lived here through the
# commit that added it. The wiring commit replaced them with the invariant that
# actually matters going forward -- with MULTI_CONTRACT_SIZING off the
# effective quantity is 1 whatever the capital -- which is asserted in
# test_multi_contract_orders.py alongside the cover-quantity tests it belongs
# with.


def test_sizing_is_reachable_from_the_order_path():
    """The counterpart: size_put is no longer dead code."""
    import real_overseer as ro
    assert hasattr(ro, "_sized_contracts")
    assert ls.MULTI_CONTRACT_SIZING is True, \
        "sizing is armed as of 2026-10-14"
