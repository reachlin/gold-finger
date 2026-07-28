"""Tests for real_overseer — the fully-automated real-money overseer.

Written before the implementation (TDD). Covers the bugs found in the
2026-07-28 audit of auto_overseer.py:
  - real fills never booked to the options/cash ledgers (paper-gated)
  - _pre_trade_check trusting cashBalance (includes locked collateral)
  - OPTION_BUYBACK recorded as +P&L instead of -cost paid
  - enteredTime parsed as epoch ms when Schwab sends ISO strings
  - no assignment handling on reconcile
  - hardcoded $30,000 cash in decide()
"""
import json
import os
import sys
import pytest
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Ported guarantees from auto_overseer (prompt/parse/occ/market check)
# ---------------------------------------------------------------------------

def test_build_prompt_contains_signal_fields():
    from schwab.real_overseer import build_prompt
    signal = {"symbol": "KO", "signal": "SELL_PUT", "strike": "79.13",
              "premium": 0.71, "hv": 24.6, "adx": 15.1}
    prompt = build_prompt(signal, {"available": 22087, "required": 7913}, {})
    assert "KO" in prompt
    assert "SELL_PUT" in prompt
    assert "79.13" in prompt


def test_overseer_system_allows_duplicates():
    from schwab.real_overseer import OVERSEER_SYSTEM
    assert "ALLOWED" in OVERSEER_SYSTEM


def test_parse_yes_response():
    from schwab.real_overseer import parse_llm_response
    d, r = parse_llm_response('{"decision": "yes", "reason": "good setup"}')
    assert d == "yes" and r == "good setup"


def test_parse_malformed_falls_back_to_no():
    from schwab.real_overseer import parse_llm_response
    d, _ = parse_llm_response("I think you should buy!")
    assert d == "no"


def test_build_occ_symbol_put():
    from schwab.real_overseer import build_occ_symbol
    occ = build_occ_symbol("KO", date(2026, 7, 31), "SELL_PUT", 79.13)
    assert occ == "KO    260731P00079130"


def test_parse_occ_symbol_roundtrip():
    from schwab.real_overseer import build_occ_symbol, parse_occ_symbol
    occ = build_occ_symbol("NVDA", date(2026, 8, 21), "SELL_PUT", 195.0)
    root, expiry, put_call, strike = parse_occ_symbol(occ)
    assert root == "NVDA"
    assert expiry == date(2026, 8, 21)
    assert put_call == "P"
    assert strike == 195.0


def test_parse_occ_symbol_normalized_no_spaces():
    """Symbols normalized with .replace(" ", "") have variable-length roots —
    fixed-offset slicing crashed reconcile on the first live cycle
    ("unconverted data remains: P0")."""
    from schwab.real_overseer import parse_occ_symbol
    root, expiry, put_call, strike = parse_occ_symbol("NVDA260821P00195000")
    assert root == "NVDA"
    assert expiry == date(2026, 8, 21)
    assert put_call == "P"
    assert strike == 195.0
    root, _, _, strike = parse_occ_symbol("KO260814P00078000")
    assert root == "KO" and strike == 78.0


def test_market_open_regular_weekday():
    from schwab.real_overseer import market_open_on
    assert market_open_on(date(2026, 7, 28)) is True


def test_market_closed_weekend():
    from schwab.real_overseer import market_open_on
    assert market_open_on(date(2026, 7, 26)) is False


# ---------------------------------------------------------------------------
# available_funds — must NOT trust cashBalance (includes locked collateral)
# ---------------------------------------------------------------------------

def test_available_funds_prefers_available_over_cash_balance():
    from schwab.real_overseer import available_funds
    balances = {"cashBalance": 30000.0,
                "availableFundsNonMarginableTrade": 2700.0,
                "availableFunds": 2700.0}
    assert available_funds(balances) == 2700.0


def test_available_funds_falls_back_when_missing():
    from schwab.real_overseer import available_funds
    assert available_funds({"availableFunds": 5000.0}) == 5000.0
    assert available_funds({}) is None


def test_available_funds_skips_zero_when_sibling_has_value():
    """This account reports availableFundsNonMarginableTrade=0.0 alongside a
    real availableFunds value — a 0.0 must not shadow it (observed live:
    available=$0 while Schwab showed $11,063). All-zero is genuine $0."""
    from schwab.real_overseer import available_funds
    balances = {"availableFundsNonMarginableTrade": 0.0,
                "availableFunds": 11063.0}
    assert available_funds(balances) == 11063.0
    assert available_funds({"availableFundsNonMarginableTrade": 0.0,
                            "availableFunds": 0.0}) == 0.0


# ---------------------------------------------------------------------------
# parse_entered_time — Schwab sends ISO8601 strings, not epoch ms
# ---------------------------------------------------------------------------

def test_parse_entered_time_iso_string():
    from schwab.real_overseer import parse_entered_time
    dt = parse_entered_time({"enteredTime": "2026-07-28T13:00:08+0000"})
    assert dt is not None
    assert dt.year == 2026 and dt.month == 7 and dt.day == 28
    assert dt.tzinfo is not None


def test_parse_entered_time_missing_returns_none():
    from schwab.real_overseer import parse_entered_time
    assert parse_entered_time({}) is None
    assert parse_entered_time({"enteredTime": "garbage"}) is None


# ---------------------------------------------------------------------------
# Booking — single source of truth for ledger + cash writes
# ---------------------------------------------------------------------------

@pytest.fixture
def ledgers(tmp_path):
    """Fresh options ledger CSV path + CashLedger seeded with $30,000."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "schwab"))
    from cash_ledger import CashLedger
    opt_path = str(tmp_path / "options_ledger.csv")
    cash = CashLedger(str(tmp_path / "cash_ledger.csv"), starting_capital=30_000.0)
    return opt_path, cash


def _opening_row(opt_path):
    """Write one APPROVED SELL_PUT row and return it as read back."""
    import options_ledger as ol
    ol.append_row(opt_path, {
        "date": "2026-07-20 22:38:25", "symbol": "KO", "signal": "SELL_PUT",
        "close": 82.14, "strike": 78.0, "premium_sh": 0.65, "premium_ct": 65.0,
        "premium_pct": 0.77, "dte": 25, "hv": 27.7, "adx": 15.9,
        "confidence": 77, "verdict": "APPROVED", "reason": "test open",
    })
    return ol.read_rows(opt_path)[0]


def test_book_open_fill_writes_approved_row_and_premium(ledgers):
    """Real-mode fills must be booked: APPROVED row + OPTION_SELL credit.
    (auto_overseer relied on the scanner's paper-gated logging — real fills
    were never recorded at all.)"""
    from schwab.real_overseer import book_open_fill
    import options_ledger as ol
    opt_path, cash = ledgers
    s = {"symbol": "KO", "signal": "SELL_PUT", "close": 82.14, "strike": 78.0,
         "premium_pct": 0.77, "dte": 25, "hv": 27.7, "adx": 15.9,
         "confidence": 77, "regime": "RECLAMATION"}
    book_open_fill(opt_path, cash, s, fill_price=0.66)

    rows = ol.read_rows(opt_path)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "APPROVED"
    assert float(rows[0]["premium_sh"]) == 0.66
    assert float(rows[0]["premium_ct"]) == 66.0
    assert len(ol.open_options(rows)) == 1          # counts as open collateral

    last = cash.rows()[-1]
    assert last["event_type"] == "OPTION_SELL"
    assert float(last["amount"]) == 66.0
    assert float(last["running_balance"]) == 30_066.0


def test_book_buyback_debits_cost_not_pnl(ledgers):
    """OPTION_BUYBACK must record -(fill x 100) — the cash actually paid —
    not +P&L. (Recording +$40 P&L on the KO close inflated the ledger $65
    above Schwab's real balance, observed 2026-07-28.)"""
    from schwab.real_overseer import book_buyback
    import options_ledger as ol
    opt_path, cash = ledgers
    opening = _opening_row(opt_path)

    book_buyback(opt_path, cash, opening, fill_price=0.25)

    rows = ol.read_rows(opt_path)
    closed = rows[-1]
    assert closed["verdict"] == "CLOSED"
    assert float(closed["pnl"]) == 40.0             # (0.65 - 0.25) x 100
    assert len(ol.open_options(rows)) == 0

    last = cash.rows()[-1]
    assert last["event_type"] == "OPTION_BUYBACK"
    assert float(last["amount"]) == -25.0           # cost paid, negative
    assert float(last["running_balance"]) == 29_975.0


def test_book_expired_zero_cash_delta(ledgers):
    from schwab.real_overseer import book_expired
    import options_ledger as ol
    opt_path, cash = ledgers
    opening = _opening_row(opt_path)

    book_expired(opt_path, cash, opening)

    rows = ol.read_rows(opt_path)
    assert rows[-1]["verdict"] == "CLOSED"
    assert float(rows[-1]["pnl"]) == 65.0           # full premium kept
    last = cash.rows()[-1]
    assert last["event_type"] == "OPTION_EXPIRED"   # not "OPTION_EXPIRE"
    assert float(last["amount"]) == 0.0             # premium already credited


def test_book_assigned_debits_collateral_and_updates_holdings(ledgers, tmp_path):
    """Assignment is a core wheel event: ASSIGNED row, -(strike x 100) cash,
    and shares recorded in wheel holdings at (strike - premium) cost basis.
    auto_overseer booked this as 'expired worthless' — wrong."""
    from schwab.real_overseer import book_assigned
    import options_ledger as ol
    opt_path, cash = ledgers
    holdings_path = str(tmp_path / "wheel_holdings.json")
    opening = _opening_row(opt_path)

    book_assigned(opt_path, cash, opening, holdings_path)

    rows = ol.read_rows(opt_path)
    assert rows[-1]["verdict"] == "ASSIGNED"
    states = ol.open_options(rows)
    assert len(states) == 0                          # no longer an open option

    last = cash.rows()[-1]
    assert last["event_type"] == "OPTION_ASSIGNED"
    assert float(last["amount"]) == -7800.0          # strike x 100

    with open(holdings_path) as f:
        holdings = json.load(f)
    assert holdings["KO"]["contracts"] == 1
    assert holdings["KO"]["cost_basis"] == pytest.approx(77.35)  # 78 - 0.65


# ---------------------------------------------------------------------------
# decide() — real cash from Schwab sync, never a hardcoded constant
# ---------------------------------------------------------------------------

def test_decide_uses_schwab_available_funds(monkeypatch):
    from schwab.real_overseer import RealOverseer
    import live_scanner as scanner

    ov = RealOverseer.__new__(RealOverseer)          # skip LLM init
    ov.llm = MagicMock()
    ov.llm.provider = "test"; ov.llm.model = "test"
    ov.llm.chat.return_value = '{"decision": "no", "reason": "test"}'

    monkeypatch.setattr(scanner, "_schwab_available", 11063.0, raising=False)
    monkeypatch.setattr(scanner, "_current_kronos_cache", {}, raising=False)
    monkeypatch.setattr(scanner, "_current_scan_signals", [], raising=False)

    captured = {}
    def fake_build_prompt(signal, portfolio_state, kronos, **kw):
        captured.update(portfolio_state)
        return "prompt"
    import schwab.real_overseer as ro
    monkeypatch.setattr(ro, "build_prompt", fake_build_prompt)

    verdict = ov.decide({"symbol": "KO", "signal": "SELL_PUT", "strike": 78.0})
    assert verdict == "n"
    assert captured["available"] == 11063.0          # not 30000 - committed


# ---------------------------------------------------------------------------
# Entry point — real-only, no paper/semi flags, scan hook always wired
# ---------------------------------------------------------------------------

def test_no_paper_or_semi_flags():
    from schwab.real_overseer import build_arg_parser
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--paper"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--semi"])
    args = parser.parse_args(["--provider", "deepseek"])
    assert args.provider == "deepseek"


def test_main_requires_really_real(monkeypatch, capsys):
    """Fully-automated real overseer must refuse to start without the
    REALLY_REAL=true safety gate — not silently run order-less."""
    import schwab.real_overseer as ro
    monkeypatch.setattr(sys, "argv", ["real_overseer.py"])
    monkeypatch.delenv("REALLY_REAL", raising=False)
    with pytest.raises(SystemExit):
        ro.main()


def test_main_wires_decision_and_scan_hook(monkeypatch):
    import schwab.real_overseer as ro
    import live_scanner as scanner

    monkeypatch.setattr(sys, "argv", ["real_overseer.py"])
    monkeypatch.setenv("REALLY_REAL", "true")
    fake = MagicMock()
    fake.llm.provider = "test"
    monkeypatch.setattr(ro, "RealOverseer", lambda **kw: fake)
    monkeypatch.setattr(ro, "check_market_open", lambda *a, **k: False)

    decision, hooks = [], []
    monkeypatch.setattr(scanner, "set_decision_fn", lambda fn: decision.append(fn))
    monkeypatch.setattr(scanner, "set_scan_hook_fn", lambda fn: hooks.append(fn))
    monkeypatch.setattr(scanner, "_send_slack", lambda *a, **k: None)

    ro.main()

    assert decision == [fake.decide]
    assert hooks == [fake.scan_hook]
