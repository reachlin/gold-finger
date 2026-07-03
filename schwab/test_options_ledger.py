"""
Tests for options_ledger.py — paper options ledger lifecycle.

Covers:
  - CSV read/write + migration of legacy header (no pnl/ref columns)
  - open/assigned/closed state machine
  - committed collateral (assigned puts stay committed)
  - expiry processing: expired worthless, assignment, early exit,
    covered call expiry, called away
"""
import os
import sys
import csv
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import options_ledger as ol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _put_row(symbol="KO", day="2026-06-01", strike=79.13, premium_sh=0.71,
             dte=30, hv=24.6, verdict="APPROVED", **extra):
    row = {
        "date": f"{day} 10:00:00",
        "symbol": symbol,
        "signal": "SELL_PUT",
        "close": round(strike / 0.95, 2),
        "strike": strike,
        "premium_sh": premium_sh,
        "premium_ct": round(premium_sh * 100, 2),
        "premium_pct": 0.9,
        "dte": dte,
        "hv": hv,
        "adx": 15.0,
        "regime": "WASTELAND",
        "verdict": verdict,
        "reason": "test put",
    }
    row.update(extra)
    return row


def _call_row(symbol="KO", day="2026-06-01", strike=90.0, premium_sh=0.55,
              dte=30, hv=24.6, verdict="APPROVED", **extra):
    row = _put_row(symbol=symbol, day=day, strike=strike,
                   premium_sh=premium_sh, dte=dte, hv=hv, verdict=verdict)
    row["signal"] = "SELL_CALL"
    row["reason"] = "test call"
    row.update(extra)
    return row


def _write_ledger(path, rows):
    for r in rows:
        ol.append_row(path, r)


# ---------------------------------------------------------------------------
# CSV round-trip + migration
# ---------------------------------------------------------------------------

class TestCsvRoundTrip:
    def test_append_and_read(self, tmp_path):
        path = str(tmp_path / "ledger.csv")
        ol.append_row(path, _put_row())
        rows = ol.read_rows(path)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "KO"
        assert rows[0]["verdict"] == "APPROVED"
        # New columns exist even when not set
        assert "pnl" in rows[0]
        assert "ref" in rows[0]

    def test_read_missing_file(self, tmp_path):
        assert ol.read_rows(str(tmp_path / "nope.csv")) == []

    def test_legacy_header_migrated_on_append(self, tmp_path):
        """A ledger written before pnl/ref existed must survive an append."""
        path = str(tmp_path / "ledger.csv")
        legacy_fields = [f for f in ol.FIELDNAMES if f not in ("pnl", "ref")]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=legacy_fields)
            w.writeheader()
            legacy = {k: v for k, v in _put_row().items() if k in legacy_fields}
            w.writerow(legacy)

        ol.append_row(path, _put_row(symbol="PG", day="2026-06-02"))
        rows = ol.read_rows(path)
        assert len(rows) == 2
        assert rows[0]["symbol"] == "KO"
        assert rows[0]["pnl"] == ""          # legacy row backfilled empty
        assert rows[1]["symbol"] == "PG"


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class TestOpenOptions:
    def test_approved_is_open(self, tmp_path):
        path = str(tmp_path / "l.csv")
        _write_ledger(path, [_put_row()])
        open_rows = ol.open_options(ol.read_rows(path))
        assert len(open_rows) == 1

    def test_blocked_rows_never_open(self, tmp_path):
        path = str(tmp_path / "l.csv")
        _write_ledger(path, [
            _put_row(verdict="BUDGET_BLOCK"),
            _put_row(verdict="SKIPPED", day="2026-06-02"),
            _put_row(verdict="RISKOFF_BLOCK", day="2026-06-03"),
        ])
        assert ol.open_options(ol.read_rows(path)) == []

    def test_closed_ref_removes_open(self, tmp_path):
        path = str(tmp_path / "l.csv")
        opening = _put_row()
        key = ol.row_key(opening)
        _write_ledger(path, [
            opening,
            _put_row(day="2026-07-01", verdict="CLOSED", ref=key, pnl=71.0),
        ])
        assert ol.open_options(ol.read_rows(path)) == []

    def test_assigned_not_open_but_committed(self, tmp_path):
        path = str(tmp_path / "l.csv")
        opening = _put_row()
        key = ol.row_key(opening)
        _write_ledger(path, [
            opening,
            _put_row(day="2026-07-01", verdict="ASSIGNED", ref=key),
        ])
        rows = ol.read_rows(path)
        assert ol.open_options(rows) == []
        # Cash became shares — collateral still tied up
        assert ol.committed_collateral(rows) == pytest.approx(79.13 * 100)

    def test_symbol_filter(self, tmp_path):
        path = str(tmp_path / "l.csv")
        _write_ledger(path, [_put_row(), _put_row(symbol="PG", day="2026-06-02")])
        rows = ol.read_rows(path)
        assert len(ol.open_options(rows, symbol="KO")) == 1
        assert len(ol.open_options(rows)) == 2


class TestCommittedCollateral:
    def test_open_put_committed(self, tmp_path):
        path = str(tmp_path / "l.csv")
        _write_ledger(path, [_put_row(strike=100.0)])
        assert ol.committed_collateral(ol.read_rows(path)) == 10_000.0

    def test_closed_put_released(self, tmp_path):
        path = str(tmp_path / "l.csv")
        opening = _put_row(strike=100.0)
        _write_ledger(path, [
            opening,
            _put_row(day="2026-07-01", verdict="CLOSED",
                     ref=ol.row_key(opening)),
        ])
        assert ol.committed_collateral(ol.read_rows(path)) == 0.0

    def test_calls_require_no_cash(self, tmp_path):
        path = str(tmp_path / "l.csv")
        _write_ledger(path, [_call_row(strike=90.0)])
        assert ol.committed_collateral(ol.read_rows(path)) == 0.0

    def test_duplicate_puts_stack(self, tmp_path):
        """Duplicate positions on the same symbol are allowed — each commits."""
        path = str(tmp_path / "l.csv")
        _write_ledger(path, [
            _put_row(strike=100.0, day="2026-06-01"),
            _put_row(strike=98.0, day="2026-06-02"),
        ])
        assert ol.committed_collateral(ol.read_rows(path)) == 19_800.0


# ---------------------------------------------------------------------------
# Expiry / assignment / early-exit processing
# ---------------------------------------------------------------------------

def _fetcher(close, hv=0.25):
    return lambda sym: {"close": close, "hv": hv}


class TestProcess:
    def test_put_expires_worthless(self, tmp_path):
        ledger = str(tmp_path / "l.csv")
        holdings = str(tmp_path / "h.json")
        _write_ledger(ledger, [_put_row(strike=79.0, premium_sh=0.71, dte=30)])
        today = date(2026, 7, 2)   # open 2026-06-01 + 30d = 2026-07-01 → expired

        events = ol.process_expirations(ledger, holdings, _fetcher(close=85.0),
                                        today=today)
        assert len(events) == 1
        assert events[0]["action"] == "put_expired"
        assert events[0]["pnl"] == pytest.approx(71.0)

        rows = ol.read_rows(ledger)
        assert ol.open_options(rows) == []
        assert ol.committed_collateral(rows) == 0.0
        assert ol.load_holdings(holdings) == {}

    def test_put_assigned_creates_holding(self, tmp_path):
        ledger = str(tmp_path / "l.csv")
        holdings = str(tmp_path / "h.json")
        _write_ledger(ledger, [_put_row(strike=79.0, premium_sh=0.71, dte=30)])
        today = date(2026, 7, 2)

        events = ol.process_expirations(ledger, holdings, _fetcher(close=75.0),
                                        today=today)
        assert events[0]["action"] == "put_assigned"

        h = ol.load_holdings(holdings)
        assert "KO" in h
        assert h["KO"]["cost_basis"] == pytest.approx(79.0 - 0.71)
        assert h["KO"]["contracts"] == 1

        # Collateral stays committed — cash became shares
        rows = ol.read_rows(ledger)
        assert ol.committed_collateral(rows) == pytest.approx(7_900.0)

    def test_put_early_exit_when_value_decays(self, tmp_path):
        """Deep-OTM put with most DTE burned → BS value ≈ 0 → early exit."""
        ledger = str(tmp_path / "l.csv")
        holdings = str(tmp_path / "h.json")
        _write_ledger(ledger, [_put_row(strike=79.0, premium_sh=0.71, dte=30)])
        today = date(2026, 6, 28)   # 3 days left, far from expiry cutoff

        events = ol.process_expirations(ledger, holdings,
                                        _fetcher(close=95.0, hv=0.20),
                                        today=today)
        assert len(events) == 1
        assert events[0]["action"] == "early_exit"
        assert events[0]["pnl"] > 0
        assert ol.open_options(ol.read_rows(ledger)) == []

    def test_no_early_exit_when_value_high(self, tmp_path):
        """ATM put with high vol keeps its value — no exit before expiry."""
        ledger = str(tmp_path / "l.csv")
        holdings = str(tmp_path / "h.json")
        _write_ledger(ledger, [_put_row(strike=79.0, premium_sh=0.71, dte=30)])
        today = date(2026, 6, 10)

        events = ol.process_expirations(ledger, holdings,
                                        _fetcher(close=79.0, hv=0.60),
                                        today=today)
        assert events == []
        assert len(ol.open_options(ol.read_rows(ledger))) == 1

    def test_call_expires_worthless_keeps_holding(self, tmp_path):
        ledger = str(tmp_path / "l.csv")
        holdings = str(tmp_path / "h.json")
        ol.save_holdings(holdings, {"KO": {"contracts": 1, "cost_basis": 78.29,
                                           "put_key": "KO_2026-05-01 10:00:00",
                                           "assigned_date": "2026-06-01"}})
        _write_ledger(ledger, [_call_row(strike=90.0, premium_sh=0.55, dte=30)])
        today = date(2026, 7, 2)

        events = ol.process_expirations(ledger, holdings, _fetcher(close=85.0),
                                        today=today)
        assert events[0]["action"] == "call_expired"
        assert events[0]["pnl"] == pytest.approx(55.0)
        # Shares still held — ready for the next call
        assert "KO" in ol.load_holdings(holdings)

    def test_called_away_closes_cycle(self, tmp_path):
        ledger = str(tmp_path / "l.csv")
        holdings = str(tmp_path / "h.json")
        put_key = "KO_2026-05-01 10:00:00"
        ol.save_holdings(holdings, {"KO": {"contracts": 1, "cost_basis": 78.29,
                                           "put_key": put_key,
                                           "assigned_date": "2026-06-01"}})
        _write_ledger(ledger, [_call_row(strike=90.0, premium_sh=0.55, dte=30)])
        today = date(2026, 7, 2)

        events = ol.process_expirations(ledger, holdings, _fetcher(close=95.0),
                                        today=today)
        assert events[0]["action"] == "called_away"
        # (90 - 78.29) * 100 shares + 55 call premium
        assert events[0]["pnl"] == pytest.approx((90.0 - 78.29) * 100 + 55.0)
        assert ol.load_holdings(holdings) == {}

        # The originating put's collateral is finally released
        rows = ol.read_rows(ledger)
        closed_refs = [r["ref"] for r in rows if r["verdict"] == "CLOSED"]
        assert put_key in closed_refs

    def test_fetch_failure_leaves_position_open(self, tmp_path):
        ledger = str(tmp_path / "l.csv")
        holdings = str(tmp_path / "h.json")
        _write_ledger(ledger, [_put_row()])

        def broken(sym):
            raise RuntimeError("no quote")

        events = ol.process_expirations(ledger, holdings, broken,
                                        today=date(2026, 7, 2))
        assert events == []
        assert len(ol.open_options(ol.read_rows(ledger))) == 1
