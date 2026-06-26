"""
Tests for paper_portfolio.py — paper trading position tracker with balance + logs.

Run: /Users/lincai/anaconda3/envs/gold-finger/bin/python -m pytest schwab/test_paper_portfolio.py -v
"""
import os
import sys
import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))


def _make_portfolio(tmp_path=None, starting_capital=10_000.0):
    from paper_portfolio import PaperPortfolio
    if tmp_path is None:
        base = tempfile.mktemp(suffix=".json")
    else:
        base = str(tmp_path / "paper_trades.json")
    return PaperPortfolio(base, starting_capital=starting_capital)


def _prices(high=135.0, low=128.0, close=133.0, ema20=132.0, ema50=128.0):
    return {"high": high, "low": low, "close": close, "ema20": ema20, "ema50": ema50}


# ---------------------------------------------------------------------------
# Balance tracking
# ---------------------------------------------------------------------------

class TestBalance:
    def test_starting_cash_equals_capital(self, tmp_path):
        p = _make_portfolio(tmp_path, starting_capital=10_000.0)
        assert p.cash == pytest.approx(10_000.0)
        assert p.starting_capital == pytest.approx(10_000.0)

    def test_buy_deducts_cash(self, tmp_path):
        p = _make_portfolio(tmp_path, starting_capital=10_000.0)
        p.open_position("NVDA", entry=100.0, target=120.0, stop=92.0, shares=10)
        assert p.cash == pytest.approx(9_000.0)

    def test_exit_returns_cash(self, tmp_path):
        p = _make_portfolio(tmp_path, starting_capital=10_000.0)
        p.open_position("NVDA", entry=100.0, target=120.0, stop=92.0, shares=10)
        p.check_positions(lambda s: _prices(high=121.0, low=99.0, close=120.0))
        # exit at target $120 × 10 shares = $1200 back
        assert p.cash == pytest.approx(10_200.0)

    def test_balance_persists_across_reload(self, tmp_path):
        from paper_portfolio import PaperPortfolio
        path = str(tmp_path / "trades.json")
        p = PaperPortfolio(path, starting_capital=10_000.0)
        p.open_position("NVDA", entry=100.0, target=120.0, stop=92.0, shares=10)

        p2 = PaperPortfolio(path)
        assert p2.cash == pytest.approx(9_000.0)
        assert p2.starting_capital == pytest.approx(10_000.0)

    def test_insufficient_cash_caps_shares(self, tmp_path):
        p = _make_portfolio(tmp_path, starting_capital=500.0)
        pos = p.open_position("NVDA", entry=100.0, target=120.0, stop=92.0, shares=10)
        # Can only afford 5 shares with $500
        assert pos["shares"] == 5
        assert p.cash == pytest.approx(0.0)

    def test_total_value_is_cash_plus_invested(self, tmp_path):
        p = _make_portfolio(tmp_path, starting_capital=10_000.0)
        p.open_position("NVDA", entry=100.0, target=120.0, stop=92.0, shares=10)
        s = p.summary(current_prices={"NVDA": 110.0})
        # cash=$9000, invested at current=$1100, total=$10100
        assert s["total_value"] == pytest.approx(10_100.0)
        assert s["unrealized_pnl_dollar"] == pytest.approx(100.0)

    def test_total_pnl_includes_closed_and_unrealized(self, tmp_path):
        p = _make_portfolio(tmp_path, starting_capital=10_000.0)
        p.open_position("NVDA", entry=100.0, target=120.0, stop=92.0, shares=10)
        # Close at target: realized +$200
        p.check_positions(lambda s: _prices(high=121.0, low=99.0, close=120.0))
        # Open another, currently up $50
        p.open_position("AMD", entry=100.0, target=120.0, stop=92.0, shares=5)
        s = p.summary(current_prices={"AMD": 110.0})
        assert s["realized_pnl_dollar"] == pytest.approx(200.0)
        assert s["unrealized_pnl_dollar"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# open_position
# ---------------------------------------------------------------------------

class TestOpenPosition:
    def test_adds_to_open_list(self, tmp_path):
        p = _make_portfolio(tmp_path)
        p.open_position("NVDA", entry=130.0, target=156.0, stop=114.0, shares=5)
        assert len(p.open_positions) == 1
        assert p.open_positions[0]["symbol"] == "NVDA"

    def test_stores_all_fields(self, tmp_path):
        p = _make_portfolio(tmp_path)
        pos = p.open_position("NVDA", entry=130.0, target=156.0, stop=114.0, shares=5)
        for key in ("symbol", "entry", "target", "stop", "shares", "cost",
                    "entry_date", "trade_id"):
            assert key in pos, f"Missing field: {key}"

    def test_trade_id_is_unique(self, tmp_path):
        p = _make_portfolio(tmp_path)
        p1 = p.open_position("NVDA", 130.0, 156.0, 114.0, 5)
        p2 = p.open_position("AMD",  120.0, 144.0, 105.0, 4)
        assert p1["trade_id"] != p2["trade_id"]

    def test_multiple_positions(self, tmp_path):
        p = _make_portfolio(tmp_path)
        p.open_position("NVDA", 130.0, 156.0, 114.0, 5)
        p.open_position("AMD",  120.0, 144.0, 105.0, 4)
        assert len(p.open_positions) == 2

    def test_persists_to_file(self, tmp_path):
        from paper_portfolio import PaperPortfolio
        path = str(tmp_path / "trades.json")
        p = PaperPortfolio(path, starting_capital=10_000.0)
        p.open_position("NVDA", 130.0, 156.0, 114.0, 5)
        p2 = PaperPortfolio(path)
        assert len(p2.open_positions) == 1
        assert p2.open_positions[0]["symbol"] == "NVDA"


# ---------------------------------------------------------------------------
# check_positions — exit conditions
# ---------------------------------------------------------------------------

class TestCheckPositions:
    def _open(self, tmp_path, symbol="NVDA", entry=130.0, target=156.0,
              stop=114.0, shares=5):
        p = _make_portfolio(tmp_path)
        p.open_position(symbol, entry, target, stop, shares)
        return p

    def test_target_hit_closes_position(self, tmp_path):
        p = self._open(tmp_path)
        closed = p.check_positions(lambda s: _prices(high=157.0, low=129.0, close=156.5))
        assert len(closed) == 1
        assert closed[0]["exit_reason"] == "target"
        assert closed[0]["exit"] == pytest.approx(156.0)
        assert len(p.open_positions) == 0

    def test_stop_hit_closes_position(self, tmp_path):
        p = self._open(tmp_path)
        closed = p.check_positions(lambda s: _prices(high=131.0, low=113.0, close=114.0))
        assert len(closed) == 1
        assert closed[0]["exit_reason"] == "stop"
        assert closed[0]["exit"] == pytest.approx(114.0)

    def test_trend_end_closes_at_close(self, tmp_path):
        p = self._open(tmp_path)
        closed = p.check_positions(
            lambda s: _prices(high=132.0, low=129.0, close=131.0,
                              ema20=126.0, ema50=130.0)
        )
        assert len(closed) == 1
        assert closed[0]["exit_reason"] == "trend_end"
        assert closed[0]["exit"] == pytest.approx(131.0)

    def test_no_exit_keeps_position_open(self, tmp_path):
        p = self._open(tmp_path)
        closed = p.check_positions(lambda s: _prices(high=133.0, low=129.0, close=132.0))
        assert len(closed) == 0
        assert len(p.open_positions) == 1

    def test_pnl_correct_on_target(self, tmp_path):
        p = _make_portfolio(tmp_path)
        p.open_position("NVDA", entry=100.0, target=120.0, stop=92.0, shares=10)
        closed = p.check_positions(lambda s: _prices(high=121.0, low=99.0, close=120.0))
        assert closed[0]["pnl_pct"] == pytest.approx(20.0, abs=0.01)
        assert closed[0]["pnl_dollar"] == pytest.approx(200.0, abs=0.01)

    def test_pnl_negative_on_stop(self, tmp_path):
        p = _make_portfolio(tmp_path)
        p.open_position("NVDA", entry=100.0, target=120.0, stop=92.0, shares=10)
        closed = p.check_positions(lambda s: _prices(high=99.0, low=91.0, close=92.0))
        assert closed[0]["pnl_dollar"] == pytest.approx(-80.0, abs=0.01)

    def test_price_fetcher_exception_keeps_position(self, tmp_path):
        p = self._open(tmp_path)
        closed = p.check_positions(lambda s: (_ for _ in ()).throw(RuntimeError("down")))
        assert len(closed) == 0
        assert len(p.open_positions) == 1

    def test_max_hold_timeout(self, tmp_path):
        from paper_portfolio import PaperPortfolio
        path = str(tmp_path / "t.json")
        p = PaperPortfolio(path, starting_capital=10_000.0)
        pos = p.open_position("NVDA", 130.0, 156.0, 114.0, 5)
        pos["entry_date"] = (datetime.now() - timedelta(days=31)).isoformat()
        p._save()
        closed = p.check_positions(lambda s: _prices(high=133.0, low=129.0, close=132.0))
        assert closed[0]["exit_reason"] == "timeout"

    def test_multiple_positions_independent(self, tmp_path):
        p = _make_portfolio(tmp_path)
        p.open_position("NVDA", 130.0, 156.0, 114.0, 5)
        p.open_position("AMD",  120.0, 144.0, 105.0, 4)
        def fetcher(symbol):
            if symbol == "NVDA":
                return _prices(high=157.0, low=129.0, close=156.5)
            return _prices(high=122.0, low=119.0, close=121.0)
        closed = p.check_positions(fetcher)
        assert len(closed) == 1
        assert closed[0]["symbol"] == "NVDA"
        assert len(p.open_positions) == 1


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_empty_portfolio(self, tmp_path):
        p = _make_portfolio(tmp_path)
        s = p.summary()
        assert s["open_count"] == 0
        assert s["closed_count"] == 0
        assert s["realized_pnl_dollar"] == 0.0
        assert s["unrealized_pnl_dollar"] == 0.0

    def test_win_rate_and_pnl(self, tmp_path):
        from paper_portfolio import PaperPortfolio
        path = str(tmp_path / "t.json")
        p = PaperPortfolio(path, starting_capital=10_000.0)
        p.open_position("NVDA", 100.0, 120.0, 92.0, 10)
        p.open_position("AMD",  100.0, 120.0, 92.0, 5)
        def fetcher(symbol):
            if symbol == "NVDA":
                return _prices(high=121.0, low=99.0, close=120.0)
            return _prices(high=99.0, low=91.0, close=92.0)
        p.check_positions(fetcher)
        s = p.summary()
        assert s["closed_count"] == 2
        assert s["realized_pnl_dollar"] == pytest.approx(200.0 - 40.0, abs=0.01)
        assert s["win_rate"] == pytest.approx(50.0, abs=0.1)


# ---------------------------------------------------------------------------
# JSONL event log
# ---------------------------------------------------------------------------

class TestEventLog:
    def test_log_file_created_on_first_event(self, tmp_path):
        p = _make_portfolio(tmp_path)
        p.open_position("NVDA", 130.0, 156.0, 114.0, 5)
        log_path = p.log_path
        assert os.path.exists(log_path)

    def test_buy_event_logged(self, tmp_path):
        p = _make_portfolio(tmp_path)
        p.open_position("NVDA", 130.0, 156.0, 114.0, 5)
        events = _read_log(p.log_path)
        buy_events = [e for e in events if e["event"] == "BUY"]
        assert len(buy_events) == 1
        assert buy_events[0]["symbol"] == "NVDA"
        assert buy_events[0]["shares"] == 5

    def test_exit_event_logged(self, tmp_path):
        p = _make_portfolio(tmp_path)
        p.open_position("NVDA", 100.0, 120.0, 92.0, 10)
        p.check_positions(lambda s: _prices(high=121.0, low=99.0, close=120.0))
        events = _read_log(p.log_path)
        exit_events = [e for e in events if e["event"] == "EXIT"]
        assert len(exit_events) == 1
        assert exit_events[0]["exit_reason"] == "target"
        assert "pnl_dollar" in exit_events[0]

    def test_scan_event_logged(self, tmp_path):
        p = _make_portfolio(tmp_path)
        p.log_scan(scan_num=1, symbols_scanned=10, signals_found=0)
        events = _read_log(p.log_path)
        scan_events = [e for e in events if e["event"] == "SCAN"]
        assert len(scan_events) == 1
        assert scan_events[0]["scan_num"] == 1

    def test_signal_event_logged(self, tmp_path):
        p = _make_portfolio(tmp_path)
        p.log_signal("NVDA", entry=130.0, target=156.0, stop=114.0,
                     rsi=42.0, adx=18.0, verdict="APPROVED")
        events = _read_log(p.log_path)
        sig_events = [e for e in events if e["event"] == "SIGNAL"]
        assert len(sig_events) == 1
        assert sig_events[0]["verdict"] == "APPROVED"

    def test_log_entries_have_timestamps(self, tmp_path):
        p = _make_portfolio(tmp_path)
        p.open_position("NVDA", 130.0, 156.0, 114.0, 5)
        events = _read_log(p.log_path)
        for e in events:
            assert "ts" in e

    def test_log_survives_multiple_sessions(self, tmp_path):
        from paper_portfolio import PaperPortfolio
        path = str(tmp_path / "trades.json")
        p1 = PaperPortfolio(path, starting_capital=10_000.0)
        p1.open_position("NVDA", 130.0, 156.0, 114.0, 5)
        p2 = PaperPortfolio(path)
        p2.open_position("AMD", 120.0, 144.0, 105.0, 4)
        events = _read_log(p2.log_path)
        buy_events = [e for e in events if e["event"] == "BUY"]
        assert len(buy_events) == 2


def _read_log(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
