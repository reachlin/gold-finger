"""Tests for cash_ledger.py — append-only audit log for all cash movements."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import cash_ledger as cl


@pytest.fixture
def ledger(tmp_path):
    path = str(tmp_path / "cash_ledger.csv")
    return cl.CashLedger(path, starting_capital=30_000.0)


class TestInit:
    def test_starting_balance(self, ledger):
        assert ledger.balance() == 30_000.0

    def test_creates_file_with_header(self, ledger):
        assert os.path.exists(ledger.path)
        with open(ledger.path) as f:
            header = f.readline()
        assert "timestamp" in header and "running_balance" in header

    def test_load_existing_does_not_double_count(self, tmp_path):
        path = str(tmp_path / "cash_ledger.csv")
        cl.CashLedger(path, starting_capital=30_000.0)
        # reopen — must not re-emit STARTING_CAPITAL
        l2 = cl.CashLedger(path, starting_capital=30_000.0)
        assert l2.balance() == 30_000.0


class TestRecord:
    def test_option_sell_credits_premium(self, ledger):
        ledger.record("OPTION_SELL", "KO", 71.0, "KO SELL_PUT premium")
        assert ledger.balance() == 30_071.0

    def test_option_buyback_debits(self, ledger):
        ledger.record("OPTION_SELL", "KO", 71.0, "KO put sold")
        ledger.record("OPTION_BUYBACK", "KO", -27.38, "KO put bought back")
        assert abs(ledger.balance() - 30_043.62) < 0.01

    def test_stock_buy_debits(self, ledger):
        ledger.record("STOCK_BUY", "NVDA", -1_500.0, "buy 5 NVDA @ 300")
        assert ledger.balance() == 28_500.0

    def test_stock_sell_credits(self, ledger):
        ledger.record("STOCK_BUY", "NVDA", -1_500.0, "buy 5 NVDA @ 300")
        ledger.record("STOCK_SELL", "NVDA", 1_650.0, "sell 5 NVDA @ 330")
        assert abs(ledger.balance() - 30_150.0) < 0.01

    def test_running_balance_persists_across_reloads(self, tmp_path):
        path = str(tmp_path / "cash_ledger.csv")
        l1 = cl.CashLedger(path, 30_000.0)
        l1.record("OPTION_SELL", "KO", 71.0, "KO put")
        l2 = cl.CashLedger(path, 30_000.0)
        assert l2.balance() == 30_071.0

    def test_amount_sign_preserved(self, ledger):
        ledger.record("OPTION_SELL", "KO", 71.0, "income")
        ledger.record("OPTION_BUYBACK", "KO", -50.0, "expense")
        rows = ledger.rows()
        amounts = [r["amount"] for r in rows if r["event_type"] != "STARTING_CAPITAL"]
        assert float(amounts[0]) == 71.0
        assert float(amounts[1]) == -50.0


class TestRows:
    def test_rows_returns_all_events(self, ledger):
        ledger.record("OPTION_SELL", "KO", 71.0, "a")
        ledger.record("OPTION_SELL", "PG", 116.0, "b")
        rows = ledger.rows()
        # STARTING_CAPITAL + 2 trades
        assert len(rows) == 3

    def test_rows_have_required_fields(self, ledger):
        ledger.record("OPTION_SELL", "KO", 71.0, "test")
        for row in ledger.rows():
            assert {"timestamp", "event_type", "symbol",
                    "amount", "running_balance", "description"} <= set(row.keys())


class TestOptionEvents:
    def test_assignment_debits_strike_times_100(self, ledger):
        ledger.record("OPTION_SELL", "KO", 71.0, "sold put")
        ledger.record("OPTION_ASSIGNED", "KO", -7_913.0, "assigned 100sh @79.13")
        # premium in, shares out at strike price
        assert abs(ledger.balance() - (30_000 + 71 - 7_913)) < 0.01

    def test_expired_worthless_is_zero_delta(self, ledger):
        ledger.record("OPTION_SELL", "KO", 71.0, "sold put")
        ledger.record("OPTION_EXPIRED", "KO", 0.0, "expired worthless — premium kept")
        assert ledger.balance() == 30_071.0

    def test_called_away_credits_strike_times_100(self, ledger):
        ledger.record("OPTION_SELL", "KO", 71.0, "sold put")
        ledger.record("OPTION_ASSIGNED", "KO", -7_913.0, "assigned")
        ledger.record("OPTION_SELL", "KO", 50.0, "sold call")
        ledger.record("OPTION_CALLED_AWAY", "KO", 8_100.0, "called away @81")
        bal = ledger.balance()
        assert bal == 30_000 + 71 - 7_913 + 50 + 8_100
