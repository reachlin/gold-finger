"""
Tests for vault20/vault20.py — manual position tracker.

Run: /Users/lincai/anaconda3/envs/gold-finger/bin/python -m pytest vault20/test_vault20.py -v
"""
import sys, os, json, tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def tmp_path_file():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(b'{"open": [], "closed": []}')
        return f.name


@pytest.fixture
def ledger(tmp_path_file):
    from vault20.vault20 import Ledger
    return Ledger(tmp_path_file)


class TestLedgerAdd:
    def test_add_position(self, ledger):
        ledger.add("AAPL", shares=10, entry=185.50, entry_date="2026-06-28")
        assert len(ledger.open) == 1
        assert ledger.open[0]["symbol"] == "AAPL"

    def test_add_stores_required_fields(self, ledger):
        ledger.add("MSFT", shares=5, entry=420.0, entry_date="2026-06-28")
        pos = ledger.open[0]
        for field in ("symbol", "shares", "entry", "entry_date"):
            assert field in pos

    def test_add_optional_target_and_stop(self, ledger):
        ledger.add("NVDA", shares=2, entry=190.0, entry_date="2026-06-28",
                   target=220.0, stop=175.0)
        pos = ledger.open[0]
        assert pos["target"] == 220.0
        assert pos["stop"]   == 175.0

    def test_add_optional_note(self, ledger):
        ledger.add("KO", shares=20, entry=82.0, entry_date="2026-06-28",
                   note="dividend play")
        assert ledger.open[0]["note"] == "dividend play"

    def test_add_persists_to_file(self, tmp_path_file):
        from vault20.vault20 import Ledger
        Ledger(tmp_path_file).add("IBM", 3, 271.0, "2026-06-28")
        data = json.loads(open(tmp_path_file).read())
        assert len(data["open"]) == 1
        assert data["open"][0]["symbol"] == "IBM"

    def test_add_duplicate_symbol_raises(self, ledger):
        ledger.add("AAPL", 10, 185.0, "2026-06-28")
        with pytest.raises(ValueError, match="already open"):
            ledger.add("AAPL", 5, 190.0, "2026-06-28")


class TestLedgerClose:
    def test_close_moves_to_closed(self, ledger):
        ledger.add("AAPL", 10, 185.0, "2026-06-27")
        ledger.close("AAPL", exit_price=195.0, exit_date="2026-06-28")
        assert len(ledger.open)   == 0
        assert len(ledger.closed) == 1

    def test_close_calculates_pnl(self, ledger):
        ledger.add("AAPL", 10, 185.0, "2026-06-27")
        ledger.close("AAPL", exit_price=195.0, exit_date="2026-06-28")
        t = ledger.closed[0]
        assert abs(t["pnl_dollar"] - 100.0) < 0.01   # 10 * (195 - 185)
        assert abs(t["pnl_pct"]   - 5.405) < 0.01    # 10/185 * 100

    def test_close_unknown_symbol_raises(self, ledger):
        with pytest.raises(ValueError, match="not found"):
            ledger.close("XYZ", 100.0, "2026-06-28")

    def test_close_loss_position(self, ledger):
        ledger.add("INTC", 10, 30.0, "2026-06-27")
        ledger.close("INTC", exit_price=25.0, exit_date="2026-06-28")
        assert ledger.closed[0]["pnl_dollar"] == pytest.approx(-50.0)

    def test_close_persists(self, tmp_path_file):
        from vault20.vault20 import Ledger
        l = Ledger(tmp_path_file)
        l.add("PG", 5, 149.0, "2026-06-27")
        l.close("PG", 155.0, "2026-06-28")
        data = json.loads(open(tmp_path_file).read())
        assert len(data["open"])   == 0
        assert len(data["closed"]) == 1


class TestLedgerSummary:
    def test_summary_total_cost(self, ledger):
        ledger.add("AAPL", 10, 185.0, "2026-06-28")
        ledger.add("KO",   20, 82.0,  "2026-06-28")
        s = ledger.summary()
        assert s["total_cost"] == pytest.approx(10 * 185.0 + 20 * 82.0)

    def test_summary_realized_pnl(self, ledger):
        ledger.add("AAPL", 10, 185.0, "2026-06-27")
        ledger.close("AAPL", 195.0, "2026-06-28")
        s = ledger.summary()
        assert s["realized_pnl"] == pytest.approx(100.0)

    def test_summary_with_prices(self, ledger):
        ledger.add("AAPL", 10, 185.0, "2026-06-28")
        s = ledger.summary(prices={"AAPL": 190.0})
        assert s["unrealized_pnl"] == pytest.approx(50.0)

    def test_empty_ledger_summary(self, ledger):
        s = ledger.summary()
        assert s["open_count"]    == 0
        assert s["realized_pnl"]  == 0.0
        assert s["unrealized_pnl"] == 0.0
