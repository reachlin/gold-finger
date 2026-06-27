"""
Tests for signal_verifier.py — data-fetch utilities.

All external calls (yfinance, Schwab client, feedparser) are mocked.

Run: /Users/lincai/anaconda3/envs/gold-finger/bin/python -m pytest schwab/test_signal_verifier.py -v
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# check_vix
# ---------------------------------------------------------------------------

class TestCheckVix:
    def test_low_vix_returns_pass(self):
        from signal_verifier import check_vix
        assert check_vix(15.0)["status"] == "PASS"

    def test_elevated_vix_returns_warn(self):
        from signal_verifier import check_vix
        assert check_vix(22.0)["status"] == "WARN"

    def test_high_vix_returns_block(self):
        from signal_verifier import check_vix
        assert check_vix(35.0)["status"] == "BLOCK"

    def test_boundary_vix_20_is_warn(self):
        from signal_verifier import check_vix
        assert check_vix(20.0)["status"] == "WARN"

    def test_boundary_vix_30_is_block(self):
        from signal_verifier import check_vix
        assert check_vix(30.0)["status"] == "BLOCK"

    def test_returns_vix_value(self):
        from signal_verifier import check_vix
        result = check_vix(18.5)
        assert result["vix"] == pytest.approx(18.5)

    @patch("signal_verifier.yf")
    def test_fetch_vix_calls_yfinance(self, mock_yf):
        from signal_verifier import fetch_vix
        ticker = MagicMock()
        ticker.fast_info = {"lastPrice": 18.5}
        mock_yf.Ticker.return_value = ticker
        val = fetch_vix()
        assert val == pytest.approx(18.5)
        mock_yf.Ticker.assert_called_once_with("^VIX")


# ---------------------------------------------------------------------------
# check_earnings_proximity — via Schwab client
# ---------------------------------------------------------------------------

def _make_schwab_client(next_earnings_date: str | None = None,
                        field: str = "nextEarningsDate"):
    """Build a mock Schwab client that returns fundamentals with the given date."""
    client = MagicMock()
    fund = {field: next_earnings_date} if next_earnings_date else {}
    client.get_instruments.return_value.raise_for_status = MagicMock()
    client.get_instruments.return_value.json.return_value = {
        "instruments": [{"symbol": "NVDA", "fundamental": fund}]
    }
    client.Instrument.Projection.FUNDAMENTAL = "fundamental"
    return client


class TestCheckEarningsProximity:
    def test_no_earnings_returns_false(self):
        from signal_verifier import check_earnings_proximity
        client = _make_schwab_client(next_earnings_date=None)
        assert check_earnings_proximity("NVDA", schwab_client=client) is False

    def test_earnings_within_5_days_returns_true(self):
        from signal_verifier import check_earnings_proximity
        near = (date.today() + timedelta(days=3)).isoformat()
        client = _make_schwab_client(near)
        assert check_earnings_proximity("NVDA", schwab_client=client) is True

    def test_earnings_far_away_returns_false(self):
        from signal_verifier import check_earnings_proximity
        far = (date.today() + timedelta(days=30)).isoformat()
        client = _make_schwab_client(far)
        assert check_earnings_proximity("NVDA", schwab_client=client) is False

    def test_earnings_today_returns_true(self):
        from signal_verifier import check_earnings_proximity
        today = date.today().isoformat()
        client = _make_schwab_client(today)
        assert check_earnings_proximity("NVDA", schwab_client=client) is True

    def test_earnings_exactly_5_days_returns_true(self):
        from signal_verifier import check_earnings_proximity
        cutoff = (date.today() + timedelta(days=5)).isoformat()
        client = _make_schwab_client(cutoff)
        assert check_earnings_proximity("NVDA", schwab_client=client) is True

    def test_date_with_timestamp_suffix_parsed(self):
        """Schwab may return dates like '2025-08-27 00:00:00.0'"""
        from signal_verifier import check_earnings_proximity
        near = (date.today() + timedelta(days=2)).isoformat() + " 00:00:00.0"
        client = _make_schwab_client(near)
        assert check_earnings_proximity("NVDA", schwab_client=client) is True

    def test_schwab_api_exception_falls_back_to_yfinance(self):
        from signal_verifier import check_earnings_proximity
        bad_client = MagicMock()
        bad_client.get_instruments.side_effect = Exception("network error")

        near = date.today() + timedelta(days=3)
        with patch("signal_verifier.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.calendar = {"Earnings Date": [near]}
            mock_yf.Ticker.return_value = mock_ticker
            result = check_earnings_proximity("NVDA", schwab_client=bad_client)
        assert result is True

    def test_no_client_falls_back_to_yfinance(self):
        """Without schwab_client, use yfinance calendar only."""
        from signal_verifier import check_earnings_proximity
        near = date.today() + timedelta(days=2)
        with patch("signal_verifier.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.calendar = {"Earnings Date": [near]}
            mock_yf.Ticker.return_value = mock_ticker
            result = check_earnings_proximity("NVDA")
        assert result is True

    def test_both_sources_fail_returns_false(self):
        from signal_verifier import check_earnings_proximity
        bad_client = MagicMock()
        bad_client.get_instruments.side_effect = Exception("fail")
        with patch("signal_verifier.yf") as mock_yf:
            mock_yf.Ticker.return_value.calendar = None
            result = check_earnings_proximity("NVDA", schwab_client=bad_client)
        assert result is False

    def test_empty_instruments_list_returns_false(self):
        from signal_verifier import check_earnings_proximity
        client = MagicMock()
        client.get_instruments.return_value.raise_for_status = MagicMock()
        client.get_instruments.return_value.json.return_value = {"instruments": []}
        client.Instrument.Projection.FUNDAMENTAL = "fundamental"
        with patch("signal_verifier.yf") as mock_yf:
            mock_yf.Ticker.return_value.calendar = {}
            result = check_earnings_proximity("NVDA", schwab_client=client)
        assert result is False


# ---------------------------------------------------------------------------
# fetch_headlines — RSS only (no Finnhub)
# ---------------------------------------------------------------------------

class TestFetchHeadlines:
    @patch("signal_verifier.feedparser")
    def test_returns_list_of_strings(self, mock_feedparser):
        from signal_verifier import fetch_headlines
        mock_feedparser.parse.return_value = MagicMock(
            entries=[MagicMock(title="Market opens higher")]
        )
        headlines = fetch_headlines("NVDA")
        assert isinstance(headlines, list)
        assert all(isinstance(h, str) for h in headlines)

    @patch("signal_verifier.feedparser")
    def test_caps_at_max_headlines(self, mock_feedparser):
        from signal_verifier import fetch_headlines
        mock_feedparser.parse.return_value = MagicMock(
            entries=[MagicMock(title=f"Story {i}") for i in range(50)]
        )
        assert len(fetch_headlines("NVDA", max_headlines=10)) <= 10

    @patch("signal_verifier.feedparser")
    def test_handles_empty_feeds_gracefully(self, mock_feedparser):
        from signal_verifier import fetch_headlines
        mock_feedparser.parse.return_value = MagicMock(entries=[])
        assert fetch_headlines("NVDA") == []

    @patch("signal_verifier.feedparser")
    def test_rss_exception_returns_empty(self, mock_feedparser):
        from signal_verifier import fetch_headlines
        mock_feedparser.parse.side_effect = Exception("RSS down")
        assert fetch_headlines("NVDA") == []


# ---------------------------------------------------------------------------
# gather_signal_context
# ---------------------------------------------------------------------------

class TestGatherSignalContext:
    @patch("signal_verifier.fetch_vix", return_value=18.0)
    @patch("signal_verifier.check_earnings_proximity", return_value=False)
    @patch("signal_verifier.fetch_headlines", return_value=["NVDA rallies"])
    def test_returns_required_keys(self, *_):
        from signal_verifier import gather_signal_context
        ctx = gather_signal_context("NVDA")
        for key in ("symbol", "vix", "vix_status", "near_earnings",
                    "hard_blocks", "headlines"):
            assert key in ctx

    @patch("signal_verifier.fetch_vix", return_value=35.0)
    @patch("signal_verifier.check_earnings_proximity", return_value=False)
    @patch("signal_verifier.fetch_headlines", return_value=[])
    def test_high_vix_adds_hard_block(self, *_):
        from signal_verifier import gather_signal_context
        ctx = gather_signal_context("NVDA")
        assert ctx["vix_status"] == "BLOCK"
        assert len(ctx["hard_blocks"]) >= 1
        assert any("VIX" in b for b in ctx["hard_blocks"])

    @patch("signal_verifier.fetch_vix", return_value=18.0)
    @patch("signal_verifier.check_earnings_proximity", return_value=True)
    @patch("signal_verifier.fetch_headlines", return_value=[])
    def test_near_earnings_adds_hard_block(self, *_):
        from signal_verifier import gather_signal_context
        ctx = gather_signal_context("NVDA")
        assert ctx["near_earnings"] is True
        assert len(ctx["hard_blocks"]) >= 1
        assert any("earnings" in b.lower() for b in ctx["hard_blocks"])

    @patch("signal_verifier.fetch_vix", return_value=18.0)
    @patch("signal_verifier.check_earnings_proximity", return_value=False)
    @patch("signal_verifier.fetch_headlines", return_value=["good news"])
    def test_all_clear_has_no_hard_blocks(self, *_):
        from signal_verifier import gather_signal_context
        ctx = gather_signal_context("NVDA")
        assert ctx["hard_blocks"] == []

    def test_schwab_client_passed_through(self):
        """schwab_client kwarg is forwarded to check_earnings_proximity."""
        from signal_verifier import gather_signal_context
        mock_client = MagicMock()
        with patch("signal_verifier.fetch_vix", return_value=15.0), \
             patch("signal_verifier.check_earnings_proximity", return_value=False) as mock_ep, \
             patch("signal_verifier.fetch_headlines", return_value=[]):
            gather_signal_context("NVDA", schwab_client=mock_client)
            mock_ep.assert_called_once_with("NVDA", schwab_client=mock_client)
