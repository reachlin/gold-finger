"""
Tests for signal_verifier.py — data-fetch utilities.

All external calls (yfinance, Finnhub, feedparser) are mocked.

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
# check_earnings_proximity
# ---------------------------------------------------------------------------

class TestCheckEarningsProximity:
    @patch("signal_verifier.finnhub_client")
    def test_no_earnings_returns_false(self, mock_client):
        from signal_verifier import check_earnings_proximity
        mock_client.earnings_calendar.return_value = {"earningsCalendar": []}
        assert check_earnings_proximity("NVDA") is False

    @patch("signal_verifier.finnhub_client")
    def test_earnings_within_5_days_returns_true(self, mock_client):
        from signal_verifier import check_earnings_proximity
        near_date = (date.today() + timedelta(days=3)).isoformat()
        mock_client.earnings_calendar.return_value = {
            "earningsCalendar": [{"date": near_date, "symbol": "NVDA"}]
        }
        assert check_earnings_proximity("NVDA") is True

    @patch("signal_verifier.finnhub_client")
    def test_earnings_far_away_returns_false(self, mock_client):
        from signal_verifier import check_earnings_proximity
        far_date = (date.today() + timedelta(days=30)).isoformat()
        mock_client.earnings_calendar.return_value = {
            "earningsCalendar": [{"date": far_date, "symbol": "NVDA"}]
        }
        assert check_earnings_proximity("NVDA") is False

    @patch("signal_verifier.finnhub_client")
    def test_api_exception_returns_false(self, mock_client):
        from signal_verifier import check_earnings_proximity
        mock_client.earnings_calendar.side_effect = Exception("API error")
        assert check_earnings_proximity("NVDA") is False


# ---------------------------------------------------------------------------
# fetch_headlines
# ---------------------------------------------------------------------------

class TestFetchHeadlines:
    @patch("signal_verifier.finnhub_client")
    @patch("signal_verifier.feedparser")
    def test_returns_list_of_strings(self, mock_feedparser, mock_finnhub):
        from signal_verifier import fetch_headlines
        mock_finnhub.company_news.return_value = [
            {"headline": "NVDA beats earnings"},
            {"headline": "NVDA new GPU launch"},
        ]
        mock_feedparser.parse.return_value = MagicMock(
            entries=[MagicMock(title="Market opens higher")]
        )
        headlines = fetch_headlines("NVDA")
        assert isinstance(headlines, list)
        assert all(isinstance(h, str) for h in headlines)

    @patch("signal_verifier.finnhub_client")
    @patch("signal_verifier.feedparser")
    def test_caps_at_max_headlines(self, mock_feedparser, mock_finnhub):
        from signal_verifier import fetch_headlines
        mock_finnhub.company_news.return_value = [
            {"headline": f"Story {i}"} for i in range(30)
        ]
        mock_feedparser.parse.return_value = MagicMock(
            entries=[MagicMock(title=f"Macro {i}") for i in range(30)]
        )
        assert len(fetch_headlines("NVDA", max_headlines=10)) <= 10

    @patch("signal_verifier.finnhub_client")
    @patch("signal_verifier.feedparser")
    def test_handles_empty_news_gracefully(self, mock_feedparser, mock_finnhub):
        from signal_verifier import fetch_headlines
        mock_finnhub.company_news.return_value = []
        mock_feedparser.parse.return_value = MagicMock(entries=[])
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
