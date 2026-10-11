"""
Tests for fetch_fundamentals.split_reports — the forward earnings calendar.

yfinance returns reported AND scheduled prints in one frame; the scheduled
rows have no EPS. They used to be dropped, which is why no signal ever
carried an earnings date (see earnings_calendar.py). These tests pin the
split so the forward calendar cannot silently go empty again.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fetch_fundamentals as ff


def _frame(rows):
    """rows: list of (timestamp str, eps_est, eps or None) — yfinance shape:
    tz-aware DatetimeIndex named 'Earnings Date', newest first."""
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz="America/New_York")
                            for t, _, _ in rows], name="Earnings Date")
    return pd.DataFrame({
        "EPS Estimate":  [e for _, e, _ in rows],
        "Reported EPS":  [r for _, _, r in rows],
        "Surprise(%)":   [None for _ in rows],
    }, index=idx)


class TestSplitReports:

    def test_a_scheduled_print_goes_to_upcoming(self):
        ed = _frame([("2026-10-21 16:00:00", 2.86, None),
                     ("2026-07-22 16:00:00", 2.93, 2.93)])
        reported, upcoming = ff.split_reports(
            ed, today=pd.Timestamp("2026-10-11"))
        assert list(upcoming["date"].astype(str)) == ["2026-10-21 16:00:00"]
        assert len(reported) == 1

    def test_reported_history_is_unchanged_by_the_split(self):
        """The LGBM fundamental features read this file — it must not shift."""
        ed = _frame([("2026-10-21 16:00:00", 2.86, None),
                     ("2026-07-22 16:00:00", 2.93, 2.93),
                     ("2026-04-22 16:00:00", 1.81, 1.91)])
        reported, _ = ff.split_reports(ed, today=pd.Timestamp("2026-10-11"))
        assert list(reported["eps"]) == [1.91, 2.93]      # oldest first
        assert reported["date"].is_monotonic_increasing
        assert reported["eps"].notna().all()

    def test_timezone_is_stripped_so_dates_compare_cleanly(self):
        ed = _frame([("2026-10-21 16:00:00", 2.86, None)])
        _, upcoming = ff.split_reports(ed, today=pd.Timestamp("2026-10-11"))
        assert upcoming["date"].iloc[0].tz is None

    def test_a_past_date_with_no_eps_is_not_upcoming(self):
        """A missed/never-filled old row must not look like a future print."""
        ed = _frame([("2026-10-21 16:00:00", 2.86, None),
                     ("2025-01-15 16:00:00", 1.00, None)])
        _, upcoming = ff.split_reports(ed, today=pd.Timestamp("2026-10-11"))
        assert list(upcoming["date"].astype(str)) == ["2026-10-21 16:00:00"]

    def test_todays_print_is_still_upcoming_until_the_eps_lands(self):
        """
        Reports land in the evening and the feed lags. A row dated today with
        no number has not happened yet as far as the gate is concerned.
        """
        ed = _frame([("2026-10-11 16:00:00", 2.86, None)])
        _, upcoming = ff.split_reports(ed, today=pd.Timestamp("2026-10-11"))
        assert len(upcoming) == 1

    def test_no_scheduled_print_yields_an_empty_calendar_not_an_error(self):
        ed = _frame([("2026-07-22 16:00:00", 2.93, 2.93)])
        reported, upcoming = ff.split_reports(
            ed, today=pd.Timestamp("2026-10-11"))
        assert upcoming.empty and len(reported) == 1

    def test_upcoming_carries_only_the_date_and_estimate(self):
        """earnings_calendar reads date + fetched_at; eps/surprise are noise."""
        ed = _frame([("2026-10-21 16:00:00", 2.86, None)])
        _, upcoming = ff.split_reports(ed, today=pd.Timestamp("2026-10-11"))
        assert list(upcoming.columns) == ["date", "eps_est"]
