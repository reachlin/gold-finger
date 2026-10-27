"""
The earnings gate let a put through two days before an AMZN print.

2026-10-26, live, Vault 76. Scan #34 at 11:56 ET refused the trade:

    [Earnings] ⊘ AMZN SELL_PUT $237.5 dropped — AMZN earnings 2026-10-29
    in 3d (confirmed) falls before expiry — typical print moves 7.2%

Scan #35 at 12:01 ET did not mention AMZN at all, approved a $235 put at
5.5% OTM, and filled it for $648 of credit against $23,500 of collateral.
Nothing about the signal changed in those five minutes. The upcoming-earnings
cache crossed CACHE_MAX_AGE_DAYS between the two scans, so `_read_confirmed`
stopped returning 2026-10-29 and the projection took over:

    AMZN  cache_age=8  next_earnings=(2026-10-24, 'projected')  days_away=-3
          earnings_before_expiry=False   block_reason -> None

Two independent defects, each of which would have been harmless alone:

1. `_project` returned a date THREE DAYS IN THE PAST. It walks the median
   quarterly gap forward from the last report (2026-07-30 + 91d = 2026-10-29)
   and then subtracts PROJECTION_MARGIN_DAYS, which put it at 2026-10-24.
   That violates `next_earnings`'s own contract — "the next report after
   `on_date`" — and `annotate` tests `today < when <= expiry`, so a past date
   reads as an affirmative "no print in this window".

2. `block_reason` failed OPEN. It blocked only on
   `earnings_before_expiry is True`, so unknown, missing, corrupt and
   nonsensical all passed, and the defect above never had to survive a second
   check. The module docstring chose that deliberately — "Unknown earnings
   does NOT block. A broken yfinance feed must not be able to halt the income
   strategy" — against `next_earnings`'s "the caller must treat that as
   unknown, not safe". This file is where that contradiction gets settled, in
   favour of the stricter half: for a cash-secured put, blind is not safe.
   Blocking costs a mean $+23.65 cycle; selling through a print is p5 $-312
   and worst $-13,012 on the same measurement.

The blast radius was not one symbol. At 8-day staleness, six of fourteen were
ungated because their projection had gone negative: AAPL (-3d), AMZN (-3d),
META (-4d), MSFT (-4d), PG (-4d), XOM (-2d) — in the middle of earnings
season.

And the reason the cache was ever 8 days old: `refresh_if_stale(WATCHLIST)`
runs once, at scanner startup. A process that stays up longer than
CACHE_MAX_AGE_DAYS loses its confirmed dates and never gets them back. The
live fix on 2026-10-26 was a restart, which is not a fix.
"""
import datetime as dt
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import earnings_calendar as ec


@pytest.fixture
def cache(tmp_path):
    fund_dir = tmp_path / "data" / "fundamentals"
    fund_dir.mkdir(parents=True)
    return str(tmp_path / "data"), str(fund_dir)


def _write_upcoming(fund_dir, symbol, dates, fetched_at):
    pd.DataFrame({
        "date":       [str(d) for d in dates],
        "eps_est":    [1.0] * len(dates),
        "fetched_at": [str(fetched_at)] * len(dates),
    }).to_csv(os.path.join(fund_dir, f"{symbol.lower()}_earnings_upcoming.csv"),
              index=False)


def _write_history(fund_dir, symbol, dates):
    pd.DataFrame({
        "date":         [f"{d} 16:00:00" for d in dates],
        "eps_est":      [1.0] * len(dates),
        "eps":          [1.1] * len(dates),
        "surprise_pct": [5.0] * len(dates),
    }).to_csv(os.path.join(fund_dir, f"{symbol.lower()}_earnings.csv"),
              index=False)


# AMZN's real reporting history as of the incident. The median gap is 91 days,
# so the next print projects to 2026-07-30 + 91d = 2026-10-29 — which is what
# the confirmed feed also said, before the cache went stale.
AMZN_HISTORY = ["2025-05-01", "2025-07-31", "2025-10-30",
                "2026-02-05", "2026-04-29", "2026-07-30"]


# ---------------------------------------------------------------------------
# Defect 1 — a projection may never land on or before the date it is asked about
# ---------------------------------------------------------------------------

class TestAProjectionIsNeverInThePast:

    def test_the_amzn_projection_of_2026_10_26_is_not_three_days_ago(self, cache):
        """The measured failure, reproduced from AMZN's own cadence."""
        _, fund_dir = cache
        _write_history(fund_dir, "AMZN", AMZN_HISTORY)
        when = ec._project("AMZN", dt.date(2026, 10, 26), fund_dir)
        assert when is not None
        assert when > dt.date(2026, 10, 26), (
            f"projected {when}, which is in the past — this is the bug")

    def test_next_earnings_keeps_its_documented_contract(self, cache):
        """
        "(date, source) of the next report AFTER `on_date`". A past date is
        not a next report, and `annotate` reads one as proof of safety.
        """
        _, fund_dir = cache
        _write_history(fund_dir, "AMZN", AMZN_HISTORY)
        got = ec.next_earnings("AMZN", dt.date(2026, 10, 26), fund_dir=fund_dir)
        assert got is not None
        assert got[0] > dt.date(2026, 10, 26)

    def test_the_margin_still_pulls_the_date_earlier_when_it_can(self, cache):
        """
        The clamp must not cost the safety margin its job. With the print a
        month out there is room to subtract the full margin, and the gate
        wants that date earlier, not later.
        """
        _, fund_dir = cache
        _write_history(fund_dir, "ACME", ["2025-10-20", "2026-01-20",
                                          "2026-04-20", "2026-07-20"])
        when = ec._project("ACME", dt.date(2026, 10, 1), fund_dir)
        naive = dt.date(2026, 7, 20) + dt.timedelta(days=91)
        assert when == naive - dt.timedelta(days=ec.PROJECTION_MARGIN_DAYS)

    def test_a_print_inside_the_margin_window_reads_as_imminent(self, cache):
        """
        When the margin pulls the date to today or earlier, the honest reading
        is not "it already happened" but "it could be any day now" — the
        projection is only accurate to within the margin. Imminent is the safe
        resolution for a put seller, and it is also the true one: the real
        AMZN print was three days out, not three days past.
        """
        _, fund_dir = cache
        _write_history(fund_dir, "AMZN", AMZN_HISTORY)
        on = dt.date(2026, 10, 26)
        when = ec._project("AMZN", on, fund_dir)
        assert on < when <= dt.date(2026, 10, 29)

    def test_the_whole_watchlist_projects_into_the_future(self):
        """
        An invariant over the real fundamentals directory, swept across a year
        of as-of dates: whatever the cache state, a projection is never on or
        before the day it is asked about. This is the check that would have
        caught all six ungated symbols at once.
        """
        watchlist = ["NVDA", "AMD", "AAPL", "AMZN", "META", "MSFT", "GOOGL",
                     "IBM", "INTC", "KO", "MMM", "PG", "XOM", "UNH"]
        start = dt.date(2026, 1, 1)
        offenders = []
        for sym in watchlist:
            for k in range(0, 365, 7):
                on = start + dt.timedelta(days=k)
                when = ec._project(sym, on, ec.FUND_DIR)
                if when is not None and when <= on:
                    offenders.append((sym, str(on), str(when)))
        assert not offenders, f"projections in the past: {offenders[:12]}"


# ---------------------------------------------------------------------------
# Defect 2 — the gate fails closed
# ---------------------------------------------------------------------------

class TestTheGateFailsClosed:

    def _signal(self, **kw):
        s = {"symbol": "AMZN", "signal": "SELL_PUT", "strike": 235.0,
             "dte": 32, "otm_pct": 5.5,
             "earnings_date": "unknown", "earnings_source": None,
             "earnings_days_away": None, "earnings_before_expiry": None,
             "earnings_move_pct": None, "earnings_no_history": False}
        s.update(kw)
        return s

    def test_an_unknown_date_blocks_a_put_on_a_symbol_that_reports(self):
        """
        This is the reversal. A symbol with a reporting history and no placeable
        next date is a blind spot, and the position being secured is $23,500 of
        cash against a print that moves the stock 7%.
        """
        reason = ec.block_reason(self._signal())
        assert reason is not None
        assert "AMZN" in reason

    def test_the_reason_says_the_date_is_unknown_rather_than_naming_one(self):
        """A log line that invents a date would send the next reader hunting."""
        reason = ec.block_reason(self._signal())
        assert "unknown" in reason.lower()

    def test_an_affirmative_no_is_still_the_only_way_through(self):
        assert ec.block_reason(
            self._signal(earnings_date="2027-01-29",
                         earnings_source="confirmed",
                         earnings_days_away=94,
                         earnings_before_expiry=False)) is None

    def test_a_known_date_with_no_testable_window_blocks(self):
        """
        `_expiry_of` returns None when a signal carries neither expiry nor dte,
        and `annotate` then leaves earnings_before_expiry unset. Knowing the
        print exists but not whether it lands inside the window is still blind.
        """
        reason = ec.block_reason(
            self._signal(earnings_date="2026-10-29",
                         earnings_source="confirmed",
                         earnings_days_away=3,
                         earnings_before_expiry=None))
        assert reason is not None

    def test_an_instrument_with_no_reporting_history_still_passes(self):
        """
        The carve-out the module docstring was really protecting: something
        that does not report earnings at all — an ETF — must not be blocked
        forever by a gate about earnings. Evidence of no reports, not absence
        of evidence, is what earns the pass.
        """
        assert ec.block_reason(
            self._signal(symbol="SCHD", earnings_no_history=True)) is None

    def test_a_missing_no_history_field_blocks_rather_than_passes(self):
        """
        Any caller that annotates with an older copy of this module, or hands
        over a hand-built dict, must not get the lenient path by omission.
        """
        s = self._signal()
        del s["earnings_no_history"]
        assert ec.block_reason(s) is not None

    def test_covered_calls_are_untouched_by_the_reversal(self):
        """
        Being called away through a print caps a gain. Failing closed here
        would refuse income to protect against nothing — and SELL_CALL income
        is how an assignment gets worked out of.
        """
        assert ec.block_reason(self._signal(signal="SELL_CALL")) is None
        assert ec.block_reason(self._signal(signal="BUY")) is None

    def test_a_print_inside_the_window_still_blocks_with_its_date(self):
        """The original gate, unchanged."""
        reason = ec.block_reason(
            self._signal(earnings_date="2026-10-29", earnings_source="confirmed",
                         earnings_days_away=3, earnings_before_expiry=True,
                         earnings_move_pct=7.22))
        assert reason is not None
        assert "2026-10-29" in reason and "7.2%" in reason


# ---------------------------------------------------------------------------
# annotate carries the evidence the gate needs
# ---------------------------------------------------------------------------

class TestAnnotateReportsWhetherTheSymbolEverReports:

    def test_a_symbol_with_history_is_not_marked_no_history(self, cache):
        data_dir, fund_dir = cache
        _write_history(fund_dir, "AMZN", AMZN_HISTORY)
        s = ec.annotate({"symbol": "AMZN", "signal": "SELL_PUT", "dte": 32},
                        today=dt.date(2026, 10, 26),
                        data_dir=data_dir, fund_dir=fund_dir)
        assert s["earnings_no_history"] is False

    def test_an_empty_fundamentals_dir_marks_no_history(self, cache):
        data_dir, fund_dir = cache
        s = ec.annotate({"symbol": "SCHD", "signal": "SELL_PUT", "dte": 32},
                        today=dt.date(2026, 10, 26),
                        data_dir=data_dir, fund_dir=fund_dir)
        assert s["earnings_no_history"] is True
        assert s["earnings_date"] == "unknown"

    def test_a_corrupt_history_file_does_not_earn_the_lenient_path(self, cache):
        """
        Unreadable is not the same as "this instrument has no earnings". A
        truncated or half-written CSV must leave the gate closed.
        """
        data_dir, fund_dir = cache
        with open(os.path.join(fund_dir, "amzn_earnings.csv"), "w") as fh:
            fh.write("date,eps\n2026-07-3")        # truncated mid-row
        s = ec.annotate({"symbol": "AMZN", "signal": "SELL_PUT", "dte": 32},
                        today=dt.date(2026, 10, 26),
                        data_dir=data_dir, fund_dir=fund_dir)
        assert s["earnings_no_history"] is False
        assert ec.block_reason(s) is not None

    def test_the_stale_cache_path_now_blocks_end_to_end(self, cache):
        """
        The live sequence, start to finish: an 8-day-old cache naming
        2026-10-29, a history whose cadence projects to the same date, and a
        32-day put. Before the fix the cache was discarded, the projection
        came back 2026-10-24, and earnings_before_expiry=False let it through.
        """
        data_dir, fund_dir = cache
        _write_upcoming(fund_dir, "AMZN", [dt.date(2026, 10, 29)],
                        fetched_at=dt.date(2026, 10, 18))
        _write_history(fund_dir, "AMZN", AMZN_HISTORY)
        s = ec.annotate({"symbol": "AMZN", "signal": "SELL_PUT",
                         "strike": 235.0, "dte": 32},
                        today=dt.date(2026, 10, 26),
                        data_dir=data_dir, fund_dir=fund_dir)
        assert s["earnings_source"] == "stale cache"
        assert s["earnings_before_expiry"] is True
        assert ec.block_reason(s) is not None


# ---------------------------------------------------------------------------
# Defect 3 — the calendar has to refresh while the process is up
# ---------------------------------------------------------------------------

class TestTheCalendarRefreshesWhileRunning:

    @pytest.fixture(autouse=True)
    def _clean_clock(self):
        ec._last_refresh_at = None
        yield
        ec._last_refresh_at = None

    def test_a_scan_refreshes_a_stale_cache_without_a_restart(self, cache):
        """
        The root cause of the incident: refresh_if_stale ran only at startup,
        so an 8-day uptime silently lost its confirmed dates. A restart is what
        actually fixed it live on 2026-10-26, and a restart is not a fix.
        """
        _, fund_dir = cache
        _write_upcoming(fund_dir, "AMZN", [dt.date(2026, 10, 29)],
                        fetched_at=dt.date(2026, 10, 18))
        fetched = []
        out = ec.maybe_refresh(["AMZN"], fund_dir=fund_dir,
                               now=dt.datetime(2026, 10, 26, 12, 1),
                               fetcher=fetched.append)
        assert fetched == ["AMZN"]
        assert out == {"fresh": 0, "refreshed": 1, "failed": 0}

    def test_a_fresh_cache_costs_no_fetch(self, cache):
        _, fund_dir = cache
        _write_upcoming(fund_dir, "AMZN", [dt.date(2026, 10, 29)],
                        fetched_at=dt.date(2026, 10, 26))
        fetched = []
        ec.maybe_refresh(["AMZN"], fund_dir=fund_dir,
                         now=dt.datetime(2026, 10, 26, 12, 1),
                         fetcher=fetched.append)
        assert fetched == []

    def test_the_next_scan_five_minutes_later_does_not_fetch_again(self, cache):
        """
        Scans run every five minutes and yfinance throttles. Checking the
        calendar is cheap; re-scraping it 78 times a session is not — and the
        429 that killed a live GTC cover on 2026-10-13 is what rate-limit
        pressure costs here.
        """
        _, fund_dir = cache
        fetched = []
        ec.maybe_refresh(["AMZN"], fund_dir=fund_dir,
                         now=dt.datetime(2026, 10, 26, 12, 1),
                         fetcher=fetched.append)
        again = ec.maybe_refresh(["AMZN"], fund_dir=fund_dir,
                                 now=dt.datetime(2026, 10, 26, 12, 6),
                                 fetcher=fetched.append)
        assert again is None
        assert len(fetched) == 1

    def test_a_failed_refresh_is_retried_the_same_session(self, cache):
        """
        A symbol that fails to fetch must not wait for the next restart. With
        the gate now failing closed, a blind symbol stops trading, so the
        retry interval is also the outage length.
        """
        _, fund_dir = cache

        def _boom(sym):
            raise RuntimeError("yfinance said no")

        first = ec.maybe_refresh(["AMZN"], fund_dir=fund_dir,
                                 now=dt.datetime(2026, 10, 26, 10, 0),
                                 fetcher=_boom)
        assert first == {"fresh": 0, "refreshed": 0, "failed": 1}

        fetched = []
        later = dt.datetime(2026, 10, 26, 10, 0) + dt.timedelta(
            hours=ec.REFRESH_INTERVAL_HOURS, minutes=1)
        out = ec.maybe_refresh(["AMZN"], fund_dir=fund_dir, now=later,
                               fetcher=fetched.append)
        assert out is not None and fetched == ["AMZN"]

    def test_the_refresher_never_raises_into_the_scan(self, cache):
        """A calendar problem must not be able to stop a scan from running."""
        _, fund_dir = cache
        out = ec.maybe_refresh(["AMZN"], fund_dir="/nonexistent/path",
                               now=dt.datetime(2026, 10, 26, 12, 1),
                               fetcher=lambda s: (_ for _ in ()).throw(
                                   OSError("no such dir")))
        assert out == {"fresh": 0, "refreshed": 0, "failed": 1}


# ---------------------------------------------------------------------------
# Defect 4 — found while verifying the fix, the same hole facing the other way
# ---------------------------------------------------------------------------

class TestAProjectionDoesNotSkipALatePrint:
    """
    Replaying 2026-10-26 at 8-day staleness across the watchlist, AMZN was
    fixed but GOOGL and INTC came back clean — and both were about to report:

        GOOGL  last report 2026-07-22  cadence 91d -> 2026-10-21, 5d overdue
               projection answered 2027-01-15 (81d) -> OPEN
               real print 2026-10-28, TWO DAYS AWAY
        INTC   last report 2026-07-23  cadence 91d -> 2026-10-22, 4d overdue
               projection answered 2027-01-16 (82d) -> OPEN
               real print 2026-10-29, three days away

    `_project` walked forward `while when <= on_date`, so a cadence date that
    has merely passed was assumed to have been reported. A company that
    reports a week late skips a whole quarter and the gate affirms a clean
    window. This is worse than the AMZN case: a past date at least looked
    wrong, while 2027-01-15 looks right.

    Two fixes, measured against the live fundamentals directory below:
    a stale cache's date is used before any cadence guess (it is a date
    somebody actually published — the 8-day-old cache said 2026-10-29, the
    correct answer, and throwing it away is what produced 2026-10-24), and a
    projection only steps past a date that is more than OVERDUE_GRACE_DAYS
    overdue.
    """

    def test_a_print_days_overdue_is_not_treated_as_reported(self, cache):
        """GOOGL's cadence, with no cache to fall back on."""
        _, fund_dir = cache
        _write_history(fund_dir, "GOOGL", ["2025-10-22", "2026-01-21",
                                           "2026-04-22", "2026-07-22"])
        when = ec._project("GOOGL", dt.date(2026, 10, 26), fund_dir)
        assert when is not None
        assert when < dt.date(2026, 11, 26), (
            f"projected {when} — skipped a quarter past a 5-day-late print")

    def test_a_long_overdue_projection_does_step_to_the_next_quarter(self, cache):
        """
        The grace must not become a permanent block. Six weeks past the
        cadence date the report really did happen and the history file is
        simply behind, so the next print is a quarter out.
        """
        _, fund_dir = cache
        _write_history(fund_dir, "GOOGL", ["2025-10-22", "2026-01-21",
                                           "2026-04-22", "2026-07-22"])
        when = ec._project("GOOGL", dt.date(2026, 12, 1), fund_dir)
        assert when is not None and when > dt.date(2026, 12, 20)

    def test_a_report_already_on_file_is_not_re_predicted(self, cache):
        """
        The grace applies to steps FORWARD, never to the last known report
        itself. META reported 2026-10-21; on 2026-10-26 its next print is a
        quarter out, and treating the report on file as "4 days overdue" would
        block a clean symbol for three weeks.
        """
        _, fund_dir = cache
        _write_history(fund_dir, "META", ["2026-01-28", "2026-04-29",
                                          "2026-07-29", "2026-10-21"])
        when = ec._project("META", dt.date(2026, 10, 26), fund_dir)
        assert when is not None and when > dt.date(2026, 12, 31)

    def test_a_stale_cache_date_beats_a_cadence_guess(self, cache):
        """
        The 8-day-old AMZN cache still said 2026-10-29 — the date the company
        actually published. A scrape with a known age is degraded information,
        not absent information, and ranks above a median-gap estimate.
        """
        _, fund_dir = cache
        _write_upcoming(fund_dir, "AMZN", [dt.date(2026, 10, 29)],
                        fetched_at=dt.date(2026, 10, 18))
        _write_history(fund_dir, "AMZN", AMZN_HISTORY)
        got = ec.next_earnings("AMZN", dt.date(2026, 10, 26), fund_dir=fund_dir)
        assert got is not None
        when, source = got
        assert when == dt.date(2026, 10, 29)
        assert source != "confirmed", "a stale scrape must not pass as fresh"

    def test_the_degraded_source_is_visible_in_the_block_reason(self, cache):
        """Whoever reads the log has to be able to see which date they got."""
        data_dir, fund_dir = cache
        _write_upcoming(fund_dir, "AMZN", [dt.date(2026, 10, 29)],
                        fetched_at=dt.date(2026, 10, 18))
        _write_history(fund_dir, "AMZN", AMZN_HISTORY)
        s = ec.annotate({"symbol": "AMZN", "signal": "SELL_PUT",
                         "strike": 235.0, "dte": 32},
                        today=dt.date(2026, 10, 26),
                        data_dir=data_dir, fund_dir=fund_dir)
        reason = ec.block_reason(s)
        assert reason is not None
        assert "stale" in reason.lower()

    def test_an_expired_cache_date_is_not_mistaken_for_the_next_one(self, cache):
        """
        A stale cache naming only dates that have passed says nothing about
        the next print, so the cadence projection has to take over rather than
        the gate reading a past date as a clean window.
        """
        _, fund_dir = cache
        _write_upcoming(fund_dir, "AMZN", [dt.date(2026, 7, 30)],
                        fetched_at=dt.date(2026, 7, 20))
        _write_history(fund_dir, "AMZN", AMZN_HISTORY)
        got = ec.next_earnings("AMZN", dt.date(2026, 10, 26), fund_dir=fund_dir)
        assert got is not None
        assert got[0] > dt.date(2026, 10, 26)

    def test_a_stale_cache_never_answers_later_than_the_truth(self):
        """
        The invariant over the live fundamentals directory: for every
        watchlist symbol, age its cache past CACHE_MAX_AGE_DAYS and the answer
        must not land LATER than the fresh confirmed date. Answering later is
        what opens the gate — 2027-01-15 against a real print on 2026-10-28.

        Scoped to the cached path on purpose. With no cache at all the cadence
        guess can still overshoot by a few days (NVDA projects 2026-11-20
        against a confirmed 2026-11-17), and what bounds that is
        PROJECTION_MARGIN_DAYS, a measured parameter, not this clamp.
        """
        watchlist = ["NVDA", "AMD", "AAPL", "AMZN", "META", "MSFT", "GOOGL",
                     "IBM", "INTC", "KO", "MMM", "PG", "XOM", "UNH"]
        on = dt.date.today()
        late = []
        for sym in watchlist:
            truth = ec._read_confirmed(sym, on, ec.FUND_DIR)
            if truth is None:
                continue
            aged = ec._read_cached(sym, on, ec.FUND_DIR)   # ignores age
            if aged is not None and aged > truth:
                late.append((sym, str(truth), str(aged)))
        assert not late, f"answered later than the real print: {late}"
