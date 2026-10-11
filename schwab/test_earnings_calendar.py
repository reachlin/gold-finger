"""
Tests for earnings_calendar.py — the earnings gate on SELL_PUT signals.

Why this module exists: on 2026-10-11 all three open positions (IBM $220P
exp 10-23, GOOGL $335P and AMZN $245P exp 11-06) held an earnings print
inside their window, and nothing in the system knew it. The LLM prompt
instructs a veto for "earnings within ~5 trading days" but the model was
never handed a date, and no deterministic gate existed.

## What the measurement said (production backtest, 4,360 put cycles)

Run through walk_forward_scavenger with sell_calls=False so each cycle's
P&L is complete (with the wheel on, `put_assigned` books $0 and the real
loss lands in a later covered-call event — the first cut of this
measurement was censored that way and said the opposite):

    no earnings in window   n=3320   mean $+23.65   p5 $-19    worst $-4,176
    earnings IN window      n=1040   mean $-33.46   p5 $-312   worst $-13,012

The win RATE is higher through a print (87.6% vs 82.9%) — you win slightly
more often and lose very much bigger. Blocking the whole population is
worth +$34,803 across the watchlist, 23.9% of cycles cut.

## Why there is no cushion-ratio gate

The obvious design — allow the trade when the OTM cushion exceeds the move
that symbol actually makes — does not survive measurement. Bucketed by
otm% / median-print%, every bucket still loses money, and the rank
correlation between cushion ratio and P&L is +0.038:

    ratio 0.60–0.80   n=266   mean $-86.35
    ratio 0.80–1.00   n=106   mean $ -9.92
    ratio 1.00–1.25   n=245   mean $-26.62
    ratio 1.25–1.60   n=145   mean $-17.50
    ratio 1.60+       n=257   mean $ -5.51

So `earnings_move_pct` is carried as ADVISORY context for the LLM and
gates nothing. test_cushion_ratio_does_not_gate pins that, so re-adding
such a gate has to come with a fresh measurement.
"""
import datetime as dt
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import earnings_calendar as ec


# ---------------------------------------------------------------------------
# Fixtures — a fake fundamentals/history cache on disk
# ---------------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path):
    """A (data_dir, fund_dir) pair with writable caches."""
    data_dir = tmp_path / "data"
    fund_dir = data_dir / "fundamentals"
    fund_dir.mkdir(parents=True)
    return str(data_dir), str(fund_dir)


def _write_upcoming(fund_dir, symbol, dates, fetched_at=None):
    fetched_at = fetched_at or dt.date(2026, 10, 11)
    pd.DataFrame({
        "date":       [str(d) for d in dates],
        "eps_est":    [1.0] * len(dates),
        "fetched_at": [str(fetched_at)] * len(dates),
    }).to_csv(os.path.join(fund_dir, f"{symbol.lower()}_earnings_upcoming.csv"),
              index=False)


def _write_history(fund_dir, symbol, dates):
    """Past reports, newest last — the shape fetch_fundamentals writes."""
    pd.DataFrame({
        "date":         [f"{d} 16:00:00" for d in dates],
        "eps_est":      [1.0] * len(dates),
        "eps":          [1.1] * len(dates),
        "surprise_pct": [5.0] * len(dates),
    }).to_csv(os.path.join(fund_dir, f"{symbol.lower()}_earnings.csv"),
              index=False)


def _write_prices(data_dir, symbol, rows):
    """rows: list of (date, close)."""
    os.makedirs(data_dir, exist_ok=True)
    pd.DataFrame({
        "datetime": [f"{d} 00:00:00" for d, _ in rows],
        "open":     [c for _, c in rows],
        "high":     [c for _, c in rows],
        "low":      [c for _, c in rows],
        "close":    [c for _, c in rows],
        "volume":   [1_000_000] * len(rows),
    }).to_csv(os.path.join(data_dir, f"{symbol.lower()}_history.csv"),
              index=False)


# ---------------------------------------------------------------------------
# next_earnings — finding the date
# ---------------------------------------------------------------------------

class TestNextEarnings:

    def test_confirmed_date_from_the_upcoming_cache(self, cache):
        _, fund_dir = cache
        _write_upcoming(fund_dir, "IBM", [dt.date(2026, 10, 21)])
        got = ec.next_earnings("IBM", dt.date(2026, 10, 11), fund_dir=fund_dir)
        assert got == (dt.date(2026, 10, 21), "confirmed")

    def test_a_date_already_past_is_not_the_next_one(self, cache):
        _, fund_dir = cache
        _write_upcoming(fund_dir, "IBM", [dt.date(2026, 10, 21),
                                          dt.date(2027, 1, 20)],
                        fetched_at=dt.date(2026, 10, 25))
        got = ec.next_earnings("IBM", dt.date(2026, 10, 25), fund_dir=fund_dir)
        assert got == (dt.date(2027, 1, 20), "confirmed")

    def test_no_cache_at_all_is_unknown_not_a_guess(self, cache):
        _, fund_dir = cache
        assert ec.next_earnings("ZZZZ", dt.date(2026, 10, 11),
                                fund_dir=fund_dir) is None

    def test_a_stale_cache_falls_back_to_a_projection(self, cache):
        """
        The confirmed feed is only as good as its last refresh. A cache older
        than CACHE_MAX_AGE_DAYS must not be trusted as "confirmed" — but the
        quarterly cadence is still known, so project instead of giving up.
        """
        _, fund_dir = cache
        _write_upcoming(fund_dir, "IBM", [dt.date(2026, 10, 21)],
                        fetched_at=dt.date(2026, 8, 1))
        _write_history(fund_dir, "IBM", ["2025-10-22", "2026-01-28",
                                         "2026-04-22", "2026-07-22"])
        got = ec.next_earnings("IBM", dt.date(2026, 10, 11), fund_dir=fund_dir)
        assert got is not None
        when, source = got
        assert source == "projected"
        # ~91 days after 2026-07-22 lands in the second half of October
        assert dt.date(2026, 10, 14) <= when <= dt.date(2026, 10, 31)

    def test_projection_from_history_when_there_is_no_upcoming_file(self, cache):
        _, fund_dir = cache
        _write_history(fund_dir, "AMZN", ["2025-10-30", "2026-02-05",
                                          "2026-04-29", "2026-07-30"])
        got = ec.next_earnings("AMZN", dt.date(2026, 10, 11), fund_dir=fund_dir)
        assert got is not None and got[1] == "projected"

    def test_history_too_old_to_project_is_unknown(self, cache):
        """A history file that stops years ago cannot place the next print."""
        _, fund_dir = cache
        _write_history(fund_dir, "MMM", ["2021-01-26", "2021-04-27"])
        assert ec.next_earnings("MMM", dt.date(2026, 10, 11),
                                fund_dir=fund_dir) is None

    def test_a_projection_carries_a_safety_margin(self, cache):
        """
        A projected date is +/- a few days. PROJECTION_MARGIN_DAYS pulls it
        earlier so a print that slips a week still lands inside the window
        the gate is testing, rather than just outside it.
        """
        _, fund_dir = cache
        _write_history(fund_dir, "ACME", ["2025-10-20", "2026-01-20",
                                          "2026-04-20", "2026-07-20"])
        when, source = ec.next_earnings("ACME", dt.date(2026, 10, 1),
                                        fund_dir=fund_dir)
        naive = dt.date(2026, 7, 20) + dt.timedelta(days=91)
        assert source == "projected"
        assert when == naive - dt.timedelta(days=ec.PROJECTION_MARGIN_DAYS)


# ---------------------------------------------------------------------------
# typical_earnings_move — advisory context, not a gate
# ---------------------------------------------------------------------------

class TestTypicalEarningsMove:

    def test_measures_the_reaction_day_after_an_after_close_report(self, cache):
        """
        A 16:00 report moves the NEXT session. Using the report day's own
        close would measure the day before the news.
        """
        data_dir, fund_dir = cache
        _write_history(fund_dir, "ACME", ["2026-01-15", "2026-04-15",
                                          "2026-07-15", "2026-10-15"])
        rows = []
        # flat at 100 except a -10% reaction the day after each report
        for month in (1, 4, 7, 10):
            rows += [(f"2026-{month:02d}-14", 100.0),
                     (f"2026-{month:02d}-15", 100.0),   # report lands at 16:00
                     (f"2026-{month:02d}-16", 90.0),    # the reaction
                     (f"2026-{month:02d}-17", 90.0)]
        _write_prices(data_dir, "ACME", rows)
        move = ec.typical_earnings_move("ACME", data_dir=data_dir,
                                        fund_dir=fund_dir, min_reports=4)
        assert move == pytest.approx(10.0, abs=0.5)

    def test_too_few_reports_is_unknown(self, cache):
        data_dir, fund_dir = cache
        _write_history(fund_dir, "ACME", ["2026-07-15"])
        _write_prices(data_dir, "ACME", [("2026-07-15", 100.0),
                                         ("2026-07-16", 90.0)])
        assert ec.typical_earnings_move("ACME", data_dir=data_dir,
                                        fund_dir=fund_dir,
                                        min_reports=4) is None

    def test_direction_is_ignored_only_size_matters(self, cache):
        """An 8% rally and an 8% drop are the same size of surprise."""
        data_dir, fund_dir = cache
        _write_history(fund_dir, "ACME", ["2026-01-15", "2026-04-15",
                                          "2026-07-15", "2026-10-15"])
        rows, sign = [], 1
        for month in (1, 4, 7, 10):
            rows += [(f"2026-{month:02d}-15", 100.0),
                     (f"2026-{month:02d}-16", 100.0 + 8.0 * sign),
                     (f"2026-{month:02d}-17", 100.0 + 8.0 * sign)]
            sign = -sign
        _write_prices(data_dir, "ACME", rows)
        move = ec.typical_earnings_move("ACME", data_dir=data_dir,
                                        fund_dir=fund_dir, min_reports=4)
        assert move == pytest.approx(8.0, abs=0.6)


# ---------------------------------------------------------------------------
# annotate — the fields the LLM sees
# ---------------------------------------------------------------------------

class TestAnnotate:

    def _signal(self, **kw):
        s = {"symbol": "IBM", "signal": "SELL_PUT", "strike": 220.0,
             "close": 233.48, "otm_pct": 5.8, "dte": 32,
             "expiry": "2026-11-12"}
        s.update(kw)
        return s

    def test_a_print_inside_the_window_is_flagged_with_its_date(self, cache):
        data_dir, fund_dir = cache
        _write_upcoming(fund_dir, "IBM", [dt.date(2026, 10, 21)])
        s = self._signal()
        ec.annotate(s, today=dt.date(2026, 10, 11),
                    data_dir=data_dir, fund_dir=fund_dir)
        assert s["earnings_date"] == "2026-10-21"
        assert s["earnings_source"] == "confirmed"
        assert s["earnings_before_expiry"] is True
        assert s["earnings_days_away"] == 10

    def test_a_print_after_expiry_is_reported_but_not_a_concern(self, cache):
        data_dir, fund_dir = cache
        _write_upcoming(fund_dir, "IBM", [dt.date(2026, 12, 20)])
        s = self._signal()
        ec.annotate(s, today=dt.date(2026, 10, 11),
                    data_dir=data_dir, fund_dir=fund_dir)
        assert s["earnings_date"] == "2026-12-20"
        assert s["earnings_before_expiry"] is False

    def test_the_expiry_date_itself_counts_as_inside_the_window(self, cache):
        """A print on expiry day is the worst case, not a miss."""
        data_dir, fund_dir = cache
        _write_upcoming(fund_dir, "IBM", [dt.date(2026, 11, 12)])
        s = self._signal(expiry="2026-11-12")
        ec.annotate(s, today=dt.date(2026, 10, 11),
                    data_dir=data_dir, fund_dir=fund_dir)
        assert s["earnings_before_expiry"] is True

    def test_dte_is_used_when_the_signal_carries_no_expiry(self, cache):
        data_dir, fund_dir = cache
        _write_upcoming(fund_dir, "IBM", [dt.date(2026, 10, 21)])
        s = self._signal(dte=32)
        s.pop("expiry")
        ec.annotate(s, today=dt.date(2026, 10, 11),
                    data_dir=data_dir, fund_dir=fund_dir)
        assert s["earnings_before_expiry"] is True

    def test_unknown_earnings_is_stated_not_silently_omitted(self, cache):
        """
        A missing field reads to the LLM as "no earnings". It has to say
        unknown out loud — that is the lesson of the silent quote filters.
        """
        data_dir, fund_dir = cache
        s = self._signal(symbol="ZZZZ")
        ec.annotate(s, today=dt.date(2026, 10, 11),
                    data_dir=data_dir, fund_dir=fund_dir)
        assert s["earnings_date"] == "unknown"
        assert s["earnings_before_expiry"] is None

    def test_the_measured_move_is_attached_as_context(self, cache):
        """
        annotate() uses the production MIN_REPORTS_FOR_MOVE, so the fixture
        needs a real run of quarters — four prints is not enough to call a
        symbol's typical reaction, by design.
        """
        data_dir, fund_dir = cache
        _write_upcoming(fund_dir, "ACME", [dt.date(2026, 10, 21)])
        quarters = [(y, m) for y in (2025, 2026) for m in (1, 4, 7, 10)]
        _write_history(fund_dir, "ACME",
                       [f"{y}-{m:02d}-15" for y, m in quarters])
        rows = []
        for y, m in quarters:
            rows += [(f"{y}-{m:02d}-15", 100.0),
                     (f"{y}-{m:02d}-16", 90.0),     # -10% reaction
                     (f"{y}-{m:02d}-17", 90.0)]
        _write_prices(data_dir, "ACME", rows)
        s = self._signal(symbol="ACME", otm_pct=5.0)
        ec.annotate(s, today=dt.date(2026, 10, 11),
                    data_dir=data_dir, fund_dir=fund_dir)
        assert s["earnings_move_pct"] == pytest.approx(10.0, abs=0.5)

    def test_annotate_never_raises_on_a_broken_cache(self, cache):
        """
        The scanner must survive a corrupt cache file. An exception here
        would kill the scan for every symbol.
        """
        data_dir, fund_dir = cache
        with open(os.path.join(fund_dir, "ibm_earnings_upcoming.csv"), "w") as f:
            f.write("this is not,a valid\ncsv at all,,,\n")
        s = self._signal()
        ec.annotate(s, today=dt.date(2026, 10, 11),
                    data_dir=data_dir, fund_dir=fund_dir)
        assert s["earnings_date"] == "unknown"


# ---------------------------------------------------------------------------
# block_reason — the deterministic gate
# ---------------------------------------------------------------------------

class TestBlockReason:

    def _annotated(self, **kw):
        s = {"symbol": "X", "signal": "SELL_PUT", "otm_pct": 5.0,
             "earnings_date": "2026-10-21", "earnings_source": "confirmed",
             "earnings_before_expiry": True, "earnings_days_away": 10,
             "earnings_move_pct": 2.0}
        s.update(kw)
        return s

    def test_a_print_inside_the_window_is_blocked(self):
        reason = ec.block_reason(self._annotated())
        assert reason is not None
        assert "2026-10-21" in reason

    def test_no_print_inside_the_window_passes(self):
        assert ec.block_reason(
            self._annotated(earnings_before_expiry=False)) is None

    def test_unknown_earnings_does_not_block(self):
        """
        A broken yfinance feed must not halt the income strategy. Unknown
        passes — loudly, via the annotated field — rather than failing closed.
        """
        assert ec.block_reason(self._annotated(earnings_date="unknown",
                                               earnings_before_expiry=None,
                                               earnings_move_pct=None)) is None

    def test_a_projected_date_blocks_the_same_as_a_confirmed_one(self):
        """
        A projection is the only thing available when the feed is stale, and
        a quarterly cadence is accurate to within days. Treating it as "not
        known" would reopen exactly the hole this gate closes.
        """
        reason = ec.block_reason(self._annotated(earnings_source="projected"))
        assert reason is not None
        assert "projected" in reason

    def test_only_sell_put_is_gated(self):
        """
        A covered call through a print risks being called away — a capped
        gain, not a loss. Gating it would refuse income for no protection.
        """
        assert ec.block_reason(self._annotated(signal="SELL_CALL")) is None

    def test_cushion_ratio_does_not_gate(self):
        """
        Measured: bucketed by otm% / median-print%, EVERY bucket loses money
        and the rank correlation with P&L is +0.038. A roomy cushion is not
        protection. If this test is ever changed, the change needs a fresh
        measurement behind it — see this module's docstring.
        """
        # A 4x cushion over the measured move still does not earn a pass.
        roomy = self._annotated(otm_pct=8.0, earnings_move_pct=2.0)
        assert ec.block_reason(roomy) is not None

    def test_the_reason_names_the_symbol_and_the_days_away(self):
        """The drop has to be observable in the log, like the quote filters."""
        reason = ec.block_reason(self._annotated(symbol="AMZN",
                                                 earnings_days_away=18))
        assert "AMZN" in reason and "18" in reason


# ---------------------------------------------------------------------------
# The three positions that prompted this — a regression record
# ---------------------------------------------------------------------------

class TestTheOctober2026Positions:
    """
    Entry conditions of the three live positions open on 2026-10-11. Every
    one of them held a print inside its window, so every one of them is a
    trade the gate should have refused.
    """

    CASES = [
        # symbol, entry date, expiry, earnings date (confirmed, yfinance)
        ("IBM",   dt.date(2026, 9, 21), "2026-10-23", dt.date(2026, 10, 21)),
        ("GOOGL", dt.date(2026, 10, 8), "2026-11-06", dt.date(2026, 10, 28)),
        ("AMZN",  dt.date(2026, 10, 7), "2026-11-06", dt.date(2026, 10, 29)),
    ]

    @pytest.mark.parametrize("symbol,entry,expiry,earn", CASES)
    def test_each_of_the_three_would_have_been_refused(self, cache, symbol,
                                                       entry, expiry, earn):
        data_dir, fund_dir = cache
        _write_upcoming(fund_dir, symbol, [earn], fetched_at=entry)
        s = {"symbol": symbol, "signal": "SELL_PUT", "strike": 1.0,
             "close": 1.0, "otm_pct": 5.0, "expiry": expiry}
        ec.annotate(s, today=entry, data_dir=data_dir, fund_dir=fund_dir)
        assert s["earnings_before_expiry"] is True
        assert ec.block_reason(s) is not None


# ---------------------------------------------------------------------------
# refresh_if_stale — keeping the forward calendar current
# ---------------------------------------------------------------------------

class TestRefreshIfStale:
    """
    A confirmed date beats a projection, so the cache has to be refreshed.
    It must refresh only when needed (yfinance is slow and rate-limited) and
    must never be able to take the scanner down.
    """

    def test_a_fresh_cache_is_not_refetched(self, cache):
        _, fund_dir = cache
        _write_upcoming(fund_dir, "IBM", [dt.date(2026, 10, 21)],
                        fetched_at=dt.date(2026, 10, 10))
        called = []
        out = ec.refresh_if_stale(["IBM"], fund_dir=fund_dir,
                                  today=dt.date(2026, 10, 11),
                                  fetcher=lambda s: called.append(s))
        assert called == []
        assert out["fresh"] == 1 and out["refreshed"] == 0

    def test_a_stale_cache_is_refetched(self, cache):
        _, fund_dir = cache
        _write_upcoming(fund_dir, "IBM", [dt.date(2026, 10, 21)],
                        fetched_at=dt.date(2026, 8, 1))
        called = []
        out = ec.refresh_if_stale(["IBM"], fund_dir=fund_dir,
                                  today=dt.date(2026, 10, 11),
                                  fetcher=lambda s: called.append(s))
        assert called == ["IBM"]
        assert out["refreshed"] == 1

    def test_a_missing_cache_is_fetched(self, cache):
        _, fund_dir = cache
        called = []
        ec.refresh_if_stale(["ZZZZ"], fund_dir=fund_dir,
                            today=dt.date(2026, 10, 11),
                            fetcher=lambda s: called.append(s))
        assert called == ["ZZZZ"]

    def test_one_symbol_failing_does_not_stop_the_rest(self, cache):
        """yfinance throttles. A single failure must not cost the whole run."""
        _, fund_dir = cache
        done = []

        def flaky(sym):
            if sym == "AMD":
                raise RuntimeError("429 Too Many Requests")
            done.append(sym)

        out = ec.refresh_if_stale(["NVDA", "AMD", "IBM"], fund_dir=fund_dir,
                                  today=dt.date(2026, 10, 11), fetcher=flaky)
        assert done == ["NVDA", "IBM"]
        assert out["failed"] == 1 and out["refreshed"] == 2

    def test_refresh_never_raises(self, cache):
        """Called at scanner startup — an exception here is a dead scanner."""
        _, fund_dir = cache
        out = ec.refresh_if_stale(
            ["IBM"], fund_dir="/nonexistent/path/at/all",
            today=dt.date(2026, 10, 11),
            fetcher=lambda s: (_ for _ in ()).throw(OSError("boom")))
        assert isinstance(out, dict)
