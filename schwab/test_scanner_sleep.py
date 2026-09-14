"""Tests for live_scanner's end-of-day wait until the next market open.

Same defect as the market-check sleep (see test_market_check_sleep.py), on the
longer path: `_sleep_until_market_open` issued ONE `time.sleep(secs)` covering
up to 64.9h across a weekend. A single overrunning sleep on that path costs a
whole session — this is the call that printed Friday's
"Market closed. Sleeping 64.9h until 2026-09-14 09:00 ET."

Fixed the same way: an absolute target plus <=1h chunks re-derived from the
wall clock, so a late chunk self-corrects instead of silently overshooting.
"""
import os, sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(__file__))
from live_scanner import (
    next_market_open_at,
    _sleep_until_market_open,
    _SLEEP_CHUNK_S,
)

ET = ZoneInfo("America/New_York")


def et(y, m, d, hh=0, mm=0, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=ET)


class FakeClock:
    def __init__(self, start, drift=1.0):
        self.now_dt = start
        self.drift = drift
        self.sleeps = []

    def now(self):
        return self.now_dt

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self._advance(seconds * self.drift)

    def _advance(self, seconds):
        # Absolute-time advance (via UTC), so the DST cases are honest.
        utc = self.now_dt.astimezone(timezone.utc) + timedelta(seconds=seconds)
        self.now_dt = utc.astimezone(ET)


# --------------------------------------------------------------------------
# target resolution — including the weekend skip
# --------------------------------------------------------------------------

def test_friday_close_targets_monday_open():
    # The real 2026-09-11 EOD: Friday 16:04 ET -> Monday 09:00 ET.
    target = next_market_open_at(et(2026, 9, 11, 16, 4))
    assert target == et(2026, 9, 14, 9, 0)
    assert target.weekday() == 0


def test_friday_close_interval_is_64_9_hours():
    # Matches the figure the scanner actually logged.
    now = et(2026, 9, 11, 16, 4)
    secs = (next_market_open_at(now) - now).total_seconds()
    assert abs(secs / 3600 - 64.9) < 0.1


def test_saturday_targets_monday():
    assert next_market_open_at(et(2026, 9, 12, 10, 0)) == et(2026, 9, 14, 9, 0)


def test_sunday_targets_monday():
    assert next_market_open_at(et(2026, 9, 13, 10, 0)) == et(2026, 9, 14, 9, 0)


def test_weekday_close_targets_next_morning():
    assert next_market_open_at(et(2026, 9, 15, 16, 4)) == et(2026, 9, 16, 9, 0)


def test_before_open_targets_same_day():
    assert next_market_open_at(et(2026, 9, 15, 6, 0)) == et(2026, 9, 15, 9, 0)


def test_target_never_lands_on_a_weekend():
    # Sweep a full week of closes; every target must be Mon-Fri 09:00.
    for day in range(7, 14):
        t = next_market_open_at(et(2026, 9, day, 16, 30))
        assert t.weekday() < 5, f"{t} is a weekend"
        assert (t.hour, t.minute) == (9, 0)


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------

def test_no_single_sleep_exceeds_chunk_over_a_weekend():
    start = et(2026, 9, 11, 16, 4)
    clock = FakeClock(start)
    _sleep_until_market_open(start, sleep=clock.sleep, now=clock.now)
    assert max(clock.sleeps) <= _SLEEP_CHUNK_S
    assert len(clock.sleeps) >= 64        # 64.9h in <=1h chunks


def test_weekend_wait_ends_at_monday_open():
    start = et(2026, 9, 11, 16, 4)
    clock = FakeClock(start)
    _sleep_until_market_open(start, sleep=clock.sleep, now=clock.now)
    assert clock.now() >= et(2026, 9, 14, 9, 0)
    assert (clock.now() - et(2026, 9, 14, 9, 0)).total_seconds() < 1


def test_total_slept_matches_interval():
    start = et(2026, 9, 11, 16, 4)
    clock = FakeClock(start)
    _sleep_until_market_open(start, sleep=clock.sleep, now=clock.now)
    expected = (et(2026, 9, 14, 9, 0) - start).total_seconds()
    assert abs(sum(clock.sleeps) - expected) < 1


def test_returns_immediately_when_open_already_passed():
    start = et(2026, 9, 14, 8, 0)
    clock = FakeClock(et(2026, 9, 14, 9, 30))
    _sleep_until_market_open(start, sleep=clock.sleep, now=clock.now)
    assert clock.sleeps == []


# --------------------------------------------------------------------------
# the regression
# --------------------------------------------------------------------------

def test_overrunning_chunks_do_not_extend_the_wait():
    start = et(2026, 9, 11, 16, 4)
    clock = FakeClock(start, drift=2.0)
    _sleep_until_market_open(start, sleep=clock.sleep, now=clock.now)
    target = et(2026, 9, 14, 9, 0)
    assert clock.now() >= target
    assert (clock.now() - target).total_seconds() <= _SLEEP_CHUNK_S


def test_single_massive_overrun_still_exits():
    """One chunk returns ~3 days late — the loop must stop, not sleep on."""
    start = et(2026, 9, 11, 16, 4)
    clock = FakeClock(start)
    calls = {"n": 0}

    def hiccup(seconds):
        calls["n"] += 1
        if calls["n"] == 1:
            clock.sleeps.append(seconds)
            clock._advance(72 * 3600)
            return
        clock.sleep(seconds)

    _sleep_until_market_open(start, sleep=hiccup, now=clock.now)
    assert calls["n"] == 1
    assert clock.now() >= et(2026, 9, 14, 9, 0)


def test_chunk_is_capped_at_one_hour():
    assert 0 < _SLEEP_CHUNK_S <= 3600


# --------------------------------------------------------------------------
# DST
# --------------------------------------------------------------------------

def test_weekend_spanning_spring_forward_ends_at_wall_clock_9am():
    # 2027-03-12 Fri close -> 2027-03-15 Mon open, across spring-forward.
    start = et(2027, 3, 12, 16, 4)
    clock = FakeClock(start)
    _sleep_until_market_open(start, sleep=clock.sleep, now=clock.now)
    assert clock.now().date().isoformat() == "2027-03-15"
    assert clock.now().hour == 9
    # One hour is skipped, so real elapsed is 63.93h, not 64.93h.
    assert abs(sum(clock.sleeps) / 3600 - 63.93) < 0.1


def test_weekend_spanning_fall_back_ends_at_wall_clock_9am():
    # 2026-10-30 Fri close -> 2026-11-02 Mon open, across fall-back.
    start = et(2026, 10, 30, 16, 4)
    clock = FakeClock(start)
    _sleep_until_market_open(start, sleep=clock.sleep, now=clock.now)
    assert clock.now().date().isoformat() == "2026-11-02"
    assert clock.now().hour == 9
    assert abs(sum(clock.sleeps) / 3600 - 65.93) < 0.1
