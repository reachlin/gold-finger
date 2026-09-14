"""Tests for the market-check sleep — chunked, wall-clock-anchored waiting.

Context (2026-09-14): the overseer sat ALIVE but frozen 15+ minutes past its
scheduled Monday 09:00 ET wake and would have missed the open. It had issued a
single `time.sleep(86397)` on Sunday 09:00 ET and never returned from it
(same pid, 0% CPU, elapsed 24h15m vs a 23h59m57s sleep). System suspend was
ruled out — caffeinate held PreventSystemSleep for the full duration and pmset
logged no Sleep/Wake events.

The fix does not try to explain the overrun: it makes one survivable. The wait
is now anchored to an ABSOLUTE target datetime and re-derives the remaining
time from the wall clock on every chunk, so any single chunk returning late
(or early) self-corrects on the next pass instead of silently eating a session.

Chunking also fixes a latent DST bug: the old code froze a fixed duration up
front, so a wait spanning a DST transition landed an hour off.
"""
import os, sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(__file__))
from real_overseer import (
    next_market_check_at,
    seconds_until_market_check,
    _sleep_until_next_check,
    _SLEEP_CHUNK_S,
)

ET = ZoneInfo("America/New_York")


def et(y, m, d, hh=0, mm=0, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=ET)


class FakeClock:
    """Controllable clock + sleep. `drift` scales how far each sleep advances.

    drift=1.0 -> sleep is exact; drift=2.0 -> every sleep takes twice as long
    as asked (the observed failure mode, exaggerated).
    """

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
        # Advance in ABSOLUTE time (via UTC). Adding a timedelta straight to an
        # aware ET datetime does wall-clock arithmetic, which would make the
        # DST cases simulate the wrong elapsed time.
        utc = self.now_dt.astimezone(timezone.utc) + timedelta(seconds=seconds)
        self.now_dt = utc.astimezone(ET)


# --------------------------------------------------------------------------
# target resolution
# --------------------------------------------------------------------------

def test_target_is_today_when_before_9am():
    assert next_market_check_at(et(2026, 9, 14, 6, 30)) == et(2026, 9, 14, 9, 0)


def test_target_rolls_to_tomorrow_at_exactly_9am():
    # The 2026-09-13 case: the check itself ran at 09:00:03 ET.
    assert next_market_check_at(et(2026, 9, 13, 9, 0, 3)) == et(2026, 9, 14, 9, 0)


def test_target_rolls_to_tomorrow_when_after_9am():
    assert next_market_check_at(et(2026, 9, 13, 16, 4)) == et(2026, 9, 14, 9, 0)


def test_seconds_helper_matches_target():
    now = et(2026, 9, 13, 9, 0, 3)
    # Preserved for callers//tests that want the scalar: 23h59m57s.
    assert seconds_until_market_check(now) == 86397.0


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------

def test_no_single_sleep_exceeds_chunk():
    clock = FakeClock(et(2026, 9, 13, 9, 0, 3))
    _sleep_until_next_check(clock.now(), sleep=clock.sleep, now=clock.now)
    assert clock.sleeps, "expected at least one sleep"
    assert max(clock.sleeps) <= _SLEEP_CHUNK_S


def test_wakes_at_target_not_before_or_after():
    start = et(2026, 9, 13, 9, 0, 3)
    clock = FakeClock(start)
    _sleep_until_next_check(start, sleep=clock.sleep, now=clock.now)
    assert clock.now() >= et(2026, 9, 14, 9, 0)
    # Exact clock -> should not overshoot by more than a rounding sliver.
    assert (clock.now() - et(2026, 9, 14, 9, 0)).total_seconds() < 1


def test_total_slept_equals_full_interval():
    start = et(2026, 9, 13, 9, 0, 3)
    clock = FakeClock(start)
    _sleep_until_next_check(start, sleep=clock.sleep, now=clock.now)
    assert abs(sum(clock.sleeps) - 86397.0) < 1


def test_returns_immediately_when_target_already_passed():
    # now() is past the target the moment we start -> nothing to wait for.
    start = et(2026, 9, 14, 8, 59)
    clock = FakeClock(et(2026, 9, 14, 9, 30))  # wall clock already past 09:00
    _sleep_until_next_check(start, sleep=clock.sleep, now=clock.now)
    assert clock.sleeps == []


# --------------------------------------------------------------------------
# the regression: a chunk that returns late must self-heal
# --------------------------------------------------------------------------

def test_overrunning_sleep_does_not_extend_the_wait():
    """Each chunk takes 2x as long as requested — the wait must still end
    at/just past the target, never a chunk beyond it."""
    start = et(2026, 9, 13, 9, 0, 3)
    clock = FakeClock(start, drift=2.0)
    _sleep_until_next_check(start, sleep=clock.sleep, now=clock.now)
    target = et(2026, 9, 14, 9, 0)
    assert clock.now() >= target
    # Self-healing: overshoot is bounded by one chunk's worth of drift,
    # NOT by the whole remaining interval.
    overshoot = (clock.now() - target).total_seconds()
    assert overshoot <= _SLEEP_CHUNK_S


def test_single_massive_overrun_still_exits():
    """The observed hang, modelled: the first chunk returns ~13h late.
    The loop must notice the wall clock passed the target and stop."""
    start = et(2026, 9, 13, 9, 0, 3)
    clock = FakeClock(start)

    real_sleep = clock.sleep
    calls = {"n": 0}

    def hiccup(seconds):
        calls["n"] += 1
        if calls["n"] == 1:
            clock.sleeps.append(seconds)
            clock._advance(26 * 3600)             # way past the target
            return
        real_sleep(seconds)

    _sleep_until_next_check(start, sleep=hiccup, now=clock.now)
    assert calls["n"] == 1, "should not sleep again after the clock passed target"
    assert clock.now() >= et(2026, 9, 14, 9, 0)


def test_chunk_is_capped_at_one_hour():
    assert 0 < _SLEEP_CHUNK_S <= 3600


# --------------------------------------------------------------------------
# DST — the latent bug the old fixed-duration sleep had
# --------------------------------------------------------------------------

def test_spring_forward_lands_on_wall_clock_9am():
    # 2027-03-14 is US spring-forward. Waiting from the 13th must end at
    # 09:00 ET wall clock on the 14th (a 23h real-time interval, not 24h).
    start = et(2027, 3, 13, 9, 0, 3)
    target = next_market_check_at(start)
    assert target == et(2027, 3, 14, 9, 0)
    clock = FakeClock(start)
    _sleep_until_next_check(start, sleep=clock.sleep, now=clock.now)
    assert clock.now().hour == 9
    assert clock.now().date().isoformat() == "2027-03-14"
    # 23 real hours, because the wall clock skipped 02:00->03:00.
    assert abs(sum(clock.sleeps) - 23 * 3600 + 3) < 60


def test_fall_back_lands_on_wall_clock_9am():
    # 2026-11-01 is US fall-back -> 25 real hours between 09:00 wall clocks.
    start = et(2026, 10, 31, 9, 0, 3)
    clock = FakeClock(start)
    _sleep_until_next_check(start, sleep=clock.sleep, now=clock.now)
    assert clock.now().hour == 9
    assert clock.now().date().isoformat() == "2026-11-01"
    assert abs(sum(clock.sleeps) - 25 * 3600 + 3) < 60
