"""
earnings_calendar.py — know where the next earnings print is, and refuse to
sell a cash-secured put across it.

On 2026-10-11 all three open Vault 76 positions held an earnings print
inside their window (IBM $220P exp 10-23 over a 10-21 report, GOOGL $335P
and AMZN $245P exp 11-06 over 10-28 and 10-29). Nothing in the system knew.
OVERSEER_SYSTEM tells the LLM to veto on "earnings within ~5 trading days"
but no signal ever carried an earnings date, so the instruction was dead
text.

## The measurement behind the gate

walk_forward_scavenger over the 18-symbol watchlist, sell_calls=False so
each cycle's P&L is complete (with the wheel on, `put_assigned` books $0
and the loss surfaces in a later covered-call event — measuring it that way
reverses the sign of the answer):

    no earnings in window   n=3320   mean $+23.65   p5 $-19    worst $-4,176
    earnings IN window      n=1040   mean $-33.46   p5 $-312   worst $-13,012

Holding a print WINS MORE OFTEN (87.6% vs 82.9%) and loses far bigger. The
5th percentile is 16x worse. Refusing the whole population is worth
+$34,803 across the backtest for 23.9% of cycles given up.

## Why the gate is blanket and not cushion-aware

The appealing version — allow it when the OTM cushion exceeds the move the
symbol actually makes at earnings — does not work. Bucketed by
otm% / median-print%, every bucket still loses and the rank correlation
between cushion ratio and P&L is +0.038:

    ratio 0.60–0.80  n=266  mean $-86.35      ratio 1.00–1.25  n=245  mean $-26.62
    ratio 0.80–1.00  n=106  mean $ -9.92      ratio 1.25–1.60  n=145  mean $-17.50
                                              ratio 1.60+      n=257  mean $ -5.51

So `typical_earnings_move` is published as advisory context for the LLM and
gates nothing. A measured refinement that was NOT taken: blocking only
prints in the first two thirds of the window scored +$37,552 (vs +$34,803),
but the last-third bucket's +$8.48 mean sits inside its own noise and the
advantage is partly an artifact of covers filling before the print, which
is not knowable at entry. Not worth a fitted parameter.

## Unknown earnings DOES block (reversed 2026-10-27)

This module shipped with the opposite rule: unknown passed, so a broken
yfinance feed could not halt the income strategy. On 2026-10-26 that leniency
turned one defect into a trade. AMZN's cache crossed CACHE_MAX_AGE_DAYS
between two scans five minutes apart, `_project` returned a date THREE DAYS
IN THE PAST (the next print 2026-10-29 minus PROJECTION_MARGIN_DAYS, with
nothing checking the result was still in the future), `annotate` read that as
an affirmative "no print in this window", and a $23,500 cash-secured put went
out two days before a report that moves the stock 7%. Six of fourteen symbols
were ungated the same way, mid-earnings-season.

So the gate now fails CLOSED: the only way through is a date that was found
and lands outside the window. `_project` is clamped so it can never answer
with the past, and `maybe_refresh` lets a long-running scanner re-scrape its
own calendar instead of degrading until someone restarts it. The cost of
being blind is a symbol that stops trading; the cost of being wrong is p5
$-312 and worst $-13,012 on the measurement above. The one exemption is an
instrument with no reporting history at all — an ETF must not be refused
forever by a check about earnings.
"""
import datetime as dt
import os

import numpy as np
import pandas as pd

FUND_DIR = os.path.join(os.path.dirname(__file__), "..", "data",
                        "fundamentals")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# An upcoming-earnings cache older than this is no longer "confirmed" — a
# date can move, and a week-old scrape may have missed a reschedule.
CACHE_MAX_AGE_DAYS = 7

# A projected date is accurate to a few days. Pull it EARLIER so a print
# that lands later than projected still falls inside the window being
# tested, rather than slipping just past expiry and escaping the gate.
PROJECTION_MARGIN_DAYS = 5

# How far a projected date may be overdue before it is assumed to have been
# reported. A company that reports a week later than its own cadence must not
# make the projection skip a whole quarter: GOOGL's cadence put its print at
# 2026-10-21 and it actually reported 2026-10-28, which on 2026-10-26 had the
# projection answering 2027-01-15 and the gate calling the window clean two
# days before the print.
OVERDUE_GRACE_DAYS = 21

# Minimum past reports before a projection or a move estimate is trusted.
MIN_REPORTS_TO_PROJECT = 3
MIN_REPORTS_FOR_MOVE   = 8

# A history file whose last report is older than this cannot place the next
# print — the cadence is unknown and projecting would be invention.
MAX_HISTORY_AGE_DAYS = 200


# ---------------------------------------------------------------------------
# Cache readers
# ---------------------------------------------------------------------------

def _upcoming_path(symbol: str, fund_dir: str) -> str:
    return os.path.join(fund_dir, f"{symbol.lower()}_earnings_upcoming.csv")


def _history_path(symbol: str, fund_dir: str) -> str:
    return os.path.join(fund_dir, f"{symbol.lower()}_earnings.csv")


def _read_cached(symbol: str, on_date: dt.date, fund_dir: str,
                 max_age: "int | None" = None) -> "dt.date | None":
    """
    Next cached report strictly after `on_date`, or None when the cache is
    missing, unreadable, or older than `max_age` days (None = any age).
    Staleness is measured from the fetched_at column the fetcher stamps on
    every row.
    """
    path = _upcoming_path(symbol, fund_dir)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty or "date" not in df.columns:
        return None

    if (max_age is not None and "fetched_at" in df.columns
            and df["fetched_at"].notna().any()):
        fetched = pd.to_datetime(df["fetched_at"], errors="coerce").max()
        if pd.isna(fetched):
            return None
        if (on_date - fetched.date()).days > max_age:
            return None

    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    future = sorted(d.date() for d in dates if d.date() > on_date)
    return future[0] if future else None


def _read_confirmed(symbol: str, on_date: dt.date,
                    fund_dir: str) -> "dt.date | None":
    """Next report from a cache fresh enough to still count as confirmed."""
    return _read_cached(symbol, on_date, fund_dir,
                        max_age=CACHE_MAX_AGE_DAYS)


def _read_history(symbol: str, fund_dir: str) -> "list[dt.date]":
    path = _history_path(symbol, fund_dir)
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    if df.empty or "date" not in df.columns:
        return []
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    return sorted(d.date() for d in dates)


def _project(symbol: str, on_date: dt.date,
             fund_dir: str) -> "dt.date | None":
    """
    Next report projected from the symbol's own reporting cadence: the median
    gap between its recent reports, walked forward from the last one, minus
    PROJECTION_MARGIN_DAYS. None when history is too short or too old.
    """
    past = [d for d in _read_history(symbol, fund_dir) if d <= on_date]
    if len(past) < MIN_REPORTS_TO_PROJECT:
        return None
    if (on_date - past[-1]).days > MAX_HISTORY_AGE_DAYS:
        return None

    gaps = [(b - a).days for a, b in zip(past[-5:], past[-4:])]
    gaps = [g for g in gaps if 60 <= g <= 120]      # quarterly, sanity-bounded
    step = int(np.median(gaps)) if gaps else 91

    # One step from the last report, then forward only past dates that are
    # more than OVERDUE_GRACE_DAYS behind. The grace never applies to the last
    # report itself — that one is on file as having happened.
    when = past[-1] + dt.timedelta(days=step)
    while when <= on_date - dt.timedelta(days=OVERDUE_GRACE_DAYS):
        when += dt.timedelta(days=step)
    when -= dt.timedelta(days=PROJECTION_MARGIN_DAYS)

    # The margin can pull the date to or before `on_date` when the print is
    # only days out — AMZN on 2026-10-26 projected 2026-10-24, three days
    # PAST, and `annotate` tests `today < when <= expiry`, so the gate read
    # that as proof the window was clean. A margin states uncertainty, and the
    # honest reading of "could be as early as last Saturday" is "any day now",
    # not "already happened". Tomorrow is inside every window being tested,
    # which is the safe resolution and, on 2026-10-26, also the true one.
    return max(when, on_date + dt.timedelta(days=1))


def has_no_earnings_history(symbol: str, fund_dir: str = FUND_DIR) -> bool:
    """
    True only when nothing on file says this symbol has ever reported — i.e.
    there is no history file for it. This is the one thing that earns a pass
    from a gate that otherwise fails closed, so it answers the narrow
    question "is this even an instrument with earnings" and nothing wider.

    A file that exists but yields no usable dates returns False: unreadable is
    not the same as "does not report", and that difference decides whether a
    put goes out blind.
    """
    try:
        return not os.path.exists(_history_path(symbol, fund_dir))
    except Exception:
        return False


def next_earnings(symbol: str, on_date: dt.date,
                  fund_dir: str = FUND_DIR) -> "tuple[dt.date, str] | None":
    """
    (date, source) of the next report after `on_date`, in descending order of
    confidence: "confirmed" (a scrape still inside CACHE_MAX_AGE_DAYS),
    "stale cache" (a scrape past it, but still a date someone published), then
    "projected" (the symbol's own cadence). None when none of the three is
    available — the caller must treat that as unknown, not safe, which since
    2026-10-27 `block_reason` does.

    The stale rung exists because the 8-day-old AMZN cache said 2026-10-29 —
    the correct date — and discarding it for a cadence guess is what returned
    2026-10-24, three days in the past, and sold a put into the print.
    """
    try:
        confirmed = _read_confirmed(symbol, on_date, fund_dir)
        if confirmed is not None:
            return confirmed, "confirmed"
        stale = _read_cached(symbol, on_date, fund_dir)
        if stale is not None:
            return stale, "stale cache"
        projected = _project(symbol, on_date, fund_dir)
        if projected is not None:
            return projected, "projected"
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# How big is this symbol's print? (advisory only)
# ---------------------------------------------------------------------------

def typical_earnings_move(symbol: str, data_dir: str = DATA_DIR,
                          fund_dir: str = FUND_DIR,
                          min_reports: int = MIN_REPORTS_FOR_MOVE
                          ) -> "float | None":
    """
    Median absolute one-session reaction to a report, in percent, measured
    from local history. Reports timestamped at or after noon land after the
    close, so the reaction is the NEXT session — using the report day's own
    close would measure the day before the news.

    Advisory: shown to the LLM, gates nothing (see the module docstring).
    """
    try:
        hist = os.path.join(data_dir, f"{symbol.lower()}_history.csv")
        if not os.path.exists(hist):
            return None
        df = pd.read_csv(hist, parse_dates=["datetime"])
        if df.empty:
            return None
        df["_d"] = df["datetime"].dt.date
        df = df.sort_values("_d").reset_index(drop=True)
        df["_ret"] = df["close"].pct_change()
        sessions = list(df["_d"])
        index_of = {d: i for i, d in enumerate(sessions)}

        reports = pd.read_csv(_history_path(symbol, fund_dir)) \
            if os.path.exists(_history_path(symbol, fund_dir)) else None
        if reports is None or reports.empty or "date" not in reports.columns:
            return None
        stamps = pd.to_datetime(reports["date"], errors="coerce").dropna()

        moves = []
        for ts in stamps:
            after_close = ts.hour >= 12
            day = ts.date()
            cutoff = day if after_close else day - dt.timedelta(days=1)
            reaction = next((d for d in sessions if d > cutoff), None)
            if reaction is None:
                continue
            i = index_of[reaction]
            if i > 0 and not pd.isna(df["_ret"].iloc[i]):
                moves.append(abs(float(df["_ret"].iloc[i])) * 100)

        if len(moves) < min_reports:
            return None
        return round(float(np.median(moves)), 2)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Annotation + the gate
# ---------------------------------------------------------------------------

def _expiry_of(signal: dict, today: dt.date) -> "dt.date | None":
    expiry = signal.get("expiry")
    if expiry:
        try:
            return pd.to_datetime(expiry).date()
        except Exception:
            pass
    dte = signal.get("dte")
    if dte is not None:
        try:
            return today + dt.timedelta(days=int(dte))
        except Exception:
            pass
    return None


def annotate(signal: dict, today: "dt.date | None" = None,
             data_dir: str = DATA_DIR, fund_dir: str = FUND_DIR) -> dict:
    """
    Attach the earnings fields to `signal` in place and return it. Never
    raises — a corrupt cache would otherwise kill the scan for every symbol.

    Fields, all of which reach the LLM because build_prompt dumps the signal:
      earnings_date            "YYYY-MM-DD" or "unknown"
      earnings_source          "confirmed" | "projected" | None
      earnings_days_away       int or None
      earnings_before_expiry   True | False | None (None = unknown)
      earnings_move_pct        median absolute reaction, advisory
      earnings_no_history      True only for a symbol with no reports on file

    `earnings_no_history` is the gate's one exemption, so it is computed on
    its own and defaults to False: a dict that never reaches this function, or
    reaches an older copy of it, must not win the lenient path by omission.
    """
    today = today or dt.date.today()
    symbol = signal.get("symbol", "")

    signal["earnings_date"]          = "unknown"
    signal["earnings_source"]        = None
    signal["earnings_days_away"]     = None
    signal["earnings_before_expiry"] = None
    signal["earnings_move_pct"]      = None
    signal["earnings_no_history"]    = has_no_earnings_history(symbol, fund_dir)

    try:
        found = next_earnings(symbol, today, fund_dir=fund_dir)
        if found is None:
            return signal
        when, source = found
        expiry = _expiry_of(signal, today)

        signal["earnings_date"]      = str(when)
        signal["earnings_source"]    = source
        signal["earnings_days_away"] = (when - today).days
        if expiry is not None:
            # A print ON expiry day is the worst case, not a miss.
            signal["earnings_before_expiry"] = today < when <= expiry
        signal["earnings_move_pct"] = typical_earnings_move(
            symbol, data_dir=data_dir, fund_dir=fund_dir)
    except Exception:
        # Everything resets to unknown, which now blocks. earnings_no_history
        # is deliberately NOT reset: it was computed outside the try and a
        # corrupt cache is not evidence that the symbol has no earnings.
        signal["earnings_date"]          = "unknown"
        signal["earnings_source"]        = None
        signal["earnings_days_away"]     = None
        signal["earnings_before_expiry"] = None
        signal["earnings_move_pct"]      = None
    return signal


def block_reason(signal: dict) -> "str | None":
    """
    The gate. A human-readable reason to refuse this signal, or None to let
    it through.

    Only SELL_PUT is gated: a covered call across a print risks being called
    away, which caps a gain rather than opening a loss.

    FAILS CLOSED. The only way through is `earnings_before_expiry is False` —
    a date that was found and affirmatively lands outside the window. Unknown,
    stale, corrupt and untestable all block. The old rule blocked only on
    `is True`, which is how a projection three days in the past sold an AMZN
    put two days before the print on 2026-10-26: there was no second check for
    one defect to fail. See the module docstring.
    """
    if signal.get("signal") != "SELL_PUT":
        return None

    before = signal.get("earnings_before_expiry")
    if before is False:
        return None

    sym    = signal.get("symbol", "?")
    if before is not True:
        if signal.get("earnings_no_history") is True:
            return None                 # nothing on file ever reports — an ETF
        when = signal.get("earnings_date")
        if when and when != "unknown":
            return (f"{sym} earnings {when} but this signal carries no expiry "
                    f"to test it against — the gate fails closed")
        return (f"{sym} earnings date unknown — refusing to secure a put "
                f"blind (cache missing, stale or unreadable)")

    when   = signal.get("earnings_date", "?")
    days   = signal.get("earnings_days_away")
    source = signal.get("earnings_source") or "unknown source"
    move   = signal.get("earnings_move_pct")

    parts = [f"{sym} earnings {when}"]
    if days is not None:
        parts.append(f"in {days}d")
    parts.append(f"({source})")
    parts.append("falls before expiry")
    if move is not None:
        parts.append(f"— typical print moves {move:.1f}%")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Keeping the forward calendar current
# ---------------------------------------------------------------------------

def cache_age_days(symbol: str, fund_dir: str = FUND_DIR,
                   today: "dt.date | None" = None) -> "int | None":
    """Age of the symbol's upcoming-earnings cache in days, None if absent."""
    today = today or dt.date.today()
    path = _upcoming_path(symbol, fund_dir)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if "fetched_at" not in df.columns or df.empty:
            return None
        fetched = pd.to_datetime(df["fetched_at"], errors="coerce").max()
        if pd.isna(fetched):
            return None
        return (today - fetched.date()).days
    except Exception:
        return None


# How often a running scan may re-check the forward calendar. The check costs
# one small CSV read per symbol and scrapes only what is actually stale, but
# scans run every five minutes and yfinance throttles — and with the gate
# failing closed, this interval is also how long a failed fetch keeps a symbol
# out of the market.
REFRESH_INTERVAL_HOURS = 1

_last_refresh_at: "dt.datetime | None" = None


def maybe_refresh(symbols: list, fund_dir: str = FUND_DIR,
                  now: "dt.datetime | None" = None,
                  fetcher=None) -> "dict | None":
    """
    `refresh_if_stale` throttled to once every REFRESH_INTERVAL_HOURS, for
    calling from the scan loop. Returns its counts, or None when it was too
    soon to look again.

    This is the third defect of 2026-10-26: the refresh ran once, at startup.
    A process up for eight days crossed CACHE_MAX_AGE_DAYS, lost every
    confirmed date, and had no route back — the live fix was a restart, which
    is not a fix. Now that the gate fails closed a stale calendar stops
    trading rather than trading blind, which makes self-healing the difference
    between a quiet degradation and a quiet outage.
    """
    global _last_refresh_at
    now = now or dt.datetime.now()
    if (_last_refresh_at is not None
            and now - _last_refresh_at
                < dt.timedelta(hours=REFRESH_INTERVAL_HOURS)):
        return None
    _last_refresh_at = now
    return refresh_if_stale(symbols, fund_dir=fund_dir, today=now.date(),
                            fetcher=fetcher)


def refresh_if_stale(symbols: list, fund_dir: str = FUND_DIR,
                     today: "dt.date | None" = None,
                     fetcher=None) -> dict:
    """
    Re-scrape the forward calendar for symbols whose cache is missing or older
    than CACHE_MAX_AGE_DAYS. Returns {"fresh", "refreshed", "failed"}.

    Called at scanner startup, so it never raises and never lets one symbol's
    failure cost the others — yfinance throttles, and a dead scanner is a far
    worse outcome than a projected date.
    """
    out = {"fresh": 0, "refreshed": 0, "failed": 0}
    if fetcher is None:
        try:
            from fetch_fundamentals import fetch as fetcher
        except Exception as exc:
            print(f"  [Earnings] calendar refresh unavailable ({exc})")
            return out

    for sym in symbols:
        try:
            age = cache_age_days(sym, fund_dir=fund_dir, today=today)
            if age is not None and age <= CACHE_MAX_AGE_DAYS:
                out["fresh"] += 1
                continue
            fetcher(sym)
            out["refreshed"] += 1
        except Exception:
            out["failed"] += 1
    return out
