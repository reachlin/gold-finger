"""
A dated exclusion list: stay out of a symbol until a chosen date.

Written 2026-10-28. The book had just closed an AMZN put one day before the
print, and "make sure it stays away from AMZN" turned out to be unexpressible:
the earnings gate only covers the window around a report, and the 20%
concentration cap deliberately does not bind the first contract in a name, so
a freshly-unheld AMZN could take 52% of the book on one contract. See
test_symbol_exclusions.py for the measurement.

## Scope: SELL_PUT only

The same scope the earnings gate chose, for the same reason — that is the one
signal which opens new exposure. BUY_TO_CLOSE in particular is never blocked:
an exclusion means "do not open this", never "cannot close this". AutoClose
iterates the ledger rather than WATCHLIST, so a resting cover never consults
this module, but the guard is here anyway so the list can never become a trap
for a position that is already on.

HOLD_SHARES is not gated because nothing in the overseer places an equity
order; it is an advisory mode flag whose only effect is to suppress the
Scavenger on that symbol, which already points the same way.

## Fails closed on a bad date

A symbol named here with an unreadable date stays excluded. A typo must not
silently re-admit the symbol someone asked to keep out — that is exactly the
fail-open shape 590e756 was written to delete. The cost of being wrong in this
direction is a trade not taken; in the other direction it is a position nobody
authorised.
"""
import datetime as dt

try:
    from strategy_params import SYMBOL_EXCLUSIONS
except Exception:                       # pragma: no cover - config must not
    SYMBOL_EXCLUSIONS = {}              # take the scanner down with it


# The one signal that opens new exposure. Everything else passes.
GATED_SIGNALS = ("SELL_PUT",)


def _parse(when) -> "dt.date | None":
    """
    The configured value as a date, or None when it cannot be read.

    None is the fail-closed sentinel, not "no expiry": every caller treats an
    unreadable date as still excluded. Deliberately strict — only
    `YYYY-MM-DD`. Accepting `01/01/2027` would mean guessing day-vs-month
    order on a value that decides whether a symbol trades.
    """
    if isinstance(when, dt.datetime):
        return when.date()
    if isinstance(when, dt.date):
        return when
    if not isinstance(when, str):
        return None
    try:
        return dt.date.fromisoformat(when.strip())
    except ValueError:
        return None


def _norm(symbol) -> str:
    return symbol.strip().upper() if isinstance(symbol, str) else ""


def _lookup(symbol, exclusions):
    """
    (present, parsed_date) for this symbol.

    Matching is case- and whitespace-insensitive on both sides, because the
    list is hand-edited config and `"amzn"` or `" AMZN "` must not quietly
    fail to match the ticker it obviously names. It is still an exact ticker
    match: "AMZ" and "AMZNX" are different symbols.
    """
    sym = _norm(symbol)
    if not sym:
        return False, None
    for key, when in (exclusions or {}).items():
        if _norm(key) == sym:
            return True, _parse(when)
    return False, None


def is_excluded(symbol, today: "dt.date | None" = None,
                exclusions: "dict | None" = None) -> bool:
    """
    True while `symbol` is excluded.

    The configured date is the day the symbol becomes eligible AGAIN, so the
    date itself is not excluded. Stating it as "eligible from" lets the value
    be read off a calendar with no off-by-one argument.
    """
    today = today or dt.date.today()
    exclusions = SYMBOL_EXCLUSIONS if exclusions is None else exclusions
    present, until = _lookup(symbol, exclusions)
    if not present:
        return False
    if until is None:
        return True                     # named but unreadable -> stays out
    return today < until


def block_reason(signal: dict, today: "dt.date | None" = None,
                 exclusions: "dict | None" = None) -> "str | None":
    """
    Why this signal is refused, or None to let it through.

    Mirrors earnings_calendar.block_reason so the two gates read alike at the
    call site and in the log.
    """
    if not isinstance(signal, dict):
        return None
    if signal.get("signal") not in GATED_SIGNALS:
        return None

    symbol = signal.get("symbol")
    sym    = _norm(symbol)
    if not sym:
        return None                     # nothing to match; not this gate's job

    today = today or dt.date.today()
    exclusions = SYMBOL_EXCLUSIONS if exclusions is None else exclusions
    present, until = _lookup(symbol, exclusions)
    if not present:
        return None

    if until is None:
        return (f"{sym} is on the exclusion list with an unreadable date "
                f"({(exclusions or {}).get(symbol, (exclusions or {}).get(sym))!r}) "
                f"— refusing to open while the policy cannot be read")
    if today < until:
        days = (until - today).days
        return (f"{sym} is excluded until {until.isoformat()} ({days}d) "
                f"— no new short puts in this symbol by instruction")
    return None


def active_exclusions(today: "dt.date | None" = None,
                      exclusions: "dict | None" = None) -> dict:
    """
    {symbol: eligible_from_date_or_None} for the exclusions blocking today.

    An unreadable date maps to None and is still reported, because it is
    blocking: a typo must be visible rather than quietly shrinking the
    watchlist.
    """
    today = today or dt.date.today()
    exclusions = SYMBOL_EXCLUSIONS if exclusions is None else exclusions
    out = {}
    for key, when in (exclusions or {}).items():
        sym = _norm(key)
        if not sym:
            continue
        until = _parse(when)
        if until is None or today < until:
            out[sym] = until
    return out


def summary(today: "dt.date | None" = None,
            exclusions: "dict | None" = None) -> str:
    """One line for the startup banner, or '' when nothing is excluded."""
    act = active_exclusions(today, exclusions)
    if not act:
        return ""
    parts = [f"{s} until {d.isoformat()}" if d else f"{s} (UNREADABLE DATE)"
             for s, d in sorted(act.items())]
    return "excluded: " + ", ".join(parts)
