"""
A dated exclusion list: "stay out of this symbol until this date".

Asked for on 2026-10-28, after closing an AMZN put for +$143 one day before
the print. The question "make sure it stays away from AMZN" had no honest
answer, because nothing in the tree could express it:

  - The earnings gate blocks AMZN today, but only until the print lands.
    Worse, it then keeps blocking for an accidental reason — once 2026-10-29
    passes, the upcoming-cache still reads `fetched_at 2026-10-27` (fresh)
    while holding a date in the past, so both cache rungs return nothing, the
    projection takes over, the margin pulls it behind `on_date`, and the
    clamp pins it to "tomorrow" every day until the scrape refreshes. That
    fails closed, which is correct, but it is not a policy — it expires on
    its own and nobody chose its duration.

  - The 20% concentration cap does not bind the first contract in a name.
    `size_put` gives an unheld symbol a floor of one contract on purpose
    (without it, 20% of $45,838 is $9,167, under a single $21,500 contract,
    and the cap would block 12 of 14 symbols and stall the book). Measured
    on the live account the morning of 2026-10-28, AMZN unheld:

        $240 strike -> size_put = 1 = $24,000 collateral = 52% of total cash
        $250 strike -> size_put = 0   ...only because free cash was $24,885

    So the sole thing standing between $24,885 of freed collateral and 52%
    of the book in one name was being $115 short of the next strike. That is
    a coincidence, not a control, and it is the same shape as the 99%-in-one-
    name incident of 2026-09-18.

WHAT THIS GATES, AND WHAT IT MUST NOT
Only SELL_PUT — the one signal that opens new exposure, and the same scope
the earnings gate chose. Everything else passes untouched, and one case is a
safety requirement rather than a preference: BUY_TO_CLOSE must never be
blocked. AutoClose iterates the ledger rather than WATCHLIST, so a resting
protective cover does not depend on this list; but if an exclusion ever did
reach the cover path, an excluded symbol holding a short put would be unable
to buy it back, and the list would convert "do not open this" into "cannot
close this". `test_a_cover_is_never_blocked` is that requirement.

FAIL CLOSED ON A BAD DATE
A symbol named in the list with an unreadable date stays excluded. The
alternative is that a typo silently re-admits the symbol the user asked to
stay out of, which is the fail-open mistake 590e756 was written to delete.
An exclusion is a refusal to trade: the cost of being wrong is a trade not
taken, against a position nobody authorised.
"""
import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import symbol_exclusions as se
import strategy_params


TODAY = dt.date(2026, 10, 28)
LIST  = {"AMZN": "2027-01-01"}


def _put(symbol="AMZN", strike=240.0):
    return {"symbol": symbol, "signal": "SELL_PUT", "strike": strike,
            "expiry": "2026-11-27"}


class TestTheWindow:
    """Excluded up to the date, eligible from it."""

    def test_an_excluded_symbol_is_refused_before_its_date(self):
        assert se.is_excluded("AMZN", TODAY, LIST) is True
        why = se.block_reason(_put(), TODAY, LIST)
        assert why is not None
        assert "AMZN" in why and "2027-01-01" in why

    def test_the_day_before_the_date_is_still_excluded(self):
        assert se.is_excluded("AMZN", dt.date(2026, 12, 31), LIST) is True

    def test_the_date_itself_re_admits_the_symbol(self):
        """
        The value is the day the symbol becomes eligible again, not the last
        excluded day. Stating it as "eligible from" means the configured date
        can be read off a calendar without an off-by-one argument.
        """
        assert se.is_excluded("AMZN", dt.date(2027, 1, 1), LIST) is False
        assert se.block_reason(_put(), dt.date(2027, 1, 1), LIST) is None

    def test_after_the_date_the_symbol_is_eligible(self):
        assert se.is_excluded("AMZN", dt.date(2027, 6, 1), LIST) is False

    def test_a_symbol_not_on_the_list_is_never_refused(self):
        assert se.is_excluded("NVDA", TODAY, LIST) is False
        assert se.block_reason(_put("NVDA", 215.0), TODAY, LIST) is None

    def test_an_empty_list_refuses_nothing(self):
        assert se.is_excluded("AMZN", TODAY, {}) is False
        assert se.block_reason(_put(), TODAY, {}) is None


class TestScope:
    """Only SELL_PUT is gated. The cover case is a safety requirement."""

    def test_a_cover_is_never_blocked(self):
        """
        An exclusion says "do not open this", never "cannot close this". If
        this ever fails, an excluded symbol holding a short put cannot buy it
        back and the list has become a trap.
        """
        cover = {"symbol": "AMZN", "signal": "BUY_TO_CLOSE", "strike": 235.0}
        assert se.block_reason(cover, TODAY, LIST) is None

    @pytest.mark.parametrize("signal", [
        "BUY_TO_CLOSE", "SELL_CALL", "HOLD_SHARES", "RESUME_WHEEL",
        "PUT_ASSIGNED", "EXPIRED", None,
    ])
    def test_nothing_but_sell_put_is_gated(self, signal):
        assert se.block_reason(
            {"symbol": "AMZN", "signal": signal}, TODAY, LIST) is None

    def test_a_signal_with_no_symbol_is_not_gated(self):
        """Absent symbol is a malformed signal for some other gate to refuse;
        this one has nothing to match and must not guess."""
        assert se.block_reason({"signal": "SELL_PUT"}, TODAY, LIST) is None


class TestFailsClosed:
    """A named symbol with an unreadable date stays out."""

    @pytest.mark.parametrize("bad", [
        "", "   ", "not-a-date", "2027-13-01", "01/01/2027", "2027",
        None, 20270101, [], {},
    ])
    def test_an_unreadable_date_keeps_the_symbol_excluded(self, bad):
        lst = {"AMZN": bad}
        assert se.is_excluded("AMZN", TODAY, lst) is True, (
            f"{bad!r} must not re-admit the symbol")
        assert se.block_reason(_put(), TODAY, lst) is not None

    def test_the_reason_says_the_date_is_unreadable(self):
        why = se.block_reason(_put(), TODAY, {"AMZN": "not-a-date"})
        assert why is not None and "AMZN" in why
        assert "unreadable" in why.lower() or "unparseable" in why.lower()


class TestMatching:
    def test_the_match_is_case_insensitive(self):
        assert se.is_excluded("amzn", TODAY, LIST) is True
        assert se.is_excluded("AMZN", TODAY, {"amzn": "2027-01-01"}) is True

    def test_surrounding_whitespace_does_not_defeat_the_match(self):
        assert se.is_excluded(" AMZN ", TODAY, LIST) is True
        assert se.is_excluded("AMZN", TODAY, {" AMZN ": "2027-01-01"}) is True

    def test_a_partial_name_does_not_match(self):
        assert se.is_excluded("AMZ", TODAY, LIST) is False
        assert se.is_excluded("AMZNX", TODAY, LIST) is False


class TestReporting:
    def test_active_exclusions_lists_only_the_live_ones(self):
        lst = {"AMZN": "2027-01-01", "KO": "2026-10-01"}   # KO already lapsed
        act = se.active_exclusions(TODAY, lst)
        assert set(act) == {"AMZN"}
        assert act["AMZN"] == dt.date(2027, 1, 1)

    def test_an_unreadable_date_is_reported_as_active(self):
        """It is blocking, so it must be visible — otherwise a typo silently
        removes a symbol from the book with nothing saying why."""
        act = se.active_exclusions(TODAY, {"AMZN": "oops"})
        assert "AMZN" in act and act["AMZN"] is None


class TestTheLiveConfiguration:
    """Invariants over the dict that is actually shipped."""

    def test_the_configured_exclusions_all_parse(self):
        """
        A typo in SYMBOL_EXCLUSIONS fails closed, so it would silently drop a
        symbol from the book instead of crashing. This is what notices.
        """
        bad = {}
        for sym, when in (strategy_params.SYMBOL_EXCLUSIONS or {}).items():
            if se._parse(when) is None:
                bad[sym] = when
        assert not bad, f"unreadable exclusion dates in config: {bad}"

    def test_every_excluded_symbol_is_spelled_like_a_ticker(self):
        for sym in (strategy_params.SYMBOL_EXCLUSIONS or {}):
            assert sym == sym.strip().upper(), \
                f"{sym!r} should be upper-case and unpadded"
            assert sym.isalpha(), f"{sym!r} is not a plain ticker"

    def test_amzn_is_excluded_today(self):
        """
        The reason this file exists. If someone removes AMZN from the list,
        this fails and they have to mean it.
        """
        assert se.is_excluded("AMZN", TODAY) is True
        why = se.block_reason(_put(), TODAY)
        assert why is not None and "AMZN" in why

    def test_no_symbol_is_excluded_forever_by_accident(self):
        """
        Every exclusion carries an end date within ~5 years. A list with no
        horizon becomes a permanent, unreviewed shrinking of the watchlist.
        """
        horizon = TODAY + dt.timedelta(days=365 * 5)
        for sym, when in (strategy_params.SYMBOL_EXCLUSIONS or {}).items():
            parsed = se._parse(when)
            if parsed is not None:
                assert parsed <= horizon, \
                    f"{sym} is excluded until {parsed} — that is not a pause"


class TestTheWiring:
    """
    The module is only worth anything if the live paths call it. These assert
    through the production functions rather than by reading the source — the
    lesson of 2026-10-06, when hand-reimplemented logic passed while the real
    code was broken (see feedback_verify_via_production_code).
    """

    def test_the_scanner_imports_the_gate(self):
        import live_scanner
        assert live_scanner.symbol_exclusions is se

    def test_the_overseer_refuses_an_excluded_put_before_any_api_call(self):
        """
        `_pre_trade_check(client=None, ...)` would raise the moment it touched
        the client. A clean (False, reason) therefore proves the exclusion
        short-circuits ahead of fetch_account — so a refusal costs no API
        call, and moving the check later fails this test.
        """
        import real_overseer as ro
        ov = ro.RealOverseer.__new__(ro.RealOverseer)   # no __init__ side effects
        ok, msg = ro.RealOverseer._pre_trade_check(
            ov, None, "no-such-hash", _put())
        assert ok is False
        assert "AMZN" in msg and "excluded" in msg.lower()

    def test_the_overseer_still_reaches_the_account_for_an_allowed_symbol(self):
        """
        The mirror of the test above: a symbol that is NOT excluded must fall
        through to the real checks. With client=None that surfaces as a raised
        exception or a different refusal — what it must never be is the
        exclusion message, which would mean the gate is refusing everything.
        """
        import real_overseer as ro
        ov = ro.RealOverseer.__new__(ro.RealOverseer)
        try:
            ok, msg = ro.RealOverseer._pre_trade_check(
                ov, None, "no-such-hash", _put("NVDA", 215.0))
        except Exception:
            return                      # got past the gate into real work
        assert "excluded" not in msg.lower()

    def test_a_cover_on_an_excluded_symbol_passes_the_gate(self):
        """The trap case, asserted against the real gate function."""
        cover = {"symbol": "AMZN", "signal": "BUY_TO_CLOSE", "strike": 235.0}
        assert se.block_reason(cover) is None
