"""
Tests for chain_quotes.py — real Schwab option-chain quotes for the Scavenger.
"""
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chain_quotes as cq


# ---------------------------------------------------------------------------
# Fake Schwab chain response
# ---------------------------------------------------------------------------

def _chain_response(exp_map_key="putExpDateMap"):
    """Minimal Schwab get_option_chain JSON with two expirations."""
    def opt(bid, ask, delta, iv, oi=500):
        return [{
            "bid": bid, "ask": ask, "delta": delta,
            "volatility": iv,           # Schwab returns IV as a percentage
            "openInterest": oi, "inTheMoney": False,
        }]
    return {
        "underlying": {"last": 83.29},
        exp_map_key: {
            "2026-07-24:21": {
                "78.0": opt(0.55, 0.65, -0.22, 24.1),
                "79.0": opt(0.70, 0.80, -0.28, 24.6),
            },
            "2026-08-07:35": {
                "78.0": opt(0.95, 1.05, -0.25, 25.0),
                "79.0": opt(1.10, 1.30, -0.30, 25.5),
                "80.0": opt(1.40, 1.60, -0.35, 26.0, oi=0),   # no OI — skipped
            },
        },
    }


def _mock_client(payload):
    client = MagicMock()
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    client.get_option_chain.return_value = resp
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFetchChainQuote:
    def test_picks_strike_nearest_target(self):
        client = _mock_client(_chain_response())
        q = cq.fetch_chain_quote(client, "KO", "PUT", target_strike=79.13,
                                 target_dte=30)
        assert q is not None
        assert q["strike"] == 79.0

    def test_picks_expiry_nearest_target_dte(self):
        client = _mock_client(_chain_response())
        q = cq.fetch_chain_quote(client, "KO", "PUT", target_strike=79.13,
                                 target_dte=30)
        # 35 DTE is closer to 30 than 21 DTE
        assert q["dte"] == 35
        assert q["expiry"] == "2026-08-07"

    def test_premium_is_mid_price(self):
        """premium is the MID (the fair fill estimate); order_limit is where the
        order actually goes in. Briefly the bid on 2026-10-06 before the
        patience-pricing decision superseded it."""
        client = _mock_client(_chain_response())
        q = cq.fetch_chain_quote(client, "KO", "PUT", target_strike=79.13,
                                 target_dte=30)
        assert q["premium"] == pytest.approx((1.10 + 1.30) / 2)
        # mid 1.20 is already on a penny tick, so the limit equals it
        assert q["order_limit"] == pytest.approx(1.20)
        assert q["order_limit"] > q["bid"]

    def test_carries_real_greeks(self):
        client = _mock_client(_chain_response())
        q = cq.fetch_chain_quote(client, "KO", "PUT", target_strike=79.13,
                                 target_dte=30)
        assert q["delta"] == pytest.approx(-0.30)
        assert q["iv"] == pytest.approx(0.255)     # percent → fraction

    def test_zero_open_interest_skipped(self):
        client = _mock_client(_chain_response())
        q = cq.fetch_chain_quote(client, "KO", "PUT", target_strike=80.0,
                                 target_dte=35)
        # 80 strike has oi=0 → nearest valid is 79
        assert q["strike"] == 79.0

    def test_calls_use_call_map(self):
        client = _mock_client(_chain_response(exp_map_key="callExpDateMap"))
        q = cq.fetch_chain_quote(client, "KO", "CALL", target_strike=79.0,
                                 target_dte=30)
        assert q is not None
        assert q["strike"] == 79.0

    def test_api_failure_returns_none(self):
        client = MagicMock()
        client.get_option_chain.side_effect = RuntimeError("api down")
        assert cq.fetch_chain_quote(client, "KO", "PUT", 79.0, 30) is None

    def test_empty_chain_returns_none(self):
        client = _mock_client({"underlying": {"last": 83.29},
                               "putExpDateMap": {}})
        assert cq.fetch_chain_quote(client, "KO", "PUT", 79.0, 30) is None


class TestRequoteSignal:
    def _signal(self):
        return {"symbol": "KO", "signal": "SELL_PUT", "close": 83.29,
                "strike": 79.13, "premium": 0.71, "premium_pct": 0.85,
                "dte": 30, "hv": 24.6, "adx": 15.0, "reason": "test"}

    def test_requote_updates_premium_and_strike(self):
        client = _mock_client(_chain_response())
        s = cq.requote_signal(client, self._signal())
        assert s is not None
        assert s["strike"] == 79.0
        assert s["premium"] == pytest.approx(1.20)   # the mid
        assert s["dte"] == 35
        assert s["quote_source"] == "schwab_chain"

    def test_requote_falls_back_to_model_on_failure(self):
        client = MagicMock()
        client.get_option_chain.side_effect = RuntimeError("api down")
        s = cq.requote_signal(client, self._signal())
        assert s is not None
        assert s["premium"] == 0.71            # unchanged
        assert s["quote_source"] == "model"

    def test_requote_drops_signal_when_real_premium_too_thin(self):
        payload = _chain_response()
        # Crush the quotes so premium/close falls below the 0.5% floor.
        # Keep the spread TIGHT (0.05/0.06 = 18%): a wide one would be dropped
        # by MAX_SPREAD_PCT first and never exercise the yield floor at all.
        for strikes in payload["putExpDateMap"].values():
            for opts in strikes.values():
                opts[0]["bid"], opts[0]["ask"] = 0.05, 0.06
        client = _mock_client(payload)
        assert cq.requote_signal(client, self._signal()) is None

    def test_non_option_signals_pass_through(self):
        client = _mock_client(_chain_response())
        s = {"symbol": "NVDA", "signal": "BUY", "entry": 100.0}
        assert cq.requote_signal(client, s) is s
        client.get_option_chain.assert_not_called()


# ===========================================================================
# Executable pricing + spread guard (added 2026-10-06)
# ===========================================================================
#
# The XOM loss of 2026-10-05. The scanner priced signals off the MID while
# real_overseer places orders at the BID (real_overseer.py:900). On a tight
# spread that is a rounding difference; on this one it halved the premium:
#
#   QUOTE:   bid/ask $0.79/$2.76          <- 250% spread
#   PREMIUM: $1.775/sh ($178) +1.10% yield  <- exactly the mid
#   FILLED:  $0.79                          <- $79, 0.52% yield
#
# The LLM approved on "ample yield", true of the mid and not of the fill, and
# SCAV_MIN_PREMIUM_PCT was tested against the mid too. The position tied up
# $15,250 at 0.52% and blocked a $220-premium XOM signal 3.5h later, which had
# a normal $2.01/$2.40 spread. Closed for -$82.66.
#
# Two fixes, both pinned here:
#   1. a spread guard, so an unquotable strike is skipped rather than sold into.
#   2. (premium was briefly switched to the bid here; superseded the same day by
#      the patience-pricing decision below — premium is the mid, the ORDER is
#      placed at 0.95*mid.)


def _one_strike_chain(bid, ask, oi=500, strike="152.5", dte=32):
    return {
        "underlying": {"last": 161.65},
        "putExpDateMap": {
            f"2026-11-06:{dte}": {
                strike: [{"bid": bid, "ask": ask, "delta": -0.22,
                          "volatility": 28.8, "openInterest": oi,
                          "inTheMoney": False}]
            }
        },
    }


def _client(payload):
    c = MagicMock()
    r = MagicMock()
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    c.get_option_chain.return_value = r
    return c


# --- 1. premium must be the executable price -------------------------------

def test_mid_is_still_reported_for_context():
    q = cq.fetch_chain_quote(_client(_one_strike_chain(2.01, 2.40)),
                             "XOM", "PUT", target_strike=152.5, target_dte=32)
    assert q["mid"] == pytest.approx(2.205)
    assert q["bid"] == pytest.approx(2.01) and q["ask"] == pytest.approx(2.40)


# --- 2. the spread guard ---------------------------------------------------

def test_the_exact_xom_spread_is_rejected():
    """bid 0.79 / ask 2.76 — a 250% spread. Never sell into this."""
    q = cq.fetch_chain_quote(_client(_one_strike_chain(0.79, 2.76)),
                             "XOM", "PUT", target_strike=152.5, target_dte=32)
    assert q is None, f"a 250% spread must be skipped, got {q}"


def test_a_normal_spread_is_accepted():
    """The later XOM signal: 2.01/2.40 is ~19%, tradeable."""
    q = cq.fetch_chain_quote(_client(_one_strike_chain(2.01, 2.40)),
                             "XOM", "PUT", target_strike=152.5, target_dte=32)
    assert q is not None


def test_a_tight_spread_is_accepted():
    """What the good fills looked like: AMZN/GOOGL were 3-6%."""
    q = cq.fetch_chain_quote(_client(_one_strike_chain(3.90, 4.15)),
                             "AMZN", "PUT", target_strike=152.5, target_dte=32)
    assert q is not None


def test_the_guard_prefers_a_liquid_neighbour_over_nothing():
    """If the target strike is unquotable but a nearby one is fine, take the
    neighbour rather than dropping the symbol."""
    payload = {
        "underlying": {"last": 161.65},
        "putExpDateMap": {
            "2026-11-06:32": {
                "152.5": [{"bid": 0.79, "ask": 2.76, "delta": -0.22,
                           "volatility": 28.8, "openInterest": 75,
                           "inTheMoney": False}],          # unquotable
                "155.0": [{"bid": 2.01, "ask": 2.40, "delta": -0.25,
                           "volatility": 29.2, "openInterest": 500,
                           "inTheMoney": False}],          # fine
            }
        },
    }
    q = cq.fetch_chain_quote(_client(payload), "XOM", "PUT",
                             target_strike=152.5, target_dte=32)
    assert q is not None and q["strike"] == pytest.approx(155.0)
    assert q["premium"] == pytest.approx(2.205)      # mid of 2.01/2.40


# --- 3. the yield floor now sees the truth ---------------------------------

# ===========================================================================
# Patience pricing (2026-10-06, user decision)
# ===========================================================================
#
# Supersedes the bid-pricing change made earlier the same day. The evidence
# that drove it: across 12 STO fills placed AT the bid, the fills averaged
# +$12.50/contract ABOVE the bid (a sell limit fills at the limit or better) —
# so the bid is the guaranteed floor, not the expected fill, and showing it as
# PREMIUM understated every signal.
#
# The user's call: show PREMIUM as the MID, place the order at
# round(mid * 0.95, 2), and let the LLM see both. If it does not fill today a
# fresh signal arrives tomorrow — "we'd rather be safe than in a bad-shaped
# position." That also unifies pricing with the model-fallback path, which
# already used premium * 0.95.
#
# The spread guard stays, and is what makes the mid meaningful at all: the mid
# of a 250% stale quote is the number that cost -$82.66 on XOM.


def test_premium_is_the_mid():
    q = cq.fetch_chain_quote(_client(_one_strike_chain(2.01, 2.40)),
                             "XOM", "PUT", target_strike=152.5, target_dte=32)
    assert q["premium"] == pytest.approx(2.205)
    assert q["bid"] == pytest.approx(2.01) and q["ask"] == pytest.approx(2.40)


def test_order_limit_is_the_mid_on_the_next_tick():
    q = cq.fetch_chain_quote(_client(_one_strike_chain(2.01, 2.40)),
                             "XOM", "PUT", target_strike=152.5, target_dte=32)
    # mid 2.205, penny ticks below $3 -> 2.21
    assert q["order_limit"] == pytest.approx(2.21)
    assert q["order_limit"] > q["bid"]


def test_the_limit_is_passive_when_the_spread_is_wide():
    """A wide market must leave the order resting, never crossing."""
    q = cq.fetch_chain_quote(_client(_one_strike_chain(9.60, 10.70)),  # 11%
                             "IBM", "PUT", target_strike=152.5, target_dte=32)
    assert q["order_limit"] > q["bid"], \
        f"limit {q['order_limit']} should rest above the bid {q['bid']}"


def test_a_tight_spread_also_rests_above_the_bid():
    """This test previously asserted the opposite, and the reasoning it carried
    -- "fine, a tight spread means little is being given away" -- was wrong.
    Tight spreads are where 0.95*mid went furthest below the bid in dollar
    terms, because the tightest markets are the expensive ones: AMD $610 at a
    2.1% spread priced $1.26 under its 31.35 bid."""
    q = cq.fetch_chain_quote(_client(_one_strike_chain(3.70, 3.80)),   # 3%
                             "NVDA", "PUT", target_strike=152.5, target_dte=32)
    assert q["order_limit"] > q["bid"]
    assert q["order_limit"] == pytest.approx(3.75)


def test_the_spread_guard_still_rejects_the_xom_quote():
    """Pricing off the mid is only safe because this guard exists."""
    q = cq.fetch_chain_quote(_client(_one_strike_chain(0.79, 2.76)),
                             "XOM", "PUT", target_strike=152.5, target_dte=32)
    assert q is None


def test_requote_exposes_both_numbers_to_the_signal():
    s = {"signal": "SELL_PUT", "symbol": "XOM", "strike": 152.5,
         "close": 161.65, "dte": 32}
    out = cq.requote_signal(_client(_one_strike_chain(2.01, 2.40)), s)
    assert out is not None
    assert out["premium"] == pytest.approx(2.205)      # shown as PREMIUM
    assert out["order_limit"] == pytest.approx(2.21)   # shown as LIMIT


# ===========================================================================
# Liquidity context + a bid-size gate (2026-10-06)
# ===========================================================================
#
# Measured across the watchlist, the fields we were NOT using turned out to be
# the ones that say whether a quote is real:
#
#   sym    last lastSize  vol  bidSz askSz  trade age
#   AMZN   7.35     1     234   635   156      0.2m
#   IBM   10.00     1      11   257    46     68.4m
#   KO     0.07     4       0     0    122   5863.0m   <- 4 DAYS, bid 0.00
#
# lastSize is 1-4 contracts everywhere, and trade age ranges from 12 seconds to
# four days -- which is why `last` is context, never a price reference.
#
# KO is the latent bug: no bid, no bid size, no volume, yet openInterest clears
# MIN_OPEN_INT = 1. Only the `bid <= 0` check stopped it; a contract quoting
# bid 0.05 with bidSize 0 would have passed. You cannot sell to a buyer who
# isn't there, so bid SIZE is the real liquidity gate -- open interest counts
# contracts someone holds, not contracts anyone will buy today.


def _rich(bid=2.01, ask=2.40, bid_size=500, ask_size=400, last=2.20,
          last_size=1, volume=150, trade_age_s=300):
    quote_t = 1791223543231
    return {
        "underlying": {"last": 161.65},
        "putExpDateMap": {
            "2026-11-06:32": {
                "152.5": [{
                    "bid": bid, "ask": ask, "bidSize": bid_size,
                    "askSize": ask_size, "last": last, "lastSize": last_size,
                    "totalVolume": volume, "delta": -0.22, "volatility": 28.8,
                    "openInterest": 500, "inTheMoney": False,
                    "quoteTimeInLong": quote_t,
                    "tradeTimeInLong": quote_t - trade_age_s * 1000,
                }]
            }
        },
    }


# --- 3. the bid-size gate --------------------------------------------------

def test_a_contract_with_no_bid_size_is_rejected():
    """The KO case: you cannot sell into a bid that has no size behind it."""
    q = cq.fetch_chain_quote(_client(_rich(bid=0.05, ask=0.10, bid_size=0)),
                             "KO", "PUT", target_strike=152.5, target_dte=32)
    assert q is None


def test_a_thin_bid_size_is_rejected():
    q = cq.fetch_chain_quote(_client(_rich(bid_size=1)),
                             "XOM", "PUT", target_strike=152.5, target_dte=32)
    assert q is None, f"bidSize 1 should not qualify (floor {cq.MIN_BID_SIZE})"


def test_adequate_bid_size_is_accepted():
    q = cq.fetch_chain_quote(_client(_rich(bid_size=cq.MIN_BID_SIZE)),
                             "XOM", "PUT", target_strike=152.5, target_dte=32)
    assert q is not None


def test_a_missing_bid_size_does_not_block_the_signal():
    """Some feeds omit sizes. Absent data must not be read as zero size, or
    every signal disappears on a partial quote."""
    payload = _rich()
    del payload["putExpDateMap"]["2026-11-06:32"]["152.5"][0]["bidSize"]
    q = cq.fetch_chain_quote(_client(payload), "XOM", "PUT",
                             target_strike=152.5, target_dte=32)
    assert q is not None


# --- 2. liquidity context for the LLM -------------------------------------

def test_liquidity_fields_are_carried_through():
    q = cq.fetch_chain_quote(_client(_rich(last=2.20, last_size=3, volume=150,
                                           bid_size=500, ask_size=400)),
                             "XOM", "PUT", target_strike=152.5, target_dte=32)
    assert q["last"] == pytest.approx(2.20)
    assert q["last_size"] == 3
    assert q["volume"] == 150
    assert q["bid_size"] == 500 and q["ask_size"] == 400


def test_trade_age_is_computed_in_minutes():
    q = cq.fetch_chain_quote(_client(_rich(trade_age_s=4104)),   # 68.4 min
                             "IBM", "PUT", target_strike=152.5, target_dte=32)
    assert q["trade_age_min"] == pytest.approx(68.4, abs=0.1)


def test_a_four_day_old_trade_is_reported_not_hidden():
    """KO's last traded 5863 minutes ago. The number must reach the LLM so it
    can distrust the quote -- it is not a reason to drop the signal on its own,
    because a stale print with a live two-sided quote is still tradeable."""
    q = cq.fetch_chain_quote(_client(_rich(trade_age_s=5863*60)),
                             "KO", "PUT", target_strike=152.5, target_dte=32)
    assert q is not None
    assert q["trade_age_min"] > 5000


def test_trade_age_is_none_when_timestamps_are_missing():
    payload = _rich()
    o = payload["putExpDateMap"]["2026-11-06:32"]["152.5"][0]
    del o["tradeTimeInLong"]
    q = cq.fetch_chain_quote(_client(payload), "XOM", "PUT",
                             target_strike=152.5, target_dte=32)
    assert q is not None and q["trade_age_min"] is None


def test_requote_passes_liquidity_through_to_the_signal():
    s = {"signal": "SELL_PUT", "symbol": "XOM", "strike": 152.5,
         "close": 161.65, "dte": 32}
    out = cq.requote_signal(_client(_rich()), s)
    assert out is not None
    for k in ("last", "last_size", "volume", "bid_size", "ask_size", "trade_age_min"):
        assert k in out, f"{k} should reach the signal the LLM reads"


# --- 4. the untraded-market gate -------------------------------------------
#
# Measured on the live chain 2026-10-06: of 322 contracts that survived every
# other gate inside the delta band we sell (-0.10..-0.40), 7 had traded zero
# contracts today AND had a last print ~3 days old -- they sat untested through
# a full session and then some. Two of them:
#
#   META $705  spread  5.7%  oi  12  bidSz 239  vol 0  age 3.1d  last 27.56 / mid 22.02
#   PG   $139  spread  8.5%  oi 100  bidSz  51  vol 0  age 3.0d  last  1.73 / mid  1.29
#
# Both look healthy on every axis we gated: tight spread, deep bid, real open
# interest. The only tell is that nobody has traded them, and `last` sits 25-34%
# away from the mid -- so the mid is a market maker's opinion, not a price. This
# is the one case that is objectively broken rather than merely thin, which is
# why it earns a hard reject instead of a warning: a stale, untraded contract
# gives the LLM no way to sanity-check the quote it is pricing off.
#
# Note the original proposal here was `openInterest == 0 and volume == 0`. That
# is dead code: MIN_OPEN_INT = 1 already rejects all 823 oi==0 contracts, and
# the 740 matching oi==0-and-vol==0 are a strict subset. It would reject nothing.


def test_untraded_and_stale_contract_is_rejected():
    """Zero volume today plus a days-old print: nobody has tested this quote."""
    q = cq.fetch_chain_quote(
        _client(_rich(volume=0, trade_age_s=3 * 86400)),
        "PG", "PUT", target_strike=152.5, target_dte=32)
    assert q is None


def test_untraded_but_freshly_printed_contract_is_kept():
    """vol==0 with a recent print is thin, not broken — that is the LLM's call.

    Volume resets each session, so a contract that printed 30 minutes ago can
    legitimately show zero volume across a session boundary.
    """
    q = cq.fetch_chain_quote(
        _client(_rich(volume=0, trade_age_s=1800)),
        "PG", "PUT", target_strike=152.5, target_dte=32)
    assert q is not None
    assert q["volume"] == 0


def test_stale_contract_that_actually_traded_is_kept():
    """A days-old quote that has traded today is tested, just quiet."""
    q = cq.fetch_chain_quote(
        _client(_rich(volume=5, trade_age_s=3 * 86400)),
        "IBM", "PUT", target_strike=152.5, target_dte=32)
    assert q is not None


def test_missing_trade_timestamps_do_not_reject():
    """Absent data is not evidence of a dead market.

    Same principle as bidSize: treating a missing field as the worst case would
    silently drop every signal from a quote that happens to omit it.
    """
    chain = _rich(volume=0)
    opt = chain["putExpDateMap"]["2026-11-06:32"]["152.5"][0]
    del opt["tradeTimeInLong"]
    q = cq.fetch_chain_quote(_client(chain), "PG", "PUT",
                             target_strike=152.5, target_dte=32)
    assert q is not None
    assert q["trade_age_min"] is None


def test_missing_volume_field_does_not_reject():
    """An absent totalVolume is a data gap, not a zero."""
    chain = _rich(trade_age_s=3 * 86400)
    opt = chain["putExpDateMap"]["2026-11-06:32"]["152.5"][0]
    del opt["totalVolume"]
    q = cq.fetch_chain_quote(_client(chain), "PG", "PUT",
                             target_strike=152.5, target_dte=32)
    assert q is not None


def test_untraded_age_boundary():
    """Exactly at the limit is kept; one minute past it is rejected."""
    at = cq.fetch_chain_quote(
        _client(_rich(volume=0, trade_age_s=cq.MAX_UNTRADED_AGE_MIN * 60)),
        "PG", "PUT", target_strike=152.5, target_dte=32)
    assert at is not None, "the boundary itself must not reject"
    past = cq.fetch_chain_quote(
        _client(_rich(volume=0, trade_age_s=(cq.MAX_UNTRADED_AGE_MIN + 1) * 60)),
        "PG", "PUT", target_strike=152.5, target_dte=32)
    assert past is None


def test_healthy_contract_still_selected_after_the_gate():
    """Regression: the common case must be untouched."""
    q = cq.fetch_chain_quote(_client(_rich()), "IBM", "PUT",
                             target_strike=152.5, target_dte=32)
    assert q is not None
    assert q["volume"] == 150


def test_weekend_gap_alone_trips_the_gate_when_nothing_trades():
    """Documents the Monday asymmetry so it is a decision, not a surprise.

    A Friday-close print is ~3930 minutes old by Monday's open, well past
    MAX_UNTRADED_AGE_MIN. So on Mondays this gate reduces to "zero volume
    today". Measured cost: 5 of 314 in-zone candidates (1.6%), all with open
    interest under 50. Accepted deliberately — skipping a contract nobody has
    touched since Friday costs one day, and the next scan re-evaluates it.
    """
    fri_close_to_mon_open = 3930 * 60
    q = cq.fetch_chain_quote(
        _client(_rich(volume=0, trade_age_s=fri_close_to_mon_open)),
        "META", "PUT", target_strike=152.5, target_dte=32)
    assert q is None
    # ...but the moment it trades, it is eligible again, same stale print.
    traded = cq.fetch_chain_quote(
        _client(_rich(volume=1, trade_age_s=fri_close_to_mon_open)),
        "META", "PUT", target_strike=152.5, target_dte=32)
    assert traded is not None


# --- 5. the order limit: mid, rounded to a tradable tick --------------------
#
# The bug this replaces: limit = round(0.95 * mid, 2) lands BELOW the bid
# whenever the spread is under 10.53% of mid, because 0.95*mid > bid requires
# ask/bid > 1.05/0.95. A sell limit below the bid crosses the book and fills
# immediately AT the bid, so the "patient" pricing silently became bid-selling
# on exactly the liquid names we trade most. Measured live 2026-10-06 across 309
# sellable candidates: 240 of them (77.7%) priced below the bid, median $0.16/sh
# under it, worst $1.26 (AMD $610, spread 2.1%, bid 31.35 -> limit 30.09).
#
# The limit is now the mid rounded UP to the next valid tick. Rounding up rather
# than down is load-bearing: on a one-tick-wide market (bid 3.55 / ask 3.60) the
# mid is 3.575, and rounding DOWN gives 3.55 -- the bid again, crossing. Rounding
# up cannot exceed the ask, because the ask is itself on a valid tick and
# mid < ask, so the invariant bid < limit <= ask holds for every quote.
#
# Increments match what the market displays, confirmed against 3,837 live
# quotes: $0.01 below $3.00 (76% of those were non-nickel), $0.05 at or above
# (0 of 2,636 were non-nickel). This is NOT rejection protection -- the two
# off-nickel limits we have sent above $3 (NVDA $4.84, IBM $3.06, 2026-08-21)
# both filled at exactly those prices. It is a choice that costs ~$0.90 per
# contract against penny rounding, and keeps the limit on a displayed price.


def test_tick_is_a_penny_below_three_and_a_nickel_above():
    assert cq._tick(0.55) == 0.01
    assert cq._tick(2.99) == 0.01
    assert cq._tick(3.00) == 0.05
    assert cq._tick(31.68) == 0.05


def test_order_limit_rounds_the_mid_up_to_a_valid_tick():
    # AAPL live: mid 3.675 is not a nickel, so it must move to 3.70, not 3.65.
    assert cq.order_limit_for(3.55, 3.80) == 3.70
    # AMD live: mid 31.675 -> 31.70
    assert cq.order_limit_for(31.35, 32.00) == 31.70
    # already on a tick: left alone
    assert cq.order_limit_for(1.00, 1.50) == 1.25
    assert cq.order_limit_for(3.50, 3.70) == 3.60


def test_order_limit_never_crosses_the_bid_on_a_one_tick_market():
    """The case that rounding DOWN would break."""
    assert cq.order_limit_for(3.55, 3.60) == 3.60   # mid 3.575 -> up, not 3.55
    assert cq.order_limit_for(0.55, 0.56) == 0.56   # penny-wide


def test_order_limit_invariant_holds_across_the_whole_quote_space():
    """bid < limit <= ask for every plausible quote, both tick regimes."""
    checked = 0
    for bid_c in range(1, 600):
        bid = bid_c / 100
        if abs(round(bid / cq._tick(bid)) * cq._tick(bid) - bid) > 1e-9:
            continue                      # not a price this option could quote
        for n in range(1, 12):            # ask from one to eleven ticks wider
            ask = round(bid + n * cq._tick(bid), 2)
            if abs(round(ask / cq._tick(ask)) * cq._tick(ask) - ask) > 1e-9:
                continue                  # straddles $3.00 onto an invalid tick
            lim = cq.order_limit_for(bid, ask)
            assert lim > bid - 1e-9, f"bid {bid} ask {ask} -> limit {lim} crosses"
            assert lim <= ask + 1e-9, f"bid {bid} ask {ask} -> limit {lim} over ask"
            assert abs(round(lim / cq._tick(lim)) * cq._tick(lim) - lim) < 1e-6, \
                f"limit {lim} is not on a tradable tick"
            checked += 1
    assert checked > 3000, f"only {checked} quotes exercised"


def test_order_limit_is_at_least_the_mid():
    """We never ask for less than fair value; rounding only ever helps us."""
    for bid, ask in [(3.55, 3.80), (0.55, 0.65), (31.35, 32.00), (1.66, 2.13)]:
        assert cq.order_limit_for(bid, ask) >= (bid + ask) / 2 - 1e-9


def test_locked_market_does_not_blow_up():
    """bid == ask: nothing to be patient about, and it must not crash."""
    assert cq.order_limit_for(2.00, 2.00) == 2.00


def test_fetch_chain_quote_uses_the_tick_limit():
    q = cq.fetch_chain_quote(_client(_rich(bid=3.55, ask=3.80)), "AAPL", "PUT",
                             target_strike=152.5, target_dte=32)
    assert q is not None
    assert q["premium"] == q["mid"] == 3.675      # mid stays the fair-value ref
    assert q["order_limit"] == 3.70               # the price we actually ask
    assert q["order_limit"] > q["bid"], "must rest above the bid, never cross"


# ===========================================================================
# Strike-drift guard + honest OTM label (added 2026-10-08)
#
# The liquidity filters (spread, bid size, untraded age) reject the thin
# strikes nearest the 5%-OTM target, and the ranking then takes the nearest
# ELIGIBLE strike with no bound on how far that is. On 2026-10-08 that turned
# a 5% OTM XOM put into a 2.1% OTM, delta -0.42 contract at a different
# expiry -- and the display still said "(5% OTM)" because
# live_scanner printed s.get("otm_pct", "5") and nothing ever set otm_pct.
#
# Threshold from measurement, not taste: across 14,331 SELL_PUT signals in
# data/overseer.log the realized OTM% sits at p0.1=3.88, p1=4.00, p50=4.97.
# The natural floor is ~3.86% (NVDA, nearest listed strike on a high-priced
# stock). Only 2 signals ever fell below 3.5% -- the two pathological XOM
# re-quotes. So 0.70 x target (3.5% of a 5% target) separates the defect from
# every legitimate signal we have on record.
# ===========================================================================

def _two_strike_chain(dte=43):
    """A chain where only a far-from-target strike survives the filters."""
    def opt(bid, ask, delta, oi=1800, bsz=366, vol=26):
        return [{"bid": bid, "ask": ask, "delta": delta, "volatility": 27.0,
                 "openInterest": oi, "bidSize": bsz, "totalVolume": vol,
                 "inTheMoney": False}]
    return {
        "underlying": {"last": 168.60},
        "putExpDateMap": {
            f"2026-11-20:{dte}": {
                # the 5%-OTM strike: real quote but a thin bid size -> rejected
                "160.0": opt(3.35, 3.45, -0.236, bsz=5),
                # liquid, but far too close to the money
                "165.0": opt(5.00, 5.30, -0.420),
            },
        },
    }


def test_a_strike_that_drifted_toward_the_money_is_rejected():
    """The live XOM case: $165 on a $168.60 stock is 2.1% OTM, not 5%."""
    s = {"symbol": "XOM", "signal": "SELL_PUT", "close": 168.60,
         "strike": 160.17, "premium": 1.76, "dte": 30, "reason": "test"}
    assert cq.requote_signal(_client(_two_strike_chain()), s) is None, (
        "a 2.1% OTM substitute for a 5% OTM target must be skipped, not sold")


def test_the_real_five_percent_strike_is_accepted():
    """Same chain with the bid size healthy: $160 is eligible and wins."""
    payload = _two_strike_chain()
    payload["putExpDateMap"]["2026-11-20:43"]["160.0"][0]["bidSize"] = 366
    s = {"symbol": "XOM", "signal": "SELL_PUT", "close": 168.60,
         "strike": 160.17, "premium": 1.76, "dte": 30, "reason": "test"}
    out = cq.requote_signal(_client(payload), s)
    assert out is not None
    assert out["strike"] == 160.0
    assert out["otm_pct"] == pytest.approx(5.1, abs=0.1)


def test_the_natural_low_tail_still_passes():
    """NVDA's nearest listed strike lands at 3.86% OTM -- legitimate, keep it."""
    payload = {
        "underlying": {"last": 207.00},
        "putExpDateMap": {"2026-11-06:29": {
            "199.0": [{"bid": 4.20, "ask": 4.40, "delta": -0.28,
                       "volatility": 38.0, "openInterest": 900,
                       "bidSize": 120, "totalVolume": 400,
                       "inTheMoney": False}]}},
    }
    s = {"symbol": "NVDA", "signal": "SELL_PUT", "close": 207.00,
         "strike": 196.65, "premium": 4.30, "dte": 30, "reason": "test"}
    out = cq.requote_signal(_client(payload), s)
    assert out is not None, "3.86% OTM is the natural floor, not a defect"
    assert out["otm_pct"] == pytest.approx(3.9, abs=0.1)


def test_otm_pct_is_set_so_the_display_stops_guessing():
    """live_scanner prints s.get('otm_pct', '5') -- the key must exist."""
    out = cq.requote_signal(_client(_rich(bid=3.55, ask=3.80)), {
        "symbol": "AAPL", "signal": "SELL_PUT", "close": 160.53,
        "strike": 152.50, "premium": 3.60, "dte": 30, "reason": "test"})
    assert out is not None
    assert "otm_pct" in out, "otm_pct missing -> the label silently says 5%"
    assert out["otm_pct"] == pytest.approx(5.0, abs=0.1)


def test_otm_pct_is_measured_the_other_way_for_calls():
    payload = {
        "underlying": {"last": 100.00},
        "callExpDateMap": {"2026-11-06:29": {
            "108.0": [{"bid": 1.20, "ask": 1.30, "delta": 0.25,
                       "volatility": 30.0, "openInterest": 500,
                       "bidSize": 90, "totalVolume": 50,
                       "inTheMoney": False}]}},
    }
    out = cq.requote_signal(_client(payload), {
        "symbol": "KO", "signal": "SELL_CALL", "close": 100.00,
        "strike": 108.00, "premium": 1.25, "dte": 30, "reason": "test"})
    assert out is not None
    assert out["otm_pct"] == pytest.approx(8.0, abs=0.1)
