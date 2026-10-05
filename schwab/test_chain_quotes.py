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
        assert q["order_limit"] == pytest.approx(round(1.20 * cq.ORDER_LIMIT_PCT, 2))

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


def test_order_limit_is_premium_times_the_haircut():
    q = cq.fetch_chain_quote(_client(_one_strike_chain(2.01, 2.40)),
                             "XOM", "PUT", target_strike=152.5, target_dte=32)
    assert q["order_limit"] == pytest.approx(round(2.205 * cq.ORDER_LIMIT_PCT, 2))
    assert q["order_limit"] == pytest.approx(2.09)


def test_the_limit_is_passive_when_the_spread_is_wide():
    """0.95*mid sits ABOVE the bid once the spread exceeds 10% of mid, so the
    order rests instead of crossing. Verified live 2026-10-06: AAPL/IBM at 11%
    and XOM at 16% rest; GOOGL 7%, AMZN 4%, NVDA 3% cross."""
    q = cq.fetch_chain_quote(_client(_one_strike_chain(9.60, 10.70)),  # 11%
                             "IBM", "PUT", target_strike=152.5, target_dte=32)
    assert q["order_limit"] > q["bid"], \
        f"limit {q['order_limit']} should rest above the bid {q['bid']}"


def test_the_limit_crosses_when_the_spread_is_very_tight():
    """Below 10% the haircut lands at or under the bid, so it fills. Fine — a
    tight spread means little is being given away."""
    q = cq.fetch_chain_quote(_client(_one_strike_chain(3.70, 3.80)),   # 3%
                             "NVDA", "PUT", target_strike=152.5, target_dte=32)
    assert q["order_limit"] <= q["bid"]


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
    assert out["order_limit"] == pytest.approx(2.09)   # shown as LIMIT


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
