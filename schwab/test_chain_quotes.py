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

    def test_premium_is_the_bid(self):
        """Changed 2026-10-06: was asserting the MID, which is what caused the
        XOM loss — orders are placed at the bid (real_overseer.py:900), so the
        mid overstated every wide-spread signal's yield."""
        client = _mock_client(_chain_response())
        q = cq.fetch_chain_quote(client, "KO", "PUT", target_strike=79.13,
                                 target_dte=30)
        assert q["premium"] == pytest.approx(1.10)          # the bid
        assert q["mid"] == pytest.approx((1.10 + 1.30) / 2)  # still reported

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
        assert s["premium"] == pytest.approx(1.10)   # the bid, not the 1.20 mid
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
#   1. premium IS the bid — the price orders actually get.
#   2. a spread guard, so an unquotable strike is skipped rather than sold into.


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

def test_premium_is_the_bid_not_the_mid():
    """The order goes in at the bid, so that is what the yield must be based on."""
    q = cq.fetch_chain_quote(_client(_one_strike_chain(2.01, 2.40)),
                             "XOM", "PUT", target_strike=152.5, target_dte=32)
    assert q is not None
    assert q["premium"] == pytest.approx(2.01), \
        f"premium should be the bid 2.01, got {q['premium']} (mid would be 2.205)"


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
    assert q["premium"] == pytest.approx(2.01)


# --- 3. the yield floor now sees the truth ---------------------------------

def test_yield_floor_is_applied_to_the_bid(monkeypatch):
    """A contract whose MID clears SCAV_MIN_PREMIUM_PCT but whose BID does not
    must be dropped — that is the XOM failure in miniature."""
    monkeypatch.setattr(cq, "SCAV_MIN_PREMIUM_PCT", 0.009)   # 0.9% floor
    # close 161.65, spread 19% so the guard lets it through:
    #   bid 1.40 -> 0.866% (FAILS the floor)
    #   mid 1.55 -> 0.959% (would have PASSED on the old mid pricing)
    s = {"signal": "SELL_PUT", "symbol": "XOM", "strike": 152.5,
         "close": 161.65, "dte": 32}
    out = cq.requote_signal(_client(_one_strike_chain(1.40, 1.70)), s)
    assert out is None, "a bid-based yield below the floor must drop the signal"


def test_max_loss_uses_the_bid():
    s = {"signal": "SELL_PUT", "symbol": "XOM", "strike": 152.5,
         "close": 161.65, "dte": 32}
    out = cq.requote_signal(_client(_one_strike_chain(2.01, 2.40)), s)
    assert out is not None
    # (152.5 - 2.01) * 100, not (152.5 - 2.205) * 100
    assert out["max_loss"] == pytest.approx(15049.0)
