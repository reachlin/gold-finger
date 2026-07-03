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
        client = _mock_client(_chain_response())
        q = cq.fetch_chain_quote(client, "KO", "PUT", target_strike=79.13,
                                 target_dte=30)
        assert q["premium"] == pytest.approx((1.10 + 1.30) / 2)

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
        assert s["premium"] == pytest.approx(1.20)
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
        # Crush the quotes so premium/close falls below the 0.5% floor
        for strikes in payload["putExpDateMap"].values():
            for opts in strikes.values():
                opts[0]["bid"], opts[0]["ask"] = 0.05, 0.15
        client = _mock_client(payload)
        assert cq.requote_signal(client, self._signal()) is None

    def test_non_option_signals_pass_through(self):
        client = _mock_client(_chain_response())
        s = {"symbol": "NVDA", "signal": "BUY", "entry": 100.0}
        assert cq.requote_signal(client, s) is s
        client.get_option_chain.assert_not_called()
