"""
Tests for vault20/option_finder.py

Run: /Users/lincai/anaconda3/envs/gold-finger/bin/python -m pytest vault20/test_option_finder.py -v
"""
import sys, os
from datetime import date, timedelta
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vault20.option_finder import (
    score_covered_call,
    score_csp,
    filter_calls,
    filter_puts,
    format_candidates,
    _parse_schwab_exp_map,
)


def _make_calls(rows):
    return pd.DataFrame(rows, columns=[
        "contractSymbol", "strike", "bid", "ask", "impliedVolatility",
        "openInterest", "volume", "inTheMoney",
    ])


def _make_puts(rows):
    return _make_calls(rows)


TODAY = date.today()


def _dte(days): return (TODAY + timedelta(days=days)).isoformat()


class TestScoreCoveredCall:
    def test_returns_dict_with_required_keys(self):
        s = score_covered_call(strike=130, bid=2.0, stock_price=128.0, dte=30)
        for k in ("monthly_yield_pct", "otm_pct", "annual_yield_pct", "breakeven"):
            assert k in s

    def test_monthly_yield(self):
        # bid=2, stock=100, dte=30 → yield = 2/100 * (30/30) * 100 = 2%
        s = score_covered_call(130, 2.0, 100.0, 30)
        assert abs(s["monthly_yield_pct"] - 2.0) < 0.01

    def test_otm_pct(self):
        # strike=130, stock=128 → (130-128)/128*100 = 1.5625%
        s = score_covered_call(130, 2.0, 128.0, 30)
        assert abs(s["otm_pct"] - 1.5625) < 0.01

    def test_breakeven(self):
        # breakeven = stock - bid = 128 - 2 = 126
        s = score_covered_call(130, 2.0, 128.0, 30)
        assert abs(s["breakeven"] - 126.0) < 0.01

    def test_zero_bid_yields_zero(self):
        s = score_covered_call(130, 0.0, 128.0, 30)
        assert s["monthly_yield_pct"] == 0.0


class TestScoreCSP:
    def test_returns_required_keys(self):
        s = score_csp(strike=125, bid=2.0, stock_price=128.0, dte=30)
        for k in ("monthly_yield_pct", "otm_pct", "annual_yield_pct", "breakeven"):
            assert k in s

    def test_otm_pct_put(self):
        # strike=125, stock=128 → (128-125)/128*100 = 2.34%
        s = score_csp(125, 2.0, 128.0, 30)
        assert abs(s["otm_pct"] - 2.34375) < 0.01

    def test_breakeven_put(self):
        # breakeven = strike - bid = 125 - 2 = 123
        s = score_csp(125, 2.0, 128.0, 30)
        assert abs(s["breakeven"] - 123.0) < 0.01

    def test_monthly_yield_on_collateral(self):
        # yield = bid / strike (collateral) / (dte/30) * 100
        # bid=2, strike=125, dte=30 → 2/125*100 = 1.6%
        s = score_csp(125, 2.0, 128.0, 30)
        assert abs(s["monthly_yield_pct"] - 1.6) < 0.01


class TestFilterCalls:
    def _chain(self):
        return _make_calls([
            ("SYM1", 130, 2.00, 2.10, 0.30, 500, 100, False),  # good
            ("SYM2", 120, 3.00, 3.10, 0.40, 200,  50, True),   # ITM — skip
            ("SYM3", 135, 0.05, 0.10, 0.15,  10,   5, False),  # low OI — skip
            ("SYM4", 132, 1.50, 1.60, 0.25, 300,  80, False),  # good
        ])

    def test_removes_itm(self):
        df = self._chain()
        result = filter_calls(df, stock_price=128.0, min_oi=50, min_otm_pct=0)
        assert all(~result["inTheMoney"])

    def test_removes_low_oi(self):
        df = self._chain()
        result = filter_calls(df, stock_price=128.0, min_oi=50, min_otm_pct=0)
        assert all(result["openInterest"] >= 50)

    def test_removes_low_otm(self):
        # min_otm_pct=2 means strike must be >= stock*1.02 = 130.56 → only 132, 135
        df = self._chain()
        result = filter_calls(df, stock_price=128.0, min_oi=50, min_otm_pct=2.0)
        assert all(result["strike"] >= 128.0 * 1.02)

    def test_empty_returns_empty(self):
        df = _make_calls([])
        result = filter_calls(df, stock_price=128.0, min_oi=50, min_otm_pct=0)
        assert len(result) == 0


class TestFilterPuts:
    def _chain(self):
        return _make_puts([
            ("SYM1", 125, 2.00, 2.10, 0.30, 500, 100, False),  # good
            ("SYM2", 132, 3.00, 3.10, 0.40, 200,  50, True),   # ITM — skip
            ("SYM3", 122, 0.05, 0.10, 0.15,  10,   5, False),  # low OI — skip
            ("SYM4", 120, 1.20, 1.30, 0.25, 300,  80, False),  # good
        ])

    def test_removes_itm(self):
        df = self._chain()
        result = filter_puts(df, stock_price=128.0, min_oi=50, min_otm_pct=0)
        assert all(~result["inTheMoney"])

    def test_removes_low_oi(self):
        df = self._chain()
        result = filter_puts(df, stock_price=128.0, min_oi=50, min_otm_pct=0)
        assert all(result["openInterest"] >= 50)


class TestParseSchwebExpMap:
    def _exp_map(self, strike, bid, ask, oi, delta, theta, vega, gamma, itm=False):
        return {
            "2026-07-24:26": {
                str(float(strike)): [{
                    "bid": bid, "ask": ask, "openInterest": oi,
                    "totalVolume": 100, "volatility": 94.0,
                    "delta": delta, "gamma": gamma,
                    "theta": theta, "vega": vega,
                    "inTheMoney": itm,
                }]
            }
        }

    def test_basic_call_parsed(self):
        exp_map = self._exp_map(130, 12.0, 12.5, 500, 0.52, -0.15, 0.20, 0.01)
        rows = _parse_schwab_exp_map(exp_map, "covered-call", 128.0, 100, 1.0)
        assert len(rows) == 1
        r = rows[0]
        assert r["strike"]  == 130.0
        assert r["bid"]     == 12.0
        assert r["delta"]   == 0.52
        assert r["theta"]   == -0.15
        assert r["vega"]    == 0.20
        assert r["gamma"]   == 0.01
        assert r["expiry"]  == "2026-07-24"
        assert r["dte"]     == 26

    def test_itm_filtered(self):
        exp_map = self._exp_map(125, 5.0, 5.5, 500, 0.75, -0.20, 0.15, 0.02, itm=True)
        rows = _parse_schwab_exp_map(exp_map, "covered-call", 128.0, 100, 1.0)
        assert len(rows) == 0

    def test_low_oi_filtered(self):
        exp_map = self._exp_map(130, 12.0, 12.5, 50, 0.52, -0.15, 0.20, 0.01)
        rows = _parse_schwab_exp_map(exp_map, "covered-call", 128.0, 100, 1.0)
        assert len(rows) == 0

    def test_iv_converted_from_percentage(self):
        exp_map = self._exp_map(130, 12.0, 12.5, 500, 0.52, -0.15, 0.20, 0.01)
        rows = _parse_schwab_exp_map(exp_map, "covered-call", 128.0, 100, 1.0)
        # Schwab sends 94.0, should be stored as 0.94
        assert abs(rows[0]["iv"] - 0.94) < 0.001

    def test_scoring_fields_present(self):
        exp_map = self._exp_map(130, 12.0, 12.5, 500, 0.52, -0.15, 0.20, 0.01)
        rows = _parse_schwab_exp_map(exp_map, "covered-call", 128.0, 100, 1.0)
        for key in ("monthly_yield_pct", "otm_pct", "annual_yield_pct", "breakeven"):
            assert key in rows[0]


class TestFormatCandidates:
    def test_returns_list_of_dicts(self):
        rows = [
            {"expiry": "2026-07-18", "strike": 130.0, "bid": 2.0, "ask": 2.1,
             "dte": 20, "openInterest": 500, "iv": 0.30,
             "monthly_yield_pct": 1.5, "otm_pct": 1.6, "breakeven": 126.0},
        ]
        out = format_candidates(rows)
        assert isinstance(out, list)
        assert len(out) == 1
        assert "monthly_yield_pct" in out[0]

    def test_sorted_by_monthly_yield(self):
        rows = [
            {"expiry": "2026-07-18", "strike": 130.0, "bid": 1.0, "ask": 1.1,
             "dte": 20, "openInterest": 500, "iv": 0.30,
             "monthly_yield_pct": 1.0, "otm_pct": 1.6, "breakeven": 127.0},
            {"expiry": "2026-07-18", "strike": 132.0, "bid": 2.0, "ask": 2.1,
             "dte": 20, "openInterest": 300, "iv": 0.28,
             "monthly_yield_pct": 2.5, "otm_pct": 3.1, "breakeven": 126.0},
        ]
        out = format_candidates(rows)
        assert out[0]["monthly_yield_pct"] >= out[1]["monthly_yield_pct"]
