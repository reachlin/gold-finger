"""
Tests for allocator.py — deterministic capital allocation across a scan's
competing signals: cash-freeing signals first, then cash-consuming ones
ranked by premium/day per collateral dollar with a concentration penalty.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import allocator as al


def _put(sym, strike, premium, dte=30):
    return {"symbol": sym, "signal": "SELL_PUT", "strike": strike,
            "premium": premium, "dte": dte}


class TestCashNeeded:
    def test_sell_put_is_collateral(self):
        assert al.cash_needed(_put("KO", 79.0, 0.71)) == 7900.0

    def test_sell_call_and_resume_wheel_are_free(self):
        assert al.cash_needed({"signal": "SELL_CALL", "strike": 250.0}) == 0.0
        assert al.cash_needed({"signal": "RESUME_WHEEL", "shares": 3}) == 0.0
        assert al.cash_needed({"signal": "SELL_ETF", "shares": 7}) == 0.0

    def test_buy_etf_uses_share_cost(self):
        s = {"signal": "BUY_ETF", "shares": 7, "close": 80.7}
        assert al.cash_needed(s) == pytest.approx(564.9)

    def test_hold_shares_uses_share_cost(self):
        s = {"signal": "HOLD_SHARES", "shares": 2, "close": 262.0}
        assert al.cash_needed(s) == 524.0

    def test_buy_uses_budget(self):
        s = {"signal": "BUY", "entry": 199.0}
        assert al.cash_needed(s, budget_per_trade=600) == 600.0


class TestScore:
    def test_score_is_premium_per_day_per_collateral_dollar(self):
        # KO: 0.71 / (79 * 30); AMZN: 2.1 / (221 * 30)
        ko, amzn = _put("KO", 79.0, 0.71), _put("AMZN", 221.0, 2.1)
        assert al.score(ko) == pytest.approx(0.71 / (79.0 * 30))
        assert al.score(amzn) > al.score(ko)

    def test_concentration_penalty_halves_per_open_position(self):
        s = _put("KO", 79.0, 0.71)
        base = al.score(s)
        assert al.score(s, open_count=1) == pytest.approx(base * 0.5)
        assert al.score(s, open_count=2) == pytest.approx(base * 0.25)

    def test_edge_prior_multiplies_score(self):
        s = _put("UNH", 406.0, 4.0)
        assert al.score(s, prior=0.4) == pytest.approx(al.score(s) * 0.4)


class TestRankSignals:
    def test_cash_freeing_first_then_score_then_rest(self):
        resume = {"symbol": "MSFT", "signal": "RESUME_WHEEL", "shares": 1}
        call   = {"symbol": "IBM", "signal": "SELL_CALL", "strike": 280.0,
                  "premium": 1.5, "dte": 30}
        ko     = _put("KO", 79.0, 0.71)      # lower score
        amzn   = _put("AMZN", 221.0, 2.1)    # higher score
        buy    = {"symbol": "NVDA", "signal": "BUY", "entry": 199.0}
        ranked = al.rank_signals([ko, buy, amzn, call, resume])
        order  = [(s["symbol"], s["signal"]) for s in ranked]
        assert order[0] in [("MSFT", "RESUME_WHEEL"), ("IBM", "SELL_CALL")]
        assert order[1] in [("MSFT", "RESUME_WHEEL"), ("IBM", "SELL_CALL")]
        assert order[2] == ("AMZN", "SELL_PUT")
        assert order[3] == ("KO", "SELL_PUT")
        assert order[4] == ("NVDA", "BUY")

    def test_concentration_penalty_reorders(self):
        # AMZN scores higher raw, but 2 open AMZN positions halve it twice
        ko, amzn = _put("KO", 79.0, 0.71), _put("AMZN", 221.0, 2.1)
        ranked = al.rank_signals([amzn, ko], open_counts={"AMZN": 2})
        assert ranked[0]["symbol"] == "KO"

    def test_empty_and_single_are_safe(self):
        assert al.rank_signals([]) == []
        one = [_put("KO", 79.0, 0.71)]
        assert al.rank_signals(one) == one

    def test_priors_reorder_puts(self):
        """v2: a modest-density name with a strong edge prior outranks a
        juicy-density name with a weak prior — the fix for the -$47K
        measurement where density starved UNH."""
        ko  = _put("KO", 79.0, 0.71)     # density 3.0e-4, weak prior
        unh = _put("UNH", 406.0, 3.2)    # density 2.6e-4, strong prior
        ranked = al.rank_signals([ko, unh],
                                 priors={"KO": 0.1, "UNH": 0.4})
        assert ranked[0]["symbol"] == "UNH"

    def test_score_puts_false_keeps_put_order_neutral(self):
        """Portfolio backtest 2026-07-04: density-ranked puts LOST -$47K
        vs neutral order at $30K — the live scanner disables put scoring
        until a v2 score measures positive."""
        ko, amzn = _put("KO", 79.0, 0.71), _put("AMZN", 221.0, 2.1)
        call = {"symbol": "IBM", "signal": "SELL_CALL", "strike": 280.0}
        ranked = al.rank_signals([ko, amzn, call], score_puts=False)
        assert [s["symbol"] for s in ranked] == ["IBM", "KO", "AMZN"]

    def test_ranking_never_drops_signals(self):
        sigs = [_put("KO", 79.0, 0.71), _put("AMZN", 221.0, 2.1),
                {"symbol": "X", "signal": "WEIRD"}]
        assert len(al.rank_signals(sigs)) == 3
