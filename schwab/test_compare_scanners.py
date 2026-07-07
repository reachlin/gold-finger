"""
Tests for compare_scanners.py — side-by-side report of the A/B paper-trading
experiment (main worktree vs portfolio-management branch).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import compare_scanners as cs


FAKE_LOG = """\
===SIGNAL_START===
TIME:    2026-07-06 09:35:12 ET (local 21:35 CST)
SYMBOL:  KO  [SCAVENGER]
ACTION:  SELL_PUT
CLOSE:   $80.7
===SIGNAL_END===

  🤖 AutoOverseer [anthropic/claude-haiku-4-5]: ✅ Y  —  good premium, quality name

===SIGNAL_START===
TIME:    2026-07-06 09:40:03 ET (local 21:40 CST)
SYMBOL:  MSFT  [ROUTER]
ACTION:  HOLD_SHARES
===SIGNAL_END===

  🤖 AutoOverseer [anthropic/claude-haiku-4-5]: ⛔ N  —  earnings in 3 days
"""


class TestParseDecisions:
    def test_extracts_symbol_action_verdict_reason(self):
        out = cs.parse_decisions(FAKE_LOG)
        assert len(out) == 2
        assert out[0] == {"time": "2026-07-06 09:35:12",
                          "symbol": "KO", "action": "SELL_PUT",
                          "verdict": "Y",
                          "reason": "good premium, quality name"}
        assert out[1]["symbol"] == "MSFT"
        assert out[1]["action"] == "HOLD_SHARES"
        assert out[1]["verdict"] == "N"

    def test_block_without_verdict_is_dropped(self):
        text = FAKE_LOG.split("🤖")[0]          # first block, no verdict line
        assert cs.parse_decisions(text) == []

    def test_empty_log(self):
        assert cs.parse_decisions("") == []


class TestLedgerSummary:
    def test_counts_and_premiums(self):
        rows = [
            {"symbol": "KO", "signal": "SELL_PUT", "verdict": "APPROVED",
             "premium_ct": "71.0"},
            {"symbol": "PG", "signal": "SELL_PUT", "verdict": "APPROVED",
             "premium_ct": "116.0"},
            {"symbol": "NVDA", "signal": "SELL_PUT", "verdict": "SKIPPED",
             "premium_ct": "573.0"},
            {"symbol": "META", "signal": "SELL_PUT", "verdict": "BUDGET_BLOCK",
             "premium_ct": "1637.0"},
        ]
        s = cs.ledger_summary(rows)
        assert s["by_verdict"]["APPROVED"] == 2
        assert s["by_verdict"]["SKIPPED"] == 1
        assert s["approved_premium"] == pytest.approx(187.0)
        assert s["approved_symbols"] == ["KO", "PG"]


class TestDiffDecisions:
    def test_pairs_by_day_symbol_action(self):
        a = [{"time": "2026-07-06 09:35:12", "symbol": "KO",
              "action": "SELL_PUT", "verdict": "Y", "reason": "r1"}]
        b = [{"time": "2026-07-06 10:05:00", "symbol": "KO",
              "action": "SELL_PUT", "verdict": "N", "reason": "r2"},
             {"time": "2026-07-06 10:06:00", "symbol": "PG",
              "action": "SELL_PUT", "verdict": "Y", "reason": "r3"}]
        diff = cs.diff_decisions(a, b)
        assert ("2026-07-06", "KO", "SELL_PUT") in diff["disagreements"]
        assert ("2026-07-06", "PG", "SELL_PUT") in diff["only_b"]
        assert diff["only_a"] == {}
