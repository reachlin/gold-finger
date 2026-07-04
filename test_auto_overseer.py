"""Tests for auto_overseer — run before implementation (TDD)."""
import json
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Tests for build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_contains_signal_fields():
    from schwab.auto_overseer import build_prompt
    signal = {"symbol": "KO", "signal": "SELL_PUT", "strike": "79.13",
              "premium": 0.71, "hv": 24.6, "adx": 15.1}
    prompt = build_prompt(signal, {"available": 22087, "required": 7913}, {})
    assert "KO" in prompt
    assert "SELL_PUT" in prompt
    assert "79.13" in prompt


def test_build_prompt_includes_kronos_warning():
    from schwab.auto_overseer import build_prompt
    signal = {"symbol": "KO", "signal": "SELL_PUT", "strike": "79.13"}
    kronos = {"support": 70.0, "resistance": 85.0, "buf_pct": -5.0, "warn": True}
    prompt = build_prompt(signal, {"available": 22087, "required": 7913}, kronos)
    assert "70.0" in prompt or "70.00" in prompt
    assert "YES" in prompt  # warning flag


def test_build_prompt_no_kronos_section_when_empty():
    from schwab.auto_overseer import build_prompt
    signal = {"symbol": "IBM", "signal": "SELL_PUT", "strike": "258.69"}
    prompt = build_prompt(signal, {"available": 30000, "required": 25869}, {})
    assert "IBM" in prompt
    assert "Kronos" not in prompt


def test_build_prompt_portfolio_state():
    from schwab.auto_overseer import build_prompt
    prompt = build_prompt({"symbol": "X", "signal": "SELL_PUT"}, {"available": 12345, "required": 5000}, {})
    assert "12,345" in prompt or "12345" in prompt


def test_build_prompt_lists_open_positions_on_symbol():
    from schwab.auto_overseer import build_prompt
    signal = {"symbol": "KO", "signal": "SELL_PUT", "strike": "79.13"}
    open_positions = [
        {"signal": "SELL_PUT", "strike": "80.00", "premium_ct": "75.0",
         "date": "2026-06-20 10:00:00", "dte": "30"},
        {"signal": "SELL_PUT", "strike": "78.50", "premium_ct": "68.0",
         "date": "2026-06-25 10:00:00", "dte": "30"},
    ]
    prompt = build_prompt(signal, {"available": 30000, "required": 7913}, {},
                          open_positions=open_positions)
    assert "Existing open positions on KO (2)" in prompt
    assert "80.00" in prompt
    assert "78.50" in prompt


def test_build_prompt_says_none_when_no_open_positions():
    from schwab.auto_overseer import build_prompt
    signal = {"symbol": "KO", "signal": "SELL_PUT", "strike": "79.13"}
    prompt = build_prompt(signal, {"available": 30000, "required": 7913}, {})
    assert "Existing open positions on KO: none" in prompt


def test_overseer_system_allows_duplicates():
    """The system prompt must not claim duplicates are pre-filtered."""
    from schwab.auto_overseer import OVERSEER_SYSTEM
    assert "duplicate positions are already filtered" not in OVERSEER_SYSTEM
    assert "ALLOWED" in OVERSEER_SYSTEM


def test_overseer_system_documents_lgbm_advisories():
    """Field guide must cover every LGBM field the scanner can attach."""
    from schwab.auto_overseer import OVERSEER_SYSTEM
    for field in ("assign_risk_pct", "called_away_pct", "drop_risk_pct",
                  "model_auc", "timesfm_30d_pct"):
        assert field in OVERSEER_SYSTEM, f"missing field guide for {field}"


def test_overseer_system_documents_router_signals():
    """Router signals: default approve (backtested policy), veto on context."""
    from schwab.auto_overseer import OVERSEER_SYSTEM
    assert "HOLD_SHARES" in OVERSEER_SYSTEM
    assert "RESUME_WHEEL" in OVERSEER_SYSTEM
    assert "default is APPROVE" in OVERSEER_SYSTEM


def test_real_order_skipped_for_router_signals(monkeypatch, capsys):
    """Router signals are strategy-state changes, not options orders —
    real mode must never try to build an OCC order from them."""
    from schwab.auto_overseer import AutoOverseer
    monkeypatch.setenv("REALLY_REAL", "true")
    ao = AutoOverseer.__new__(AutoOverseer)          # skip LLM init
    for sig in ("HOLD_SHARES", "RESUME_WHEEL"):
        ao._place_real_order({"signal": sig, "symbol": "NVDA"})
        assert "no automated real order" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Tests for OCC option symbol builder
# ---------------------------------------------------------------------------

def test_build_occ_symbol_put():
    from datetime import date
    from schwab.auto_overseer import build_occ_symbol
    # KO $79.13 put expiring 2026-07-31
    occ = build_occ_symbol("KO", date(2026, 7, 31), "SELL_PUT", 79.13)
    assert occ == "KO    260731P00079130"


def test_build_occ_symbol_call():
    """SELL_CALL must produce a C contract — not a hardcoded P."""
    from datetime import date
    from schwab.auto_overseer import build_occ_symbol
    occ = build_occ_symbol("NVDA", date(2026, 8, 21), "SELL_CALL", 250.0)
    assert occ == "NVDA  260821C00250000"


# ---------------------------------------------------------------------------
# Tests for deterministic market calendar
# ---------------------------------------------------------------------------

def test_market_open_regular_weekday():
    from datetime import date
    from schwab.auto_overseer import market_open_on
    assert market_open_on(date(2026, 7, 1)) is True     # Wednesday

def test_market_closed_weekend():
    from datetime import date
    from schwab.auto_overseer import market_open_on
    assert market_open_on(date(2026, 7, 5)) is False    # Sunday

def test_market_closed_independence_day_observed():
    from datetime import date
    from schwab.auto_overseer import market_open_on
    # July 4 2026 is a Saturday → NYSE observes the holiday on Friday July 3
    assert market_open_on(date(2026, 7, 3)) is False


# ---------------------------------------------------------------------------
# Tests for parse_llm_response
# ---------------------------------------------------------------------------

def test_parse_yes_response():
    from schwab.auto_overseer import parse_llm_response
    resp = '{"decision": "yes", "reason": "good setup, HV low"}'
    decision, reason = parse_llm_response(resp)
    assert decision == "yes"
    assert "good setup" in reason


def test_parse_no_response():
    from schwab.auto_overseer import parse_llm_response
    resp = '{"decision": "no", "reason": "HV too high"}'
    decision, reason = parse_llm_response(resp)
    assert decision == "no"
    assert "HV" in reason


def test_parse_malformed_falls_back_to_no():
    from schwab.auto_overseer import parse_llm_response
    decision, reason = parse_llm_response("not json at all")
    assert decision == "no"
    assert len(reason) > 0


def test_parse_invalid_decision_value():
    from schwab.auto_overseer import parse_llm_response
    resp = '{"decision": "maybe", "reason": "unsure"}'
    decision, reason = parse_llm_response(resp)
    assert decision == "no"


def test_parse_response_with_whitespace():
    from schwab.auto_overseer import parse_llm_response
    resp = '  {"decision": "yes", "reason": "strong setup"}  '
    decision, reason = parse_llm_response(resp)
    assert decision == "yes"


# ---------------------------------------------------------------------------
# Tests for LLMClient
# ---------------------------------------------------------------------------

def test_llm_client_anthropic_calls_sdk():
    from schwab.llm_client import LLMClient
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text='{"decision": "no", "reason": "test"}')]
    with patch("anthropic.Anthropic") as MockAnthropic:
        instance = MockAnthropic.return_value
        instance.messages.create.return_value = mock_resp
        client = LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001", api_key="test")
        result = client.chat(system="sys", user="user msg")
    assert result == '{"decision": "no", "reason": "test"}'


def test_llm_client_openai_compatible_calls_sdk():
    from schwab.llm_client import LLMClient
    mock_choice = MagicMock()
    mock_choice.message.content = '{"decision": "yes", "reason": "ok"}'
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    with patch("openai.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = mock_resp
        client = LLMClient(provider="openai", model="gpt-4o-mini", api_key="test")
        result = client.chat(system="sys", user="user msg")
    assert result == '{"decision": "yes", "reason": "ok"}'


def test_llm_client_error_returns_empty_string():
    from schwab.llm_client import LLMClient
    with patch("anthropic.Anthropic") as MockAnthropic:
        instance = MockAnthropic.return_value
        instance.messages.create.side_effect = Exception("network error")
        client = LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001", api_key="test")
        result = client.chat(system="sys", user="user msg")
    assert result == ""
