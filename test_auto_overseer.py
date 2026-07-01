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
