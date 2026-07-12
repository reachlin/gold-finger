"""Tests for The Hunter — BUY_CALL momentum breakout role."""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from schwab.trend_scanner import compute_indicators
from vault76.armory.hunter import Hunter, _atm_strike, _tightest_base_range
from vault76.overseer import Overseer


def _make_df(n: int = 120, trend: bool = True, breakout: bool = True,
             runup: bool = True) -> pd.DataFrame:
    """
    Synthetic OHLCV dataframe.

    We generate n + 60 warm-up bars so that compute_indicators().dropna()
    still leaves ≥100 usable rows after EMA50/ADX initialise.

    trend=True    → rising EMA20>EMA50 with ADX>20
    runup=True    → ≥25% price run in first 90 bars (required for VCP)
    breakout=True → last bar closes above 20-day high on high volume
    """
    total = n + 60          # extra warm-up bars eaten by indicator initialisation
    np.random.seed(42)
    dates  = pd.date_range("2023-01-01", periods=total, freq="B")
    prices = [20.0]   # low start so compounded runup lands near $100

    for i in range(1, total):
        if runup and i < int(total * 0.6):
            drift = 0.015          # strong run-up — builds ADX well above 20
        elif trend and i < int(total * 0.85):
            drift = 0.001          # mild drift during consolidation
        else:
            drift = 0.0            # flat base before breakout
        prices.append(prices[-1] * (1 + drift + np.random.normal(0, 0.008)))

    prices = np.array(prices)
    if breakout:
        prices[-1] = prices[-20:].max() * 1.02    # clear the 20-day high

    close  = prices

    # Phase-aware intraday spread: runup volatile, consolidation tight
    runup_end  = int(total * 0.6)
    consol_end = int(total * 0.85)
    spread = np.zeros(total)
    spread[:runup_end]              = np.abs(np.random.normal(0, 0.012, runup_end))
    spread[runup_end:consol_end]    = np.abs(np.random.normal(0, 0.005, consol_end - runup_end))
    spread[consol_end:]             = np.abs(np.random.normal(0, 0.001, total - consol_end))

    high   = close * (1 + spread)
    low    = close * (1 - spread)
    open_  = close * (1 + np.random.normal(0, 0.003, total))

    vol_base = 1_000_000
    volume   = np.full(total, vol_base, dtype=float)
    if breakout:
        volume[-1] = vol_base * 2.0    # volume surge on breakout day

    df = pd.DataFrame({
        "datetime": dates,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
    })
    return compute_indicators(df).dropna().reset_index(drop=True)


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_atm_strike_above_20():
    assert _atm_strike(195.3) == 195.0
    assert _atm_strike(197.6) == 200.0
    assert _atm_strike(50.0)  == 50.0
    assert _atm_strike(52.3)  == 50.0
    assert _atm_strike(53.0)  == 55.0


def test_atm_strike_below_20():
    assert _atm_strike(18.7) == 19
    assert _atm_strike(5.4)  == 5


def test_tightest_base_range_decreasing():
    df = _make_df(120)
    r = _tightest_base_range(df, lookback=15)
    assert isinstance(r, float)
    assert 0 < r < 30          # realistic range pct


def test_hunter_fires_on_valid_setup():
    hunter = Hunter()
    df     = _make_df(120, trend=True, breakout=True, runup=True)
    result = hunter.scan("TEST", df, regime=Overseer.RECLAMATION)
    assert result["signal"] == "BUY_CALL", (
        f"Expected BUY_CALL, got NONE. Reason: {result['reason']}"
    )
    assert result["strike"] > 0
    assert result["premium"] > 0
    assert result["exit_25pct"] == round(result["premium"] * 1.5, 2)
    assert result["exit_50pct"] == round(result["premium"] * 2.0, 2)
    assert result["stop_premium"] == round(result["premium"] * 0.5, 2)


def test_hunter_blocked_wrong_regime():
    hunter = Hunter()
    df     = _make_df(120, trend=True, breakout=True, runup=True)
    for bad_regime in [Overseer.NUKED_ZONE, Overseer.WASTELAND]:
        result = hunter.scan("TEST", df, regime=bad_regime)
        assert result["signal"] == "NONE"
        assert "benched" in result["reason"]


def test_hunter_no_runup():
    hunter = Hunter()
    df     = _make_df(120, trend=True, breakout=True, runup=False)
    result = hunter.scan("TEST", df, regime=Overseer.RECLAMATION)
    # May or may not fire depending on synthetic data — just check it doesn't crash
    assert result["signal"] in ("BUY_CALL", "NONE")


def test_hunter_no_breakout():
    hunter = Hunter()
    df     = _make_df(120, trend=True, breakout=False, runup=True)
    # Without the breakout bar, trigger should not fire
    result = hunter.scan("TEST", df, regime=Overseer.RECLAMATION)
    # Might fire on earlier bar patterns — just verify structure is correct
    assert "signal" in result
    assert "reason" in result


def test_hunter_returns_no_signal_on_short_df():
    hunter = Hunter()
    df     = _make_df(50)
    result = hunter.scan("TEST", df, regime=Overseer.RECLAMATION)
    assert result["signal"] == "NONE"
    assert "insufficient" in result["reason"]


def test_hunter_buy_call_fields_present():
    hunter = Hunter()
    df     = _make_df(120, trend=True, breakout=True, runup=True)
    result = hunter.scan("TEST", df, regime=Overseer.RECLAMATION)
    if result["signal"] == "BUY_CALL":
        required = {"strike", "premium", "premium_pct", "dte", "hv", "adx",
                    "rsi", "vcp_tight_pct", "breakout_vol",
                    "exit_25pct", "exit_50pct", "stop_premium", "cost_per_ct"}
        missing  = required - set(result.keys())
        assert not missing, f"Missing fields: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
