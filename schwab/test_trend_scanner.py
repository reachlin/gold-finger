import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from trend_scanner import (
    compute_indicators,
    detect_trend,
    detect_pullback,
    detect_entry,
    detect_market_regime,
    compute_levels,
    scan_symbol,
    detect_prior_runup,
    detect_tight_consolidation,
    detect_breakout_trigger,
    compute_breakout_levels,
    position_size_by_risk,
    RUNUP_LOOKBACK,
    CONSOLIDATION_LOOKBACK,
)


def _make_trending_df(n=120, drift=0.002):
    """Steadily uptrending OHLCV data."""
    np.random.seed(1)
    close = 100 * np.cumprod(1 + drift + np.random.randn(n) * 0.01)
    df = pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=n, freq="B"),
        "open":  close * 0.995,
        "high":  close * 1.015,
        "low":   close * 0.985,
        "close": close,
        "volume": np.random.randint(10_000_000, 20_000_000, n).astype(float),
    })
    return df


def _make_flat_df(n=120):
    """Flat/choppy data — no trend."""
    np.random.seed(2)
    close = 100 + np.random.randn(n) * 1.5
    df = pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=n, freq="B"),
        "open":  close * 0.999,
        "high":  close * 1.01,
        "low":   close * 0.99,
        "close": close,
        "volume": np.random.randint(5_000_000, 10_000_000, n).astype(float),
    })
    return df


def test_compute_indicators_columns():
    df = _make_trending_df()
    out = compute_indicators(df)
    for col in ("ema20", "ema50", "adx", "rsi", "atr"):
        assert col in out.columns, f"Missing column: {col}"


def test_detect_trend_true_for_uptrend():
    df = compute_indicators(_make_trending_df(120))
    df = df.dropna().reset_index(drop=True)
    assert detect_trend(df) is True


def test_detect_trend_false_for_flat():
    df = compute_indicators(_make_flat_df(120))
    df = df.dropna().reset_index(drop=True)
    assert detect_trend(df) is False


def test_detect_pullback_returns_bool():
    df = compute_indicators(_make_trending_df())
    df = df.dropna().reset_index(drop=True)
    result = detect_pullback(df)
    assert isinstance(result, bool)


def test_detect_entry_returns_bool():
    df = compute_indicators(_make_trending_df())
    df = df.dropna().reset_index(drop=True)
    result = detect_entry(df)
    assert isinstance(result, bool)


def test_compute_levels():
    levels = compute_levels(entry=200.0)
    assert levels["target"] == pytest.approx(240.0)
    assert levels["stop"]   == pytest.approx(184.0)
    assert levels["risk_reward"] == pytest.approx(2.5)


def test_scan_symbol_no_signal_on_flat():
    df = compute_indicators(_make_flat_df())
    df = df.dropna().reset_index(drop=True)
    result = scan_symbol("TEST", df)
    assert result["signal"] == "NONE"


def test_scan_symbol_structure():
    df = compute_indicators(_make_trending_df())
    df = df.dropna().reset_index(drop=True)
    result = scan_symbol("NVDA", df)
    assert "symbol" in result
    assert "signal" in result
    assert result["signal"] in ("BUY", "NONE")
    if result["signal"] == "BUY":
        assert result["target"] > result["entry"]
        assert result["stop"]   < result["entry"]


# ── detect_market_regime ───────────────────────────────────────────────────

def test_regime_true_for_uptrending_index():
    df = compute_indicators(_make_trending_df(120))
    df = df.dropna().reset_index(drop=True)
    assert detect_market_regime(df) is True


def test_regime_false_for_flat_index():
    df = compute_indicators(_make_flat_df(120))
    df = df.dropna().reset_index(drop=True)
    assert detect_market_regime(df) is False


def test_regime_false_when_price_below_ema50():
    """Build an uptrend then crash below EMA50."""
    np.random.seed(5)
    n = 120
    # First 100 bars: strong uptrend to build EMA
    close_up = 100 * np.cumprod(1 + 0.003 + np.random.randn(100) * 0.005)
    # Last 20 bars: sharp crash below EMA50
    close_down = close_up[-1] * np.cumprod(1 - 0.03 + np.random.randn(20) * 0.005)
    close = np.concatenate([close_up, close_down])
    df = pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=n, freq="B"),
        "open":  close * 0.995,
        "high":  close * 1.01,
        "low":   close * 0.985,
        "close": close,
        "volume": np.ones(n) * 1e7,
    })
    df_ind = compute_indicators(df).dropna().reset_index(drop=True)
    assert detect_market_regime(df_ind) is False


def test_regime_returns_bool():
    df = compute_indicators(_make_trending_df())
    df = df.dropna().reset_index(drop=True)
    assert isinstance(detect_market_regime(df), bool)


def test_scan_symbol_blocked_by_bad_regime():
    """A valid BUY signal should be suppressed when regime_ok=False."""
    # Use a strongly trending stock df that would normally get a BUY
    stock_df = compute_indicators(_make_trending_df(200, drift=0.003))
    stock_df = stock_df.dropna().reset_index(drop=True)
    flat_spy  = compute_indicators(_make_flat_df(200))
    flat_spy  = flat_spy.dropna().reset_index(drop=True)

    regime_bad  = detect_market_regime(flat_spy)   # False
    result_bad  = scan_symbol("TEST", stock_df, regime_ok=regime_bad)

    # Whether or not a BUY fires without regime gate, with regime_ok=False it must be NONE
    assert result_bad["signal"] == "NONE"
    if not regime_bad:
        assert result_bad["reason"] == "market regime bearish"


def test_scan_symbol_regime_ok_true_allows_signal():
    """regime_ok=True should not block signals that would otherwise fire."""
    stock_df = compute_indicators(_make_trending_df(200, drift=0.003))
    stock_df = stock_df.dropna().reset_index(drop=True)
    # With regime allowed, result should be the normal scan output
    result = scan_symbol("TEST", stock_df, regime_ok=True)
    assert result["signal"] in ("BUY", "NONE")   # regime doesn't force BUY, just allows it


# ── Qullamaggie-style breakout helpers (run-up + tight consolidation + breakout) ──

def _make_breakout_df(runup_pct=0.35, breakout_pct=0.08, breakout_vol_mult=3.0,
                       runup_bars=80, consolidation_bars=CONSOLIDATION_LOOKBACK,
                       shrink_from=0.03, shrink_to=0.01, prehistory_bars=60):
    """
    Phase 0 (prehistory): flat filler so EMA50/ATR warm-up (dropna) doesn't
                           eat into the run-up itself.
    Phase 1 (run-up):     smooth climb of `runup_pct` over `runup_bars`.
    Phase 2 (consolidation): tight range narrowing from `shrink_from` to
                              `shrink_to` (as a % of price), higher lows,
                              price hugging/surfing the rising EMA20.
    Phase 3 (breakout):   two bars — the last a range-expansion close above
                           the consolidation highs on a volume surge.
    """
    close = []
    high = []
    low = []
    volume = []

    # Phase 0: flat prehistory (purely warm-up filler, gets dropna'd away)
    for _ in range(prehistory_bars):
        close.append(100.0)
        high.append(101.0)
        low.append(99.0)
        volume.append(5_000_000)

    # Phase 1: run-up
    base_start = 100.0
    base_end = base_start * (1 + runup_pct)
    for i in range(runup_bars):
        frac = i / (runup_bars - 1)
        c = base_start * (base_end / base_start) ** frac
        close.append(c)
        high.append(c * 1.02)
        low.append(c * 0.98)
        volume.append(5_000_000)

    # Phase 2: tight consolidation — range narrows, lows rise
    for i in range(consolidation_bars):
        frac = i / (consolidation_bars - 1)
        rng = shrink_from - (shrink_from - shrink_to) * frac
        center = base_end * (1 + 0.003 * frac)
        close.append(center)
        high.append(center * (1 + rng / 2))
        low.append(center * (1 - rng / 2))
        volume.append(4_000_000)

    # Phase 3: quiet bar, then the breakout bar
    last_center = close[-1]
    close.append(last_center * 1.002)
    high.append(last_center * 1.008)
    low.append(last_center * 0.996)
    volume.append(4_200_000)

    breakout_close = close[-1] * (1 + breakout_pct)
    avg_recent_vol = sum(volume[-14:]) / 14
    close.append(breakout_close)
    high.append(breakout_close * 1.01)
    low.append(close[-2] * 1.0)
    volume.append(avg_recent_vol * breakout_vol_mult)

    n = len(close)
    return pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=n, freq="B"),
        "open":  [low[i] for i in range(n)],
        "high":  high,
        "low":   low,
        "close": close,
        "volume": volume,
    })


def _make_no_runup_df(n=97):
    """Flat/choppy — never had a qualifying prior move."""
    return _make_flat_df(n)


def _make_choppy_consolidation_df(runup_pct=0.35, runup_bars=80,
                                   consolidation_bars=CONSOLIDATION_LOOKBACK):
    """Run-up happened, but the 'consolidation' just keeps making new lower
    lows with a wide, non-contracting range — not tradeable per Qullamaggie."""
    np.random.seed(11)
    close = []
    high = []
    low = []
    volume = []
    base_start = 100.0
    base_end = base_start * (1 + runup_pct)
    for i in range(runup_bars):
        frac = i / (runup_bars - 1)
        c = base_start * (base_end / base_start) ** frac
        close.append(c)
        high.append(c * 1.02)
        low.append(c * 0.98)
        volume.append(5_000_000)

    for i in range(consolidation_bars):
        c = base_end * (1 + np.random.randn() * 0.05)   # wide, random chop
        close.append(c)
        high.append(c * 1.06)
        low.append(c * 0.94)
        volume.append(4_000_000)

    n = len(close)
    return pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=n, freq="B"),
        "open":  low,
        "high":  high,
        "low":   low,
        "close": close,
        "volume": volume,
    })


def test_detect_prior_runup_true_after_big_move():
    df = compute_indicators(_make_breakout_df()).dropna().reset_index(drop=True)
    assert detect_prior_runup(df) is True


def test_detect_prior_runup_false_when_flat():
    df = compute_indicators(_make_no_runup_df()).dropna().reset_index(drop=True)
    assert detect_prior_runup(df) is False


def test_detect_tight_consolidation_true_for_breakout_setup():
    # Evaluate on the quiet bar just before the breakout bar (still consolidating)
    full = compute_indicators(_make_breakout_df()).dropna().reset_index(drop=True)
    df = full.iloc[:-1].reset_index(drop=True)
    assert detect_tight_consolidation(df) is True


def test_detect_tight_consolidation_false_when_choppy():
    df = compute_indicators(_make_choppy_consolidation_df()).dropna().reset_index(drop=True)
    assert detect_tight_consolidation(df) is False


def test_detect_breakout_trigger_true_on_breakout_bar():
    df = compute_indicators(_make_breakout_df()).dropna().reset_index(drop=True)
    assert detect_breakout_trigger(df) is True


def test_detect_breakout_trigger_false_before_breakout():
    full = compute_indicators(_make_breakout_df()).dropna().reset_index(drop=True)
    df = full.iloc[:-1].reset_index(drop=True)
    assert detect_breakout_trigger(df) is False


def test_detect_breakout_trigger_false_without_volume_surge():
    df = compute_indicators(_make_breakout_df(breakout_vol_mult=1.0)).dropna().reset_index(drop=True)
    assert detect_breakout_trigger(df) is False


def test_compute_breakout_levels_caps_stop_at_min_of_atr_and_adr():
    levels = compute_breakout_levels(entry=100.0, atr=5.0, adr_pct=0.02, r_multiple=3.0)
    # adr_pct * entry = 2.0, tighter than atr=5.0 -> stop distance should be 2.0
    assert levels["stop"] == pytest.approx(98.0)
    assert levels["target"] == pytest.approx(106.0)
    assert levels["risk_reward"] == pytest.approx(3.0)


def test_compute_breakout_levels_uses_atr_when_tighter():
    levels = compute_breakout_levels(entry=100.0, atr=1.5, adr_pct=0.05, r_multiple=3.0)
    assert levels["stop"] == pytest.approx(98.5)
    assert levels["target"] == pytest.approx(104.5)


def test_position_size_by_risk_basic():
    # $2400 account, 1% risk = $24 risk budget; $2 risk/share -> 12 shares
    shares = position_size_by_risk(equity=2400, risk_pct=0.01, entry=100.0, stop=98.0)
    assert shares == 12


def test_position_size_by_risk_zero_when_stop_above_entry():
    shares = position_size_by_risk(equity=2400, risk_pct=0.01, entry=100.0, stop=101.0)
    assert shares == 0
