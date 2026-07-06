"""
Tests for backtest_maggie.py

Run: /opt/miniconda3/envs/trader/bin/python -m pytest schwab/test_backtest_maggie.py -v
"""
import sys, os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_vix(n, level):
    return pd.DataFrame({
        "datetime": pd.date_range("2022-06-01", periods=n, freq="B"),
        "open": np.full(n, level), "high": np.full(n, level),
        "low":  np.full(n, level), "close": np.full(n, level),
        "volume": np.zeros(n),
    })


def _make_spy(n, trend="up"):
    close = np.linspace(380, 500, n) if trend == "up" else np.linspace(500, 350, n)
    return pd.DataFrame({
        "datetime": pd.date_range("2022-06-01", periods=n, freq="B"),
        "open":  close * 0.999, "high": close * 1.004,
        "low":   close * 0.996, "close": close,
        "volume": np.ones(n) * 1e8,
    })


def _make_breakout_then_reversal_df(prehistory=60, runup=80, consolidation=15,
                                     continuation=40, decline=30):
    """Run-up -> tight consolidation -> volume breakout -> continued uptrend
    (clears the R-multiple target, starts trailing) -> reversal (breaks the
    EMA10 trail) — a full round-trip Maggie trade."""
    close, high, low, volume = [], [], [], []

    for _ in range(prehistory):
        close.append(100.0); high.append(101.0); low.append(99.0); volume.append(5_000_000)

    base_start, base_end = 100.0, 100.0 * 1.35
    for i in range(runup):
        frac = i / (runup - 1)
        c = base_start * (base_end / base_start) ** frac
        close.append(c); high.append(c * 1.02); low.append(c * 0.98); volume.append(5_000_000)

    for i in range(consolidation):
        frac = i / (consolidation - 1)
        rng = 0.03 - 0.02 * frac
        center = base_end * (1 + 0.003 * frac)
        close.append(center); high.append(center * (1 + rng / 2))
        low.append(center * (1 - rng / 2)); volume.append(4_000_000)

    last_center = close[-1]
    close.append(last_center * 1.002); high.append(last_center * 1.008)
    low.append(last_center * 0.996); volume.append(4_200_000)

    breakout_close = close[-1] * 1.08
    avg_recent_vol = sum(volume[-14:]) / 14
    close.append(breakout_close); high.append(breakout_close * 1.01)
    low.append(close[-2] * 1.0); volume.append(avg_recent_vol * 3.0)

    cont_start = close[-1]
    cont_end = cont_start * 1.30
    for i in range(continuation):
        frac = i / (continuation - 1)
        c = cont_start * (cont_end / cont_start) ** frac
        close.append(c); high.append(c * 1.02); low.append(c * 0.98); volume.append(4_000_000)

    decl_start = close[-1]
    decl_end = decl_start * 0.75
    for i in range(decline):
        frac = i / (decline - 1)
        c = decl_start * (decl_end / decl_start) ** frac
        close.append(c); high.append(c * 1.02); low.append(c * 0.98); volume.append(4_000_000)

    n = len(close)
    return pd.DataFrame({
        "datetime": pd.date_range("2022-06-01", periods=n, freq="B"),
        "open": low, "high": high, "low": low, "close": close, "volume": volume,
    })


class TestWalkForwardMaggie:
    def test_returns_list(self):
        from backtest_maggie import walk_forward_maggie
        df = _make_breakout_then_reversal_df()
        assert isinstance(walk_forward_maggie(df, "TEST"), list)

    def test_no_trades_in_nuked_zone(self):
        from backtest_maggie import walk_forward_maggie
        df  = _make_breakout_then_reversal_df()
        n   = len(df)
        vix = _make_vix(n, 40.0)
        spy = _make_spy(n, "up")
        events = walk_forward_maggie(df, "TEST", spy_df=spy, vix_df=vix)
        assert len(events) == 0

    def test_no_trades_outside_reclamation(self):
        """SPY downtrend -> WASTELAND -> Maggie is benched (RECLAMATION only)."""
        from backtest_maggie import walk_forward_maggie
        df  = _make_breakout_then_reversal_df()
        n   = len(df)
        vix = _make_vix(n, 15.0)
        spy = _make_spy(n, "down")
        events = walk_forward_maggie(df, "TEST", spy_df=spy, vix_df=vix)
        assert len(events) == 0

    def test_takes_the_breakout_trade_in_reclamation(self):
        from backtest_maggie import walk_forward_maggie
        df  = _make_breakout_then_reversal_df()
        n   = len(df)
        vix = _make_vix(n, 15.0)
        spy = _make_spy(n, "up")
        events = walk_forward_maggie(df, "TEST", spy_df=spy, vix_df=vix)
        assert len(events) >= 1

    def test_event_keys_present(self):
        from backtest_maggie import walk_forward_maggie
        df  = _make_breakout_then_reversal_df()
        n   = len(df)
        spy = _make_spy(n, "up")
        vix = _make_vix(n, 15.0)
        events = walk_forward_maggie(df, "TEST", spy_df=spy, vix_df=vix)
        assert len(events) >= 1
        for e in events:
            for k in ("symbol", "event", "pnl", "entry_price",
                      "exit_price", "entry_date", "exit_date"):
                assert k in e, f"Missing key: {k}"

    def test_exit_event_types_valid(self):
        from backtest_maggie import walk_forward_maggie
        VALID = {"maggie_stop_hit", "maggie_trend_end", "maggie_max_hold"}
        df  = _make_breakout_then_reversal_df()
        n   = len(df)
        spy = _make_spy(n, "up")
        vix = _make_vix(n, 15.0)
        events = walk_forward_maggie(df, "TEST", spy_df=spy, vix_df=vix)
        assert len(events) >= 1
        for e in events:
            assert e["event"] in VALID

    def test_pnl_matches_price_diff(self):
        from backtest_maggie import walk_forward_maggie
        df  = _make_breakout_then_reversal_df()
        n   = len(df)
        spy = _make_spy(n, "up")
        vix = _make_vix(n, 15.0)
        events = walk_forward_maggie(df, "TEST", spy_df=spy, vix_df=vix)
        for e in events:
            expected = round((e["exit_price"] - e["entry_price"]) * 100, 2)
            assert abs(e["pnl"] - expected) < 0.01

    def test_symbol_recorded(self):
        from backtest_maggie import walk_forward_maggie
        df     = _make_breakout_then_reversal_df()
        n      = len(df)
        spy    = _make_spy(n, "up")
        vix    = _make_vix(n, 15.0)
        events = walk_forward_maggie(df, "NVDA", spy_df=spy, vix_df=vix)
        for e in events:
            assert e["symbol"] == "NVDA"

    def test_winning_trade_reaches_breakeven_or_better_stop(self):
        """A trade that clears the first target should never lose more than
        a whisker below breakeven — the stop was moved up."""
        from backtest_maggie import walk_forward_maggie
        df  = _make_breakout_then_reversal_df()
        n   = len(df)
        spy = _make_spy(n, "up")
        vix = _make_vix(n, 15.0)
        events = walk_forward_maggie(df, "TEST", spy_df=spy, vix_df=vix)
        assert len(events) >= 1
        assert events[0]["pnl"] > 0
