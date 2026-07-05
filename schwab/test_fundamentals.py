"""
Tests for fundamentals.py — earnings-derived features (earnings yield, EPS
growth, surprise momentum) as-of joined to daily bars with NO lookahead:
a report dated D becomes visible strictly AFTER D.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fundamentals as fu


def _bars(n=400, start="2023-01-02", price=100.0):
    dates = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame({"datetime": dates,
                         "close": np.full(n, price),
                         "open": price, "high": price, "low": price,
                         "volume": 1e6})


def _earnings():
    """5 quarterly reports: EPS 1.0 rising 0.1/quarter, surprise +5%."""
    dates = pd.to_datetime(["2023-02-01", "2023-05-01", "2023-08-01",
                            "2023-11-01", "2024-02-01"])
    return pd.DataFrame({"date": dates,
                         "eps_est": [0.95, 1.05, 1.15, 1.25, 1.35],
                         "eps":     [1.0, 1.1, 1.2, 1.3, 1.4],
                         "surprise_pct": [5.0, 4.0, 3.0, 2.0, 1.0]})


class TestBuildFeatures:
    def test_columns_and_alignment(self):
        bars = _bars()
        feats = fu.build_fundamental_features(bars, _earnings())
        assert list(feats.columns) == fu.FUND_FEATURES
        assert len(feats) == len(bars)

    def test_no_lookahead_report_visible_only_after_its_date(self):
        bars = _bars()
        feats = fu.build_fundamental_features(bars, _earnings())
        d = pd.to_datetime(bars["datetime"]).dt.date
        on_day    = feats[d == pd.Timestamp("2023-11-01").date()]
        day_after = feats[d == pd.Timestamp("2023-11-02").date()]
        # surprise from the 2023-11-01 report (2.0) must NOT show on 11-01
        assert on_day["surprise_last"].iloc[0] == pytest.approx(3.0)
        assert day_after["surprise_last"].iloc[0] == pytest.approx(2.0)

    def test_nan_before_enough_history(self):
        bars = _bars()
        feats = fu.build_fundamental_features(bars, _earnings())
        assert feats.iloc[0].isna().all()      # nothing reported yet

    def test_earnings_yield_is_ttm_eps_over_close(self):
        bars = _bars()
        feats = fu.build_fundamental_features(bars, _earnings())
        d = pd.to_datetime(bars["datetime"]).dt.date
        # after 2023-11-01 report: ttm = 1.0+1.1+1.2+1.3 = 4.6; close = 100
        row = feats[d == pd.Timestamp("2023-11-03").date()]
        assert row["earnings_yield"].iloc[0] == pytest.approx(4.6 / 100)

    def test_empty_earnings_gives_all_nan(self):
        bars = _bars()
        feats = fu.build_fundamental_features(bars, pd.DataFrame(
            columns=["date", "eps_est", "eps", "surprise_pct"]))
        assert feats.isna().all().all()


class TestLoad:
    def test_load_earnings_missing_returns_none(self, tmp_path):
        assert fu.load_earnings("ZZZ", str(tmp_path)) is None

    def test_load_earnings_roundtrip(self, tmp_path):
        _earnings().to_csv(tmp_path / "ko_earnings.csv", index=False)
        e = fu.load_earnings("KO", str(tmp_path))
        assert len(e) == 5 and "surprise_pct" in e.columns


class TestAssignmentRiskWithFundamentals:
    def test_model_trains_with_fundamentals_and_reports_auc(self):
        import assignment_risk as ar
        rng = np.random.default_rng(7)
        t = np.arange(700)
        closes = 100 * (1 + 0.08 * np.sin(t / 25) + rng.normal(0, 0.004, 700))
        bars = pd.DataFrame({
            "datetime": pd.date_range("2021-01-04", periods=700, freq="B"),
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": np.full(700, 1e6)})
        # quarterly reports starting BEFORE the bar window so ttm/growth
        # features are defined from the first bars
        qdates = pd.date_range("2018-02-01", periods=24, freq="QS")
        earnings = pd.DataFrame({
            "date": qdates,
            "eps_est": np.linspace(1.0, 2.2, 24),
            "eps": np.linspace(1.0, 2.3, 24),
            "surprise_pct": rng.normal(2, 3, 24)})
        model = ar.AssignmentRiskModel()
        model.fit(bars, earnings=earnings)
        prob = model.predict_prob(bars, earnings=earnings)
        assert 0.0 <= prob <= 1.0
        assert set(fu.FUND_FEATURES) <= set(model.feature_cols)
        # fit() stores the earnings frame so advise()/predict_prob callers
        # don't need to pass it — live scanner stays unchanged
        prob2 = model.predict_prob(bars)
        assert prob2 == pytest.approx(prob)

    def test_load_models_picks_up_cached_fundamentals(self, tmp_path):
        import assignment_risk as ar
        import fundamentals as fu
        rng = np.random.default_rng(7)
        t = np.arange(700)
        closes = 100 * (1 + 0.08 * np.sin(t / 25) + rng.normal(0, 0.004, 700))
        bars = pd.DataFrame({
            "datetime": pd.date_range("2021-01-04", periods=700, freq="B"),
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": np.full(700, 1e6)})
        bars.to_csv(tmp_path / "ko_history.csv", index=False)
        fund_dir = tmp_path / "fundamentals"
        fund_dir.mkdir()
        qdates = pd.date_range("2018-02-01", periods=24, freq="QS")
        pd.DataFrame({"date": qdates,
                      "eps_est": np.linspace(1.0, 2.2, 24),
                      "eps": np.linspace(1.0, 2.3, 24),
                      "surprise_pct": rng.normal(2, 3, 24)}
                     ).to_csv(fund_dir / "ko_earnings.csv", index=False)
        models = ar.load_models(["KO"], data_dir=str(tmp_path))
        assert set(fu.FUND_FEATURES) <= set(models["KO"].feature_cols)
        assert ar.advise(models, "KO", bars) is not None
