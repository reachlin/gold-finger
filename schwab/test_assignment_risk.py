"""
Tests for assignment_risk.py — LGBM advisory: P(price falls below 95% of
today's close within the next 30 trading days).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import assignment_risk as ar


# ---------------------------------------------------------------------------
# Synthetic price data
# ---------------------------------------------------------------------------

def _ohlcv(closes: np.ndarray) -> pd.DataFrame:
    n = len(closes)
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=n, freq="B"),
        "open":   closes * (1 + rng.normal(0, 0.002, n)),
        "high":   closes * 1.01,
        "low":    closes * 0.99,
        "close":  closes,
        "volume": rng.uniform(1e6, 2e6, n),
    })


def _wavy(n=600, base=100.0, seed=7) -> pd.DataFrame:
    """Sideways series with regular >5% dips — both label classes present."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    closes = base * (1 + 0.08 * np.sin(t / 25) + rng.normal(0, 0.004, n))
    return _ohlcv(closes)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

class TestLabels:
    def test_crash_within_horizon_is_positive(self):
        closes = np.concatenate([np.full(50, 100.0), np.full(40, 80.0)])
        labels = ar.make_labels(pd.Series(closes), horizon=30, drop_pct=0.05)
        assert labels[40] == 1        # crash at bar 50 is within 30 bars

    def test_steady_price_is_negative(self):
        closes = np.full(100, 100.0)
        labels = ar.make_labels(pd.Series(closes), horizon=30, drop_pct=0.05)
        assert labels[:70].sum() == 0

    def test_small_dip_below_threshold_is_negative(self):
        closes = np.concatenate([np.full(50, 100.0), np.full(40, 97.0)])
        labels = ar.make_labels(pd.Series(closes), horizon=30, drop_pct=0.05)
        assert labels[40] == 0        # -3% dip doesn't cross the 5% line

    def test_tail_without_full_horizon_is_nan(self):
        closes = np.full(100, 100.0)
        labels = ar.make_labels(pd.Series(closes), horizon=30, drop_pct=0.05)
        assert np.isnan(labels[-1])


class TestUpLabels:
    """Call-side mirror: did the close rise above (1+rise_pct) within horizon?"""

    def test_rally_within_horizon_is_positive(self):
        closes = np.concatenate([np.full(50, 100.0), np.full(40, 112.0)])
        labels = ar.make_labels_up(pd.Series(closes), horizon=30, rise_pct=0.08)
        assert labels[40] == 1        # rally at bar 50 is within 30 bars

    def test_steady_price_is_negative(self):
        closes = np.full(100, 100.0)
        labels = ar.make_labels_up(pd.Series(closes), horizon=30, rise_pct=0.08)
        assert labels[:70].sum() == 0

    def test_small_rise_below_threshold_is_negative(self):
        closes = np.concatenate([np.full(50, 100.0), np.full(40, 105.0)])
        labels = ar.make_labels_up(pd.Series(closes), horizon=30, rise_pct=0.08)
        assert labels[40] == 0        # +5% doesn't cross the 8% line

    def test_tail_without_full_horizon_is_nan(self):
        closes = np.full(100, 100.0)
        labels = ar.make_labels_up(pd.Series(closes), horizon=30, rise_pct=0.08)
        assert np.isnan(labels[-1])


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_feature_columns_and_no_nan(self):
        df = _wavy()
        feats = ar.make_features(df)
        assert list(feats.columns) == ar.FEATURES
        assert not feats.iloc[60:].isna().any().any()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TestModel:
    def test_fit_and_predict_prob_in_range(self):
        model = ar.AssignmentRiskModel()
        model.fit(_wavy())
        prob = model.predict_prob(_wavy(seed=8))
        assert 0.0 <= prob <= 1.0

    def test_insufficient_data_raises(self):
        model = ar.AssignmentRiskModel()
        with pytest.raises(ValueError):
            model.fit(_wavy(n=80))

    def test_holdout_auc_recorded(self):
        model = ar.AssignmentRiskModel()
        model.fit(_wavy())
        assert model.holdout_auc is None or 0.0 <= model.holdout_auc <= 1.0

    def test_up_direction_fit_and_predict(self):
        model = ar.AssignmentRiskModel(direction="up")
        model.fit(_wavy())
        prob = model.predict_prob(_wavy(seed=8))
        assert 0.0 <= prob <= 1.0

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            ar.AssignmentRiskModel(direction="sideways")

    def test_probabilities_are_calibrated(self):
        """With enough data the final model is an isotonic-calibrated wrapper."""
        from sklearn.calibration import CalibratedClassifierCV
        model = ar.AssignmentRiskModel()
        model.fit(_wavy())
        assert model.calibrated is True
        assert isinstance(model.model, CalibratedClassifierCV)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class TestLoadModels:
    def test_trains_from_csv_dir(self, tmp_path):
        _wavy().to_csv(tmp_path / "ko_history.csv", index=False)
        models = ar.load_models(["KO", "MISSING"], data_dir=str(tmp_path))
        assert "KO" in models
        assert "MISSING" not in models

    def test_advise_returns_pct(self, tmp_path):
        _wavy().to_csv(tmp_path / "ko_history.csv", index=False)
        models = ar.load_models(["KO"], data_dir=str(tmp_path))
        pct = ar.advise(models, "KO", _wavy(seed=9))
        assert pct is not None
        assert 0.0 <= pct <= 100.0

    def test_advise_unknown_symbol_returns_none(self):
        assert ar.advise({}, "ZZZ", _wavy()) is None

    def test_load_models_up_direction(self, tmp_path):
        _wavy().to_csv(tmp_path / "ko_history.csv", index=False)
        models = ar.load_models(["KO"], data_dir=str(tmp_path), direction="up")
        assert "KO" in models
        pct = ar.advise(models, "KO", _wavy(seed=9))
        assert pct is not None and 0.0 <= pct <= 100.0

    def test_model_auc_helper(self, tmp_path):
        _wavy().to_csv(tmp_path / "ko_history.csv", index=False)
        models = ar.load_models(["KO"], data_dir=str(tmp_path))
        auc = ar.model_auc(models, "KO")
        assert auc is None or 0.0 <= auc <= 1.0
        assert ar.model_auc(models, "ZZZ") is None
        assert ar.model_auc({}, "KO") is None
