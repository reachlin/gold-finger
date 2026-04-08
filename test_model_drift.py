"""Tests for model drift detection and evaluation logic."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from predict import evaluate_signals, DRIFT_THRESHOLD


# ---------------------------------------------------------------------------
# evaluate_signals tests
# ---------------------------------------------------------------------------
class TestEvaluateSignals:
    def test_buy_correct_when_positive_return(self):
        signals = ["BUY"]
        returns = [0.02]
        acc = evaluate_signals(signals, returns)
        assert acc == 1.0

    def test_buy_wrong_when_negative_return(self):
        signals = ["BUY"]
        returns = [-0.02]
        acc = evaluate_signals(signals, returns)
        assert acc == 0.0

    def test_sell_correct_when_negative_return(self):
        signals = ["SELL"]
        returns = [-0.01]
        acc = evaluate_signals(signals, returns)
        assert acc == 1.0

    def test_sell_wrong_when_positive_return(self):
        signals = ["SELL"]
        returns = [0.01]
        acc = evaluate_signals(signals, returns)
        assert acc == 0.0

    def test_hold_always_correct(self):
        signals = ["HOLD", "HOLD", "HOLD"]
        returns = [0.01, -0.01, 0.0]
        acc = evaluate_signals(signals, returns)
        assert acc == 1.0

    def test_mixed_signals(self):
        signals = ["BUY", "SELL", "HOLD", "BUY"]
        returns = [0.01, -0.01, 0.05, -0.02]
        # BUY+pos=correct, SELL+neg=correct, HOLD=correct, BUY+neg=wrong
        acc = evaluate_signals(signals, returns)
        assert acc == pytest.approx(0.75)

    def test_empty_signals(self):
        acc = evaluate_signals([], [])
        assert acc == 0.0

    def test_zero_return_counts_as_correct_for_hold(self):
        signals = ["HOLD"]
        returns = [0.0]
        acc = evaluate_signals(signals, returns)
        assert acc == 1.0

    def test_zero_return_counts_as_wrong_for_buy(self):
        # BUY expects positive return; zero is not positive
        signals = ["BUY"]
        returns = [0.0]
        acc = evaluate_signals(signals, returns)
        assert acc == 0.0


# ---------------------------------------------------------------------------
# Rolling accuracy tests
# ---------------------------------------------------------------------------
class TestRollingAccuracy:
    def test_rolling_window(self):
        from predict import rolling_accuracy
        # 10 correct then 10 wrong
        signals = ["BUY"] * 10 + ["BUY"] * 10
        returns = [0.01] * 10 + [-0.01] * 10
        roll = rolling_accuracy(signals, returns, window=10)
        # Last 10 are all wrong
        assert roll == 0.0

    def test_rolling_all_correct(self):
        from predict import rolling_accuracy
        signals = ["SELL"] * 20
        returns = [-0.01] * 20
        roll = rolling_accuracy(signals, returns, window=20)
        assert roll == 1.0

    def test_rolling_window_larger_than_data(self):
        from predict import rolling_accuracy
        signals = ["BUY", "SELL"]
        returns = [0.01, -0.01]
        # Window > len: use all data
        roll = rolling_accuracy(signals, returns, window=50)
        assert roll == 1.0


# ---------------------------------------------------------------------------
# Drift threshold tests
# ---------------------------------------------------------------------------
class TestDriftDetection:
    def test_below_threshold_needs_retrain(self):
        from predict import needs_retrain
        assert needs_retrain(0.40) is True
        assert needs_retrain(0.44) is True

    def test_above_threshold_ok(self):
        from predict import needs_retrain
        assert needs_retrain(0.50) is False
        assert needs_retrain(0.80) is False

    def test_at_threshold_ok(self):
        from predict import needs_retrain
        assert needs_retrain(DRIFT_THRESHOLD) is False

    def test_threshold_value(self):
        assert DRIFT_THRESHOLD == 0.45
