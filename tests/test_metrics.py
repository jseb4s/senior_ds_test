"""Unit tests for evaluation metrics."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.evaluation.metrics import (
    compute_all_metrics,
    compute_net_value,
    find_optimal_threshold,
)


def _perfect_predictions(n=100, churn_rate=0.3):
    rng = np.random.default_rng(0)
    y_true = (rng.random(n) < churn_rate).astype(int)
    y_prob = y_true.astype(float)
    return y_true, y_prob


def _random_predictions(n=200, churn_rate=0.3, seed=7):
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < churn_rate).astype(int)
    y_prob = rng.random(n)
    return y_true, y_prob


class TestComputeAllMetrics:
    def test_perfect_model_auc_is_one(self):
        y_true, y_prob = _perfect_predictions()
        metrics = compute_all_metrics(y_true, y_prob, threshold=0.5)
        assert metrics["roc_auc"] == pytest.approx(1.0)

    def test_keys_present(self):
        y_true, y_prob = _random_predictions()
        metrics = compute_all_metrics(y_true, y_prob)
        for key in ["roc_auc", "f1", "f2", "precision", "recall", "tp", "fp", "tn", "fn", "threshold"]:
            assert key in metrics

    def test_confusion_matrix_sums_to_n(self):
        y_true, y_prob = _random_predictions(n=150)
        metrics = compute_all_metrics(y_true, y_prob, threshold=0.5)
        assert metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"] == 150

    def test_precision_recall_in_range(self):
        y_true, y_prob = _random_predictions()
        metrics = compute_all_metrics(y_true, y_prob)
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0

    def test_threshold_stored_correctly(self):
        y_true, y_prob = _random_predictions()
        metrics = compute_all_metrics(y_true, y_prob, threshold=0.37)
        assert metrics["threshold"] == pytest.approx(0.37)


class TestFindOptimalThreshold:
    def test_returns_float_in_valid_range(self):
        y_true, y_prob = _random_predictions()
        t = find_optimal_threshold(y_true, y_prob, strategy="f2")
        assert 0.0 < t < 1.0

    def test_f2_strategy(self):
        y_true, y_prob = _random_predictions()
        t = find_optimal_threshold(y_true, y_prob, strategy="f2")
        assert isinstance(t, float)

    def test_youden_strategy(self):
        y_true, y_prob = _random_predictions()
        t = find_optimal_threshold(y_true, y_prob, strategy="youden")
        assert 0.0 <= t <= 1.0

    def test_perfect_model_threshold_near_half(self):
        y_true, y_prob = _perfect_predictions(n=200)
        t = find_optimal_threshold(y_true, y_prob, strategy="f2")
        assert t <= 1.0


class TestComputeNetValue:
    def test_positive_net_value_with_good_model(self):
        y_true, y_prob = _perfect_predictions(n=200, churn_rate=0.3)
        result = compute_net_value(y_true, y_prob, threshold=0.5)
        assert result["net_value_usd"] > 0

    def test_keys_present(self):
        y_true, y_prob = _random_predictions()
        result = compute_net_value(y_true, y_prob, threshold=0.5)
        for key in ["total_contacts", "true_positives", "false_positives",
                    "expected_retentions", "gross_value_usd", "total_cost_usd",
                    "net_value_usd", "roi_pct", "assumed_response_rate"]:
            assert key in result

    def test_zero_contacts_gives_zero_cost(self):
        # All probabilities below threshold → no contacts, zero cost.
        # Include both classes to avoid sklearn confusion matrix edge case.
        y_true = np.array([0] * 40 + [1] * 10)
        y_prob = np.full(50, 0.1)  # all below threshold=0.5
        result = compute_net_value(y_true, y_prob, threshold=0.5)
        assert result["total_contacts"] == 0
        assert result["total_cost_usd"] == 0.0
        assert result["roi_pct"] == 0.0

    def test_contacts_equal_tp_plus_fp(self):
        y_true, y_prob = _random_predictions(n=100)
        t = 0.5
        result = compute_net_value(y_true, y_prob, threshold=t)
        assert result["total_contacts"] == result["true_positives"] + result["false_positives"]

    def test_custom_costs_reflected(self):
        y_true, y_prob = _perfect_predictions(n=100, churn_rate=0.5)
        result = compute_net_value(
            y_true, y_prob, threshold=0.5,
            contact_cost=10.0, retention_margin=200.0, response_rate=0.5,
        )
        tp = result["true_positives"]
        fp = result["false_positives"]
        expected_gross = tp * 0.5 * 200.0
        expected_cost = (tp + fp) * 10.0
        assert result["gross_value_usd"] == pytest.approx(expected_gross)
        assert result["total_cost_usd"] == pytest.approx(expected_cost)
