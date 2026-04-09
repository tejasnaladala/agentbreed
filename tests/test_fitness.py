"""Tests for breed.fitness evaluators and calibration utility."""

from __future__ import annotations

import pytest

from breed.fitness import (
    BrierScoreEvaluator,
    CompositeFitness,
    PassRateEvaluator,
    calibration_error,
)


# ── BrierScoreEvaluator ─────────────────────────────────────────────────────


class TestBrierScoreEvaluator:
    def setup_method(self) -> None:
        self.evaluator = BrierScoreEvaluator()

    def test_perfect_predictions_score_one(self) -> None:
        predictions = [1.0, 0.0, 1.0, 0.0]
        outcomes = [1.0, 0.0, 1.0, 0.0]
        assert self.evaluator.evaluate(predictions, outcomes) == pytest.approx(1.0)

    def test_worst_predictions_score_zero(self) -> None:
        predictions = [1.0, 0.0, 1.0, 0.0]
        outcomes = [0.0, 1.0, 0.0, 1.0]
        assert self.evaluator.evaluate(predictions, outcomes) == pytest.approx(0.0)

    def test_random_predictions_between_zero_and_one(self) -> None:
        predictions = [0.5, 0.5, 0.5, 0.5]
        outcomes = [1.0, 0.0, 1.0, 0.0]
        score = self.evaluator.evaluate(predictions, outcomes)
        assert 0.0 < score < 1.0
        # Brier = mean((0.5-1)^2 + (0.5-0)^2 + ...) = 0.25, so 1-0.25 = 0.75
        assert score == pytest.approx(0.75)

    def test_single_prediction(self) -> None:
        assert self.evaluator.evaluate([0.8], [1.0]) == pytest.approx(1.0 - 0.04)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            self.evaluator.evaluate([0.5, 0.5], [1.0])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            self.evaluator.evaluate([], [])


# ── PassRateEvaluator ────────────────────────────────────────────────────────


class TestPassRateEvaluator:
    def setup_method(self) -> None:
        self.evaluator = PassRateEvaluator()

    def test_all_pass(self) -> None:
        predictions = [1.0, 0.8, 0.5, 1.0]
        assert self.evaluator.evaluate(predictions, []) == pytest.approx(1.0)

    def test_none_pass(self) -> None:
        predictions = [0.0, 0.1, 0.2, 0.49]
        assert self.evaluator.evaluate(predictions, []) == pytest.approx(0.0)

    def test_half_pass(self) -> None:
        predictions = [1.0, 0.0, 0.8, 0.2]
        assert self.evaluator.evaluate(predictions, []) == pytest.approx(0.5)

    def test_threshold_exactly_half(self) -> None:
        # 0.5 counts as passed
        assert self.evaluator.evaluate([0.5], []) == pytest.approx(1.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            self.evaluator.evaluate([], [])


# ── CompositeFitness ─────────────────────────────────────────────────────────


class TestCompositeFitness:
    def test_weighted_combination(self) -> None:
        comp = CompositeFitness(
            metrics={
                "accuracy": (BrierScoreEvaluator(), 0.6),
                "pass_rate": (PassRateEvaluator(), 0.4),
            }
        )

        # Perfect accuracy predictions, half pass-rate
        metric_inputs = {
            "accuracy": ([1.0, 0.0], [1.0, 0.0]),
            "pass_rate": ([1.0, 0.0], []),
        }

        composite, breakdown = comp.evaluate(metric_inputs)
        assert breakdown["accuracy"] == pytest.approx(1.0)
        assert breakdown["pass_rate"] == pytest.approx(0.5)
        # Weighted: (1.0 * 0.6 + 0.5 * 0.4) / (0.6 + 0.4) = 0.8
        assert composite == pytest.approx(0.8)

    def test_equal_weights(self) -> None:
        comp = CompositeFitness(
            metrics={
                "a": (BrierScoreEvaluator(), 1.0),
                "b": (BrierScoreEvaluator(), 1.0),
            }
        )
        metric_inputs = {
            "a": ([1.0], [1.0]),  # score = 1.0
            "b": ([0.0], [1.0]),  # score = 0.0
        }
        composite, breakdown = comp.evaluate(metric_inputs)
        assert composite == pytest.approx(0.5)

    def test_missing_metric_input_raises(self) -> None:
        comp = CompositeFitness(
            metrics={"accuracy": (BrierScoreEvaluator(), 1.0)}
        )
        with pytest.raises(KeyError, match="accuracy"):
            comp.evaluate({})

    def test_no_metrics_raises(self) -> None:
        comp = CompositeFitness()
        with pytest.raises(ValueError, match="no metrics"):
            comp.evaluate({})

    def test_weights_are_normalised(self) -> None:
        comp = CompositeFitness(
            metrics={
                "a": (BrierScoreEvaluator(), 3.0),
                "b": (BrierScoreEvaluator(), 7.0),
            }
        )
        metric_inputs = {
            "a": ([1.0], [1.0]),
            "b": ([1.0], [1.0]),
        }
        composite, _ = comp.evaluate(metric_inputs)
        # Both perfect -> composite should be 1.0 regardless of weight ratio
        assert composite == pytest.approx(1.0)


# ── calibration_error ────────────────────────────────────────────────────────


class TestCalibrationError:
    def test_perfectly_calibrated(self) -> None:
        # All predictions = 0.5, half the outcomes are 1
        predictions = [0.5] * 100
        outcomes = [1.0] * 50 + [0.0] * 50
        ece = calibration_error(predictions, outcomes, num_bins=10)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_poorly_calibrated(self) -> None:
        # Predict 0.9 but outcome is always 0
        predictions = [0.9] * 100
        outcomes = [0.0] * 100
        ece = calibration_error(predictions, outcomes, num_bins=10)
        assert ece == pytest.approx(0.9, abs=0.01)

    def test_moderate_calibration(self) -> None:
        predictions = [0.2] * 50 + [0.8] * 50
        outcomes = [0.0] * 40 + [1.0] * 10 + [1.0] * 40 + [0.0] * 10
        ece = calibration_error(predictions, outcomes, num_bins=10)
        # Should be relatively low -- both bins close to their avg outcomes
        assert 0.0 <= ece < 0.5

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            calibration_error([0.5], [1.0, 0.0])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            calibration_error([], [])

    def test_single_bin(self) -> None:
        ece = calibration_error([0.7, 0.7], [1.0, 0.0], num_bins=1)
        # avg_pred = 0.7, avg_out = 0.5, ECE = |0.7 - 0.5| = 0.2
        assert ece == pytest.approx(0.2)

    def test_invalid_num_bins_raises(self) -> None:
        with pytest.raises(ValueError, match="num_bins"):
            calibration_error([0.5], [1.0], num_bins=0)
