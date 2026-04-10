"""Tests for ``breed.stats``.

Covers calibration metrics, paired/non-parametric significance tests,
bootstrap confidence intervals, effect sizes, Holm-Bonferroni correction, and
seed aggregation, including edge cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from breed.stats import (
    BootstrapCI,
    TestResult,
    aggregate_seed_runs,
    bootstrap_ci,
    bootstrap_ci_difference,
    brier_score,
    calibration_error,
    cohens_d,
    cohens_d_paired,
    holm_bonferroni,
    paired_t_test,
    reliability_diagram_data,
    wilcoxon_signed_rank,
)


# ---------------------------------------------------------------------------
# brier_score
# ---------------------------------------------------------------------------


class TestBrierScore:
    def test_perfect_predictions_zero(self):
        preds = [0.0, 1.0, 0.0, 1.0]
        outs = [0.0, 1.0, 0.0, 1.0]
        assert brier_score(preds, outs) == pytest.approx(0.0)

    def test_worst_predictions_one(self):
        preds = [1.0, 0.0, 1.0, 0.0]
        outs = [0.0, 1.0, 0.0, 1.0]
        assert brier_score(preds, outs) == pytest.approx(1.0)

    def test_constant_half_predictions(self):
        # For p=0.5 on any binary outcomes the per-sample error is 0.25.
        preds = [0.5] * 100
        rng = np.random.default_rng(42)
        outs = rng.integers(0, 2, size=100).astype(float)
        assert brier_score(preds, outs) == pytest.approx(0.25)

    def test_random_predictions_mid(self):
        rng = np.random.default_rng(7)
        preds = rng.uniform(0, 1, size=1000)
        outs = rng.integers(0, 2, size=1000).astype(float)
        val = brier_score(preds.tolist(), outs.tolist())
        # For independent uniform predictions and fair binary outcomes,
        # expected Brier is 1/3. Allow generous slack for 1000 samples.
        assert 0.25 < val < 0.45

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            brier_score([0.5, 0.5], [1.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            brier_score([], [])

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            brier_score([0.5, float("nan")], [1.0, 0.0])

    def test_prediction_out_of_range_raises(self):
        with pytest.raises(ValueError):
            brier_score([1.5, 0.5], [1.0, 0.0])

    def test_non_binary_outcome_raises(self):
        with pytest.raises(ValueError):
            brier_score([0.5, 0.5], [0.3, 0.7])


# ---------------------------------------------------------------------------
# calibration_error
# ---------------------------------------------------------------------------


class TestCalibrationError:
    def test_perfectly_calibrated(self):
        # When predictions exactly match outcomes, ECE is 0.
        preds = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        outs = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        assert calibration_error(preds, outs, n_bins=5) == pytest.approx(0.0)

    def test_overconfident_high_ece(self):
        # Predicting 1.0 confidently but only 20% of outcomes are 1 -> ECE 0.8.
        preds = [1.0] * 10
        outs = [1.0, 1.0] + [0.0] * 8
        ece = calibration_error(preds, outs, n_bins=10)
        assert ece == pytest.approx(0.8, abs=1e-9)

    def test_well_calibrated_lower_than_overconfident(self):
        rng = np.random.default_rng(0)
        # Well-calibrated: draw p~U(0,1) then y~Bernoulli(p)
        p_calib = rng.uniform(0, 1, size=500)
        y_calib = (rng.uniform(0, 1, size=500) < p_calib).astype(float)
        # Overconfident: same p but flip in favor of 1 regardless of p
        p_over = np.full(500, 0.95)
        y_over = (rng.uniform(0, 1, size=500) < 0.5).astype(float)
        ece_calib = calibration_error(p_calib.tolist(), y_calib.tolist(), n_bins=10)
        ece_over = calibration_error(p_over.tolist(), y_over.tolist(), n_bins=10)
        assert ece_calib < ece_over

    def test_invalid_n_bins(self):
        with pytest.raises(ValueError):
            calibration_error([0.5], [1.0], n_bins=0)


# ---------------------------------------------------------------------------
# reliability_diagram_data
# ---------------------------------------------------------------------------


class TestReliabilityDiagramData:
    def test_shapes(self):
        preds = [0.1, 0.4, 0.6, 0.9]
        outs = [0.0, 0.0, 1.0, 1.0]
        centers, accs, counts = reliability_diagram_data(preds, outs, n_bins=5)
        assert centers.shape == (5,)
        assert accs.shape == (5,)
        assert counts.shape == (5,)
        assert counts.sum() == 4

    def test_centers_and_counts(self):
        preds = [0.05, 0.15, 0.45, 0.75, 0.95]
        outs = [0.0, 1.0, 0.0, 1.0, 1.0]
        centers, accs, counts = reliability_diagram_data(preds, outs, n_bins=10)
        expected_centers = np.linspace(0.05, 0.95, 10)
        np.testing.assert_allclose(centers, expected_centers)
        assert counts.sum() == 5
        # Bin with one sample and outcome 1 should have accuracy 1.0.
        assert accs[1] == pytest.approx(1.0)  # 0.15 -> bin 1
        assert accs[9] == pytest.approx(1.0)  # 0.95 -> bin 9

    def test_empty_bins_nan(self):
        preds = [0.05, 0.95]
        outs = [0.0, 1.0]
        _, accs, counts = reliability_diagram_data(preds, outs, n_bins=5)
        assert counts[0] == 1
        assert counts[-1] == 1
        # Middle bins have no data
        assert np.isnan(accs[2])
        assert counts[2] == 0


# ---------------------------------------------------------------------------
# paired_t_test
# ---------------------------------------------------------------------------


class TestPairedTTest:
    def test_significant_difference(self):
        rng = np.random.default_rng(1)
        n = 40
        base = rng.normal(0.5, 0.1, size=n)
        a = base + 0.1  # Clearly better
        b = base
        result = paired_t_test(a.tolist(), b.tolist())
        assert isinstance(result, TestResult)
        assert result.method == "paired_t"
        assert result.n == n
        assert result.p_value < 0.01
        assert result.significant is True
        assert result.effect_size is not None and result.effect_size > 0

    def test_no_difference(self):
        rng = np.random.default_rng(2)
        a = rng.normal(0.5, 0.1, size=30)
        b = a.copy()  # Identical
        result = paired_t_test(a.tolist(), b.tolist())
        # Identical -> zero diff -> our fast path returns p=1, d=0
        assert result.p_value == pytest.approx(1.0)
        assert result.significant is False
        assert result.effect_size == pytest.approx(0.0)

    def test_noisy_no_difference(self):
        rng = np.random.default_rng(3)
        a = rng.normal(0.5, 0.1, size=100)
        b = rng.normal(0.5, 0.1, size=100)
        result = paired_t_test(a.tolist(), b.tolist())
        # With identical distributions p should be large on average.
        # Not strictly guaranteed, but this seed is stable.
        assert 0.0 <= result.p_value <= 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            paired_t_test([1.0, 2.0], [1.0])

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError):
            paired_t_test([1.0], [2.0])

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            paired_t_test([1.0, 2.0], [1.0, 2.0], alpha=1.5)

    def test_str_format(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [0.5, 1.5, 2.5, 3.5, 4.5]
        result = paired_t_test(a, b)
        s = str(result)
        assert "paired_t" in s
        assert "t=" in s
        assert "p=" in s
        assert "n=5" in s


# ---------------------------------------------------------------------------
# wilcoxon_signed_rank
# ---------------------------------------------------------------------------


class TestWilcoxonSignedRank:
    def test_significant_difference(self):
        rng = np.random.default_rng(10)
        n = 30
        base = rng.normal(0.0, 1.0, size=n)
        a = base + 0.8
        b = base
        result = wilcoxon_signed_rank(a.tolist(), b.tolist())
        assert result.method == "wilcoxon"
        assert result.n == n
        assert result.p_value < 0.01
        assert result.significant is True
        assert result.effect_size is not None

    def test_identical_samples(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = list(a)
        result = wilcoxon_signed_rank(a, b)
        assert result.p_value == pytest.approx(1.0)
        assert result.effect_size == pytest.approx(0.0)
        assert result.significant is False

    def test_symmetric_no_effect(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [2.0, 1.0, 4.0, 3.0]
        result = wilcoxon_signed_rank(a, b)
        assert 0.0 <= result.p_value <= 1.0
        assert -1.0 <= result.effect_size <= 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            wilcoxon_signed_rank([1.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_known_distribution_coverage(self):
        # Bootstrap CI for the mean of a large sample should contain the
        # empirical mean exactly at the point estimate.
        rng = np.random.default_rng(42)
        samples = rng.normal(5.0, 1.0, size=200)
        ci = bootstrap_ci(samples.tolist(), n_bootstrap=2000, seed=123)
        assert isinstance(ci, BootstrapCI)
        assert ci.method == "percentile"
        assert ci.confidence == 0.95
        assert ci.n_bootstrap == 2000
        assert ci.lower < ci.point_estimate < ci.upper
        assert abs(ci.point_estimate - samples.mean()) < 1e-9
        # True mean should lie inside wide enough CI.
        assert ci.lower < 5.0 < ci.upper

    def test_median_statistic(self):
        samples = list(range(1, 11))  # median = 5.5
        ci = bootstrap_ci(
            samples,
            statistic=np.median,
            n_bootstrap=500,
            seed=0,
        )
        assert ci.lower <= ci.point_estimate <= ci.upper
        assert ci.point_estimate == pytest.approx(5.5)

    def test_reproducible_with_seed(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci1 = bootstrap_ci(samples, n_bootstrap=200, seed=7)
        ci2 = bootstrap_ci(samples, n_bootstrap=200, seed=7)
        assert ci1.lower == ci2.lower
        assert ci1.upper == ci2.upper

    def test_invalid_n_bootstrap(self):
        with pytest.raises(ValueError):
            bootstrap_ci([1.0, 2.0], n_bootstrap=0)

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            bootstrap_ci([1.0, 2.0], confidence=1.5)


# ---------------------------------------------------------------------------
# bootstrap_ci_difference
# ---------------------------------------------------------------------------


class TestBootstrapCIDifference:
    def test_zero_difference(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = bootstrap_ci_difference(samples, samples, n_bootstrap=500, seed=1)
        assert ci.point_estimate == pytest.approx(0.0)
        assert ci.lower == pytest.approx(0.0)
        assert ci.upper == pytest.approx(0.0)

    def test_large_positive_difference(self):
        rng = np.random.default_rng(11)
        n = 50
        base = rng.normal(0, 0.05, size=n)
        a = base + 1.0
        b = base
        ci = bootstrap_ci_difference(a.tolist(), b.tolist(), n_bootstrap=500, seed=2)
        assert ci.point_estimate == pytest.approx(1.0, abs=0.1)
        assert ci.lower > 0
        assert ci.upper > 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci_difference([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------


class TestCohensD:
    def test_large_effect(self):
        rng = np.random.default_rng(100)
        a = rng.normal(5.0, 1.0, size=500)
        b = rng.normal(3.0, 1.0, size=500)
        d = cohens_d(a.tolist(), b.tolist())
        # True effect is (5-3)/1 = 2.0; allow 0.4 slack for sampling noise.
        assert d == pytest.approx(2.0, abs=0.4)
        assert d > 1.5  # definitely "large" in Cohen's classification

    def test_no_effect(self):
        rng = np.random.default_rng(101)
        a = rng.normal(0.0, 1.0, size=200)
        b = rng.normal(0.0, 1.0, size=200)
        d = cohens_d(a.tolist(), b.tolist())
        assert abs(d) < 0.5

    def test_identical_samples_zero(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [1.0, 2.0, 3.0, 4.0]
        d = cohens_d(a, b)
        assert d == pytest.approx(0.0)

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError):
            cohens_d([1.0], [1.0, 2.0])

    def test_zero_pooled_sd_nonzero_mean_raises(self):
        a = [1.0, 1.0, 1.0]
        b = [2.0, 2.0, 2.0]
        with pytest.raises(ValueError):
            cohens_d(a, b)


# ---------------------------------------------------------------------------
# cohens_d_paired
# ---------------------------------------------------------------------------


class TestCohensDPaired:
    def test_known_value(self):
        # diffs = [1, 1, 1, 1] -> mean 1, sd 0 -> special case (mean!=0) raises.
        # Use varying diffs: [1, 2, 1, 2] -> mean 1.5, sd ~0.577, d_z ~ 2.598
        a = [2.0, 3.0, 2.0, 3.0]
        b = [1.0, 1.0, 1.0, 1.0]
        d = cohens_d_paired(a, b)
        diffs = np.array([1.0, 2.0, 1.0, 2.0])
        expected = diffs.mean() / diffs.std(ddof=1)
        assert d == pytest.approx(expected)

    def test_identical_samples(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [1.0, 2.0, 3.0, 4.0]
        assert cohens_d_paired(a, b) == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cohens_d_paired([1.0, 2.0], [1.0])

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            cohens_d_paired([1.0], [2.0])


# ---------------------------------------------------------------------------
# holm_bonferroni
# ---------------------------------------------------------------------------


class TestHolmBonferroni:
    def test_textbook_case(self):
        # Classic Holm example: m=4 tests.
        p_values = [0.01, 0.04, 0.03, 0.005]
        flags = holm_bonferroni(p_values, alpha=0.05)
        # Sorted: 0.005, 0.01, 0.03, 0.04
        # Thresholds: 0.0125, 0.0167, 0.025, 0.05
        # Reject: 0.005 (<0.0125), 0.01 (<0.0167), 0.03 (>0.025) stop.
        # So only two smallest are rejected.
        assert flags[3] is True   # 0.005
        assert flags[0] is True   # 0.01
        assert flags[2] is False  # 0.03
        assert flags[1] is False  # 0.04

    def test_all_significant(self):
        p_values = [0.001, 0.002, 0.003]
        flags = holm_bonferroni(p_values)
        assert all(flags)

    def test_none_significant(self):
        p_values = [0.5, 0.6, 0.7]
        flags = holm_bonferroni(p_values)
        assert not any(flags)

    def test_invalid_p_values(self):
        with pytest.raises(ValueError):
            holm_bonferroni([1.5, 0.5])

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            holm_bonferroni([0.1], alpha=0.0)


# ---------------------------------------------------------------------------
# aggregate_seed_runs
# ---------------------------------------------------------------------------


class TestAggregateSeedRuns:
    def test_known_input(self):
        seed_results = [
            [0.0, 1.0],  # mean 0.5
            [1.0, 1.0],  # mean 1.0
            [0.0, 0.0],  # mean 0.0
        ]
        summary = aggregate_seed_runs(seed_results)
        assert summary["n_seeds"] == 3
        assert summary["mean"] == pytest.approx(0.5)
        assert summary["min"] == pytest.approx(0.0)
        assert summary["max"] == pytest.approx(1.0)
        assert summary["median"] == pytest.approx(0.5)
        # std with ddof=1 across [0.5, 1.0, 0.0] = 0.5
        assert summary["std"] == pytest.approx(0.5)
        assert summary["sem"] == pytest.approx(0.5 / math.sqrt(3))

    def test_single_seed_no_spread(self):
        summary = aggregate_seed_runs([[0.1, 0.2, 0.3]])
        assert summary["n_seeds"] == 1
        assert summary["mean"] == pytest.approx(0.2)
        assert summary["std"] == 0.0
        assert summary["sem"] == 0.0

    def test_varying_inner_lengths(self):
        seed_results = [
            [1.0, 1.0, 1.0],
            [0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
        ]
        summary = aggregate_seed_runs(seed_results)
        # Per-seed means: 1.0, 0.0, 0.5 -> overall mean 0.5
        assert summary["mean"] == pytest.approx(0.5)
        assert summary["min"] == pytest.approx(0.0)
        assert summary["max"] == pytest.approx(1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_seed_runs([])

    def test_inner_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_seed_runs([[]])
