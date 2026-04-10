"""Statistical testing utilities for the agentbreed research paper.

This module provides rigorous statistical primitives for publication-quality
empirical analysis: paired/non-parametric significance tests, effect sizes,
bootstrap confidence intervals, calibration metrics (Brier score, ECE,
reliability diagrams), multiple-testing correction, and seed aggregation.

All functions consume any sequence of floats and convert to NumPy arrays
internally for performance. Inputs are validated at the boundary; NaN values
and length mismatches raise ``ValueError``.

The module deliberately depends only on NumPy and SciPy. It is the statistical
backbone for the agentbreed paper and must remain dependency-light, pure, and
deterministic where possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Sequence

import numpy as np
from scipy import stats as scipy_stats

__all__ = [
    "TestResult",
    "BootstrapCI",
    "brier_score",
    "calibration_error",
    "reliability_diagram_data",
    "paired_t_test",
    "wilcoxon_signed_rank",
    "bootstrap_ci",
    "bootstrap_ci_difference",
    "cohens_d",
    "cohens_d_paired",
    "holm_bonferroni",
    "aggregate_seed_runs",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestResult:
    """Result of a statistical hypothesis test.

    Attributes
    ----------
    statistic : float
        The test statistic (e.g. ``t`` for a t-test, ``W`` for Wilcoxon).
    p_value : float
        Two-sided p-value.
    effect_size : float or None
        Effect size associated with the test (Cohen's ``d_z`` for paired
        tests, rank-biserial correlation for Wilcoxon, ``None`` if undefined).
    method : str
        Human-readable name of the test (e.g. ``"paired_t"``).
    n : int
        Sample size used for the test.
    significant : bool
        ``True`` iff ``p_value < 0.05``.
    """

    # Tell pytest this is not a test class (name happens to start with "Test").
    __test__: ClassVar[bool] = False

    statistic: float
    p_value: float
    effect_size: float | None
    method: str
    n: int
    significant: bool

    def __str__(self) -> str:
        """Return a one-line pretty representation of the test result."""
        stat_label = "t" if self.method.startswith("paired_t") else (
            "W" if "wilcoxon" in self.method else "stat"
        )
        effect_part = ""
        if self.effect_size is not None:
            effect_label = "d" if "t" in self.method else "r"
            effect_part = f", {effect_label}={self.effect_size:.3f}"
        sig_part = "significant" if self.significant else "not_significant"
        return (
            f"{self.method}: {stat_label}={self.statistic:.3f}, "
            f"p={self.p_value:.4f}{effect_part}, n={self.n}, {sig_part}"
        )


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval for a scalar statistic.

    Attributes
    ----------
    point_estimate : float
        The statistic computed on the observed sample.
    lower : float
        Lower bound of the confidence interval.
    upper : float
        Upper bound of the confidence interval.
    confidence : float
        Nominal confidence level (e.g. ``0.95``).
    n_bootstrap : int
        Number of bootstrap resamples.
    method : str
        Bootstrap method (currently always ``"percentile"``).
    """

    point_estimate: float
    lower: float
    upper: float
    confidence: float
    n_bootstrap: int
    method: str

    def __str__(self) -> str:
        return (
            f"{self.point_estimate:.4f} "
            f"[{self.lower:.4f}, {self.upper:.4f}] "
            f"({int(self.confidence * 100)}% CI, "
            f"{self.method}, B={self.n_bootstrap})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_array(x: Sequence[float], name: str) -> np.ndarray:
    """Convert a sequence to a 1-D float ``np.ndarray`` with validation."""
    if x is None:
        raise ValueError(f"{name} must not be None")
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN values")
    return arr


def _check_paired(a: np.ndarray, b: np.ndarray) -> None:
    """Ensure two arrays have identical lengths for paired analysis."""
    if a.shape != b.shape:
        raise ValueError(
            f"paired arrays must have the same shape, got {a.shape} and {b.shape}"
        )


def _check_probabilities(p: np.ndarray, name: str) -> None:
    """Ensure values are valid probabilities in [0, 1]."""
    if (p < 0).any() or (p > 1).any():
        raise ValueError(f"{name} must lie in [0, 1]")


def _check_binary(y: np.ndarray, name: str) -> None:
    """Ensure values are 0/1 outcomes."""
    if not np.all((y == 0) | (y == 1)):
        raise ValueError(f"{name} must contain only 0/1 values")


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def brier_score(
    predictions: Sequence[float],
    outcomes: Sequence[float],
) -> float:
    """Compute the Brier score (mean squared error of probabilistic forecasts).

    Lower is better; ``0.0`` is perfect and ``1.0`` is worst-case for binary
    outcomes with the predicted probability on the wrong side.

    Parameters
    ----------
    predictions : sequence of float
        Predicted probabilities in ``[0, 1]``.
    outcomes : sequence of float
        Binary outcomes in ``{0, 1}``.

    Returns
    -------
    float
        Mean squared error of the predictions vs. outcomes.

    Raises
    ------
    ValueError
        If inputs are empty, contain NaNs, are different lengths, or fall
        outside the valid ranges.
    """
    p = _to_array(predictions, "predictions")
    y = _to_array(outcomes, "outcomes")
    _check_paired(p, y)
    _check_probabilities(p, "predictions")
    _check_binary(y, "outcomes")
    return float(np.mean((p - y) ** 2))


def calibration_error(
    predictions: Sequence[float],
    outcomes: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error (ECE) with equal-width bins.

    ECE partitions ``[0, 1]`` into ``n_bins`` equal-width bins and computes a
    weighted average of ``|accuracy - confidence|`` per bin, where the weight
    is the fraction of samples in the bin.

    Parameters
    ----------
    predictions : sequence of float
        Predicted probabilities in ``[0, 1]``.
    outcomes : sequence of float
        Binary outcomes in ``{0, 1}``.
    n_bins : int, default=10
        Number of equal-width bins.

    Returns
    -------
    float
        Expected calibration error in ``[0, 1]``.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    p = _to_array(predictions, "predictions")
    y = _to_array(outcomes, "outcomes")
    _check_paired(p, y)
    _check_probabilities(p, "predictions")
    _check_binary(y, "outcomes")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins so that p == 1.0 falls into the final bin.
    bin_idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    n = p.size
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_conf = float(p[mask].mean())
        bin_acc = float(y[mask].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_diagram_data(
    predictions: Sequence[float],
    outcomes: Sequence[float],
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return reliability-diagram coordinates and per-bin counts.

    Parameters
    ----------
    predictions : sequence of float
        Predicted probabilities in ``[0, 1]``.
    outcomes : sequence of float
        Binary outcomes in ``{0, 1}``.
    n_bins : int, default=10
        Number of equal-width bins.

    Returns
    -------
    bin_centers : np.ndarray
        Centre of each bin (length ``n_bins``).
    bin_accuracies : np.ndarray
        Empirical accuracy within each bin (NaN for empty bins).
    bin_counts : np.ndarray
        Number of samples in each bin.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    p = _to_array(predictions, "predictions")
    y = _to_array(outcomes, "outcomes")
    _check_paired(p, y)
    _check_probabilities(p, "predictions")
    _check_binary(y, "outcomes")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    accuracies = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = bin_idx == b
        c = int(mask.sum())
        counts[b] = c
        if c > 0:
            accuracies[b] = float(y[mask].mean())
    return centers, accuracies, counts


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------


def paired_t_test(
    a: Sequence[float],
    b: Sequence[float],
    alpha: float = 0.05,
) -> TestResult:
    """Perform a paired Student's t-test on per-question scores.

    Parameters
    ----------
    a, b : sequence of float
        Paired score sequences (same length).
    alpha : float, default=0.05
        Significance threshold for the ``significant`` flag (the value used
        in ``__str__`` is hard-coded to 0.05 to match standard practice).

    Returns
    -------
    TestResult
        Test result with statistic ``t``, two-sided p-value, Cohen's
        ``d_z`` as the effect size, and sample size ``n``.

    Notes
    -----
    Cohen's ``d_z`` for paired samples is the mean of differences divided by
    the standard deviation of differences (Lakens, 2013).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    arr_a = _to_array(a, "a")
    arr_b = _to_array(b, "b")
    _check_paired(arr_a, arr_b)
    if arr_a.size < 2:
        raise ValueError("paired t-test requires at least 2 paired observations")

    diff = arr_a - arr_b
    if np.allclose(diff.std(ddof=1), 0.0):
        # Zero variance: t is undefined; if mean diff is also 0 the test is
        # vacuously non-significant, otherwise it is "infinitely" significant.
        mean_diff = float(diff.mean())
        if mean_diff == 0.0:
            return TestResult(
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                method="paired_t",
                n=int(arr_a.size),
                significant=False,
            )
        return TestResult(
            statistic=float("inf") if mean_diff > 0 else float("-inf"),
            p_value=0.0,
            effect_size=float("inf") if mean_diff > 0 else float("-inf"),
            method="paired_t",
            n=int(arr_a.size),
            significant=True,
        )

    res = scipy_stats.ttest_rel(arr_a, arr_b)
    d_z = cohens_d_paired(arr_a, arr_b)
    return TestResult(
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        effect_size=float(d_z),
        method="paired_t",
        n=int(arr_a.size),
        significant=bool(res.pvalue < alpha),
    )


def wilcoxon_signed_rank(
    a: Sequence[float],
    b: Sequence[float],
    alpha: float = 0.05,
) -> TestResult:
    """Perform a Wilcoxon signed-rank test (paired non-parametric).

    Parameters
    ----------
    a, b : sequence of float
        Paired observations.
    alpha : float, default=0.05
        Significance threshold.

    Returns
    -------
    TestResult
        Test result with the Wilcoxon W statistic, two-sided p-value, and
        the rank-biserial correlation as the effect size.

    Notes
    -----
    The rank-biserial correlation effect size is computed as
    ``r = 1 - 2*W_minus / (W_plus + W_minus)``, equivalent to the
    matched-pairs rank-biserial correlation (Kerby, 2014).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    arr_a = _to_array(a, "a")
    arr_b = _to_array(b, "b")
    _check_paired(arr_a, arr_b)
    if arr_a.size < 1:
        raise ValueError("wilcoxon requires at least 1 paired observation")

    diff = arr_a - arr_b
    nonzero = diff[diff != 0]
    n_eff = int(nonzero.size)
    if n_eff == 0:
        # All differences are zero: identical samples.
        return TestResult(
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            method="wilcoxon",
            n=int(arr_a.size),
            significant=False,
        )

    # SciPy 1.11+ supports method='auto' which picks exact for small n,
    # asymptotic otherwise. zero_method='wilcox' drops zero differences.
    res = scipy_stats.wilcoxon(
        arr_a,
        arr_b,
        zero_method="wilcox",
        alternative="two-sided",
    )

    abs_ranks = scipy_stats.rankdata(np.abs(nonzero))
    w_plus = float(abs_ranks[nonzero > 0].sum())
    w_minus = float(abs_ranks[nonzero < 0].sum())
    total = w_plus + w_minus
    rank_biserial = (w_plus - w_minus) / total if total > 0 else 0.0

    return TestResult(
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        effect_size=float(rank_biserial),
        method="wilcoxon",
        n=int(arr_a.size),
        significant=bool(res.pvalue < alpha),
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def _percentile_ci(
    boot_stats: np.ndarray,
    confidence: float,
) -> tuple[float, float]:
    """Compute the symmetric percentile interval from bootstrap statistics."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    alpha = 1.0 - confidence
    lower = float(np.quantile(boot_stats, alpha / 2.0))
    upper = float(np.quantile(boot_stats, 1.0 - alpha / 2.0))
    return lower, upper


def bootstrap_ci(
    samples: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapCI:
    """Compute a percentile bootstrap confidence interval for a statistic.

    Parameters
    ----------
    samples : sequence of float
        Observed sample.
    statistic : callable, default=``np.mean``
        Function mapping a 1-D array to a scalar.
    n_bootstrap : int, default=10000
        Number of bootstrap resamples.
    confidence : float, default=0.95
        Nominal confidence level in ``(0, 1)``.
    seed : int or None
        Seed for the bootstrap RNG (for reproducibility).

    Returns
    -------
    BootstrapCI
        Point estimate and percentile interval.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    arr = _to_array(samples, "samples")
    n = arr.size
    rng = np.random.default_rng(seed)
    # Vectorised resampling: shape (n_bootstrap, n)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    resamples = arr[idx]
    # Try the fast path: pass the whole matrix with axis=1
    try:
        boot_stats = np.asarray(statistic(resamples, axis=1), dtype=float)  # type: ignore[call-arg]
        if boot_stats.shape != (n_bootstrap,):
            raise TypeError("statistic returned wrong shape")
    except TypeError:
        boot_stats = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            boot_stats[i] = float(statistic(resamples[i]))

    point = float(statistic(arr))
    lower, upper = _percentile_ci(boot_stats, confidence)
    return BootstrapCI(
        point_estimate=point,
        lower=lower,
        upper=upper,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        method="percentile",
    )


def bootstrap_ci_difference(
    a: Sequence[float],
    b: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapCI:
    """Compute a percentile bootstrap CI for the paired difference ``a - b``.

    Parameters
    ----------
    a, b : sequence of float
        Paired observations.
    statistic : callable, default=``np.mean``
        Statistic to apply to the per-element difference vector.
    n_bootstrap : int, default=10000
        Number of bootstrap resamples.
    confidence : float, default=0.95
        Nominal confidence level.
    seed : int or None
        Seed for the bootstrap RNG.

    Returns
    -------
    BootstrapCI
        Point estimate and percentile interval for ``statistic(a - b)``.
    """
    arr_a = _to_array(a, "a")
    arr_b = _to_array(b, "b")
    _check_paired(arr_a, arr_b)
    diff = arr_a - arr_b
    return bootstrap_ci(
        diff,
        statistic=statistic,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute Cohen's d for two independent samples (pooled SD).

    Parameters
    ----------
    a, b : sequence of float
        Two independent samples (lengths may differ).

    Returns
    -------
    float
        Standardised mean difference ``(mean(a) - mean(b)) / s_pooled``,
        where ``s_pooled`` is the pooled sample standard deviation with
        ``ddof=1``. Returns ``0.0`` if the means are equal and the pooled
        SD is zero.

    Raises
    ------
    ValueError
        If either sample has fewer than 2 observations or pooled SD is
        zero with non-equal means.
    """
    arr_a = _to_array(a, "a")
    arr_b = _to_array(b, "b")
    n_a, n_b = arr_a.size, arr_b.size
    if n_a < 2 or n_b < 2:
        raise ValueError("cohens_d requires at least 2 observations per group")
    var_a = float(arr_a.var(ddof=1))
    var_b = float(arr_b.var(ddof=1))
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    mean_diff = float(arr_a.mean() - arr_b.mean())
    if pooled == 0.0:
        if mean_diff == 0.0:
            return 0.0
        raise ValueError("cohens_d undefined: pooled SD is zero but means differ")
    return float(mean_diff / pooled)


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute Cohen's d_z for paired samples.

    Parameters
    ----------
    a, b : sequence of float
        Paired observations of equal length.

    Returns
    -------
    float
        ``mean(a - b) / sd(a - b)`` with ``ddof=1``. Returns ``0.0`` when
        the difference is identically zero.
    """
    arr_a = _to_array(a, "a")
    arr_b = _to_array(b, "b")
    _check_paired(arr_a, arr_b)
    if arr_a.size < 2:
        raise ValueError("cohens_d_paired requires at least 2 paired observations")
    diff = arr_a - arr_b
    sd = float(diff.std(ddof=1))
    mean_diff = float(diff.mean())
    if sd == 0.0:
        if mean_diff == 0.0:
            return 0.0
        raise ValueError("cohens_d_paired undefined: zero SD with nonzero mean")
    return float(mean_diff / sd)


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------


def holm_bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> list[bool]:
    """Apply the Holm-Bonferroni step-down correction.

    Parameters
    ----------
    p_values : sequence of float
        Raw p-values from a family of tests.
    alpha : float, default=0.05
        Family-wise error rate to control.

    Returns
    -------
    list of bool
        Significance flag for each input p-value, in the original order.
        ``True`` means the corresponding null hypothesis is rejected.

    Notes
    -----
    Holm's procedure sorts p-values ascending and rejects ``H_(i)`` iff
    ``p_(i) < alpha / (m - i + 1)`` for all preceding ``i``; once any test
    fails, no later test is rejected.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    p = _to_array(p_values, "p_values")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("p_values must lie in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    reject_sorted = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        threshold = alpha / (m - rank)
        if p[i] < threshold:
            reject_sorted[rank] = True
        else:
            break
    flags = np.zeros(m, dtype=bool)
    flags[order] = reject_sorted
    return [bool(x) for x in flags]


# ---------------------------------------------------------------------------
# Seed aggregation
# ---------------------------------------------------------------------------


def aggregate_seed_runs(
    seed_results: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Aggregate per-seed score sequences into summary statistics.

    Each element of ``seed_results`` is the per-question (or per-task) score
    list from a single seed. The function computes the per-seed mean and
    returns summary statistics across seeds.

    Parameters
    ----------
    seed_results : sequence of sequence of float
        Outer length is the number of seeds; inner lengths can vary but
        each must be non-empty.

    Returns
    -------
    dict
        ``{"mean", "std", "sem", "min", "max", "median", "n_seeds"}`` where
        ``std`` and ``sem`` use ``ddof=1`` (sample estimators). When only
        one seed is supplied, ``std`` and ``sem`` are returned as ``0.0``.
    """
    if seed_results is None:
        raise ValueError("seed_results must not be None")
    n_seeds = len(seed_results)
    if n_seeds == 0:
        raise ValueError("seed_results must contain at least one seed")
    per_seed_means = np.empty(n_seeds, dtype=float)
    for i, run in enumerate(seed_results):
        arr = _to_array(run, f"seed_results[{i}]")
        per_seed_means[i] = float(arr.mean())

    if n_seeds == 1:
        std = 0.0
        sem = 0.0
    else:
        std = float(per_seed_means.std(ddof=1))
        sem = float(std / np.sqrt(n_seeds))

    return {
        "mean": float(per_seed_means.mean()),
        "std": std,
        "sem": sem,
        "min": float(per_seed_means.min()),
        "max": float(per_seed_means.max()),
        "median": float(np.median(per_seed_means)),
        "n_seeds": float(n_seeds),
    }
