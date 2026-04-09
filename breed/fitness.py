"""Fitness evaluation system for evolving AI agents.

Provides pluggable evaluators (Brier score, pass-rate, composite) and a
calibration-error utility.  Every evaluator returns a *higher-is-better* score
in [0, 1].
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class FitnessEvaluator(ABC):
    """Base class for all fitness evaluators.

    Subclasses implement ``evaluate`` which returns a float in [0, 1] where
    higher values indicate better fitness.
    """

    @abstractmethod
    def evaluate(self, predictions: Sequence[float], outcomes: Sequence[float]) -> float:
        """Score *predictions* against ground-truth *outcomes*.

        Parameters
        ----------
        predictions:
            Model outputs.  Semantics depend on the evaluator (e.g.
            probabilities for Brier, 0/1 pass indicators for PassRate).
        outcomes:
            Ground-truth values with the same length as *predictions*.

        Returns
        -------
        float
            Fitness score in [0, 1], higher is better.
        """


# ---------------------------------------------------------------------------
# Concrete evaluators
# ---------------------------------------------------------------------------

class BrierScoreEvaluator(FitnessEvaluator):
    """1 - mean((prediction - outcome)^2).

    Brier score measures the mean squared error of probabilistic predictions.
    We return ``1 - brier`` so that **higher is better**.
    """

    def evaluate(self, predictions: Sequence[float], outcomes: Sequence[float]) -> float:
        if len(predictions) != len(outcomes):
            raise ValueError(
                f"predictions length ({len(predictions)}) != outcomes length ({len(outcomes)})"
            )
        if len(predictions) == 0:
            raise ValueError("predictions and outcomes must be non-empty")

        brier = sum(
            (p - o) ** 2 for p, o in zip(predictions, outcomes)
        ) / len(predictions)
        return 1.0 - brier


class PassRateEvaluator(FitnessEvaluator):
    """Fraction of tests passed (for coding arenas).

    *predictions* is treated as a sequence of booleans / 0-1 indicators where
    a value >= 0.5 counts as "passed".  *outcomes* is ignored (may be empty or
    a matching-length sequence -- it exists only to satisfy the interface).
    """

    def evaluate(self, predictions: Sequence[float], outcomes: Sequence[float]) -> float:
        if len(predictions) == 0:
            raise ValueError("predictions must be non-empty")
        passed = sum(1 for p in predictions if p >= 0.5)
        return passed / len(predictions)


# ---------------------------------------------------------------------------
# Composite evaluator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _MetricEntry:
    evaluator: FitnessEvaluator
    weight: float


@dataclass
class CompositeFitness:
    """Weighted combination of multiple fitness metrics.

    Parameters
    ----------
    metrics:
        ``{metric_name: (evaluator, weight)}`` mapping.  Weights are
        normalised internally so they need not sum to 1.

    Examples
    --------
    >>> comp = CompositeFitness({
    ...     "accuracy": (BrierScoreEvaluator(), 0.5),
    ...     "pass_rate": (PassRateEvaluator(), 0.5),
    ... })
    """

    metrics: dict[str, tuple[FitnessEvaluator, float]] = field(default_factory=dict)

    def evaluate(
        self,
        metric_inputs: dict[str, tuple[Sequence[float], Sequence[float]]],
    ) -> tuple[float, dict[str, float]]:
        """Evaluate all metrics and return composite + breakdown.

        Parameters
        ----------
        metric_inputs:
            ``{metric_name: (predictions, outcomes)}`` for every metric
            registered in *self.metrics*.

        Returns
        -------
        (composite_score, breakdown)
            *composite_score* is the weighted sum; *breakdown* maps each
            metric name to its individual score.
        """
        if not self.metrics:
            raise ValueError("CompositeFitness has no metrics registered")

        total_weight = sum(w for _, w in self.metrics.values())
        if total_weight == 0:
            raise ValueError("Total weight must be > 0")

        breakdown: dict[str, float] = {}
        composite = 0.0

        for name, (evaluator, weight) in self.metrics.items():
            if name not in metric_inputs:
                raise KeyError(f"Missing input for metric '{name}'")
            preds, outs = metric_inputs[name]
            score = evaluator.evaluate(preds, outs)
            breakdown[name] = score
            composite += score * (weight / total_weight)

        return composite, breakdown


# ---------------------------------------------------------------------------
# Calibration utility
# ---------------------------------------------------------------------------

def calibration_error(
    predictions: Sequence[float],
    outcomes: Sequence[float],
    num_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Groups predictions into *num_bins* equal-width bins and computes the
    weighted average of ``|avg_prediction - avg_outcome|`` per bin.

    Parameters
    ----------
    predictions:
        Predicted probabilities in [0, 1].
    outcomes:
        Binary outcomes (0 or 1).
    num_bins:
        Number of equally-spaced bins.  Default ``10``.

    Returns
    -------
    float
        ECE value in [0, 1]; lower means better calibrated.
    """
    if len(predictions) != len(outcomes):
        raise ValueError(
            f"predictions length ({len(predictions)}) != outcomes length ({len(outcomes)})"
        )
    if len(predictions) == 0:
        raise ValueError("predictions and outcomes must be non-empty")
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")

    n = len(predictions)

    # Build bins -- each bin is [low, high)
    bins: list[list[tuple[float, float]]] = [[] for _ in range(num_bins)]
    for p, o in zip(predictions, outcomes):
        # Clamp to valid bin index
        idx = min(int(math.floor(p * num_bins)), num_bins - 1)
        idx = max(idx, 0)
        bins[idx].append((p, o))

    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_pred = sum(p for p, _ in bucket) / len(bucket)
        avg_out = sum(o for _, o in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(avg_pred - avg_out)

    return ece
