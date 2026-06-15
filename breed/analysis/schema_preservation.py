"""Fitness landscape diagnostics and schema preservation metrics.

This module provides:

- `typed_hamming_distance`: a Hamming-style distance over typed genomes
  (ENUM, TEXT, FLOAT, SET, NUMERIC_VECTOR) that respects type semantics.
- `compute_ruggedness`: fitness-distance correlation analysis to
  characterize how rugged the landscape is. Smooth landscapes favor
  hill-climbing; rugged landscapes favor diverse population-based search.
- `effective_dimensionality_via_pca`: weighted PCA on one-hot genome
  features to estimate the effective dimensionality of the fitness-
  explaining subspace.
- `schema_preservation_rate`: measures what fraction of high-performing
  2-gene schemata survive a crossover operator, as a window into the
  Schema Theorem for agent configurations.

These metrics feed into both the theory section (H3 mechanism) and
the pilot diagnostics (landscape sanity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from breed.genome import GeneType, Genome


# ---------------------------------------------------------------------------
# Typed Hamming-like distance
# ---------------------------------------------------------------------------


def typed_hamming_distance(a: Genome, b: Genome) -> int:
    """Count the number of genes that differ between two genomes.

    For ENUM and TEXT genes: different if values are unequal.
    For FLOAT genes: different if ``|a - b| / range >= 0.1`` (10% of range).
    For SET genes: different if symmetric difference is non-empty.
    For NUMERIC_VECTOR genes: different if L2 distance >= 0.1.

    Parameters
    ----------
    a, b:
        Two genomes sharing the same gene template. Extra genes in
        either are counted as differences.
    """
    differences = 0
    shared = set(a.genes.keys()) & set(b.genes.keys())
    for name in shared:
        g_a = a.genes[name]
        g_b = b.genes[name]
        if g_a.type == GeneType.FLOAT:
            lo, hi = g_a.range if g_a.range else (0.0, 1.0)
            span = hi - lo if hi > lo else 1.0
            if abs(float(g_a.value) - float(g_b.value)) / span >= 0.1:
                differences += 1
        elif g_a.type == GeneType.SET:
            if set(g_a.value) != set(g_b.value):
                differences += 1
        elif g_a.type == GeneType.NUMERIC_VECTOR:
            vec_a = np.asarray(g_a.value, dtype=float)
            vec_b = np.asarray(g_b.value, dtype=float)
            if vec_a.shape == vec_b.shape:
                if float(np.linalg.norm(vec_a - vec_b)) >= 0.1:
                    differences += 1
            else:
                differences += 1
        else:  # ENUM, TEXT
            if g_a.value != g_b.value:
                differences += 1
    only_a = set(a.genes.keys()) - shared
    only_b = set(b.genes.keys()) - shared
    differences += len(only_a) + len(only_b)
    return differences


# ---------------------------------------------------------------------------
# Fitness-distance ruggedness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuggednessReport:
    """Characterization of landscape ruggedness from a sampled population."""

    n_samples: int
    mean_fitness: float
    std_fitness: float
    max_distance: int
    correlation_by_distance: dict[int, float]
    autocorrelation_length: float
    normalized_ruggedness: float  # 0 = smooth, 1 = fully random

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "mean_fitness": self.mean_fitness,
            "std_fitness": self.std_fitness,
            "max_distance": self.max_distance,
            "correlation_by_distance": dict(self.correlation_by_distance),
            "autocorrelation_length": self.autocorrelation_length,
            "normalized_ruggedness": self.normalized_ruggedness,
        }


def compute_ruggedness(
    population: Sequence[Genome],
    fitness_scores: Sequence[float],
) -> RuggednessReport:
    """Compute landscape ruggedness via fitness-distance correlation.

    For each pair of genomes, compute the typed Hamming distance and the
    squared fitness difference. The correlation at distance ``d`` is
    estimated as ``1 - 0.5 * mean((f_i - f_j)^2 | d) / var(f)``. The
    autocorrelation length is the smallest distance at which that
    correlation drops below ``1/e``. Shorter length means rougher
    landscape.

    Returns a :class:`RuggednessReport`.
    """
    n = len(population)
    if n < 3:
        raise ValueError("Need at least 3 genomes for a meaningful ruggedness estimate")

    scores = np.asarray(fitness_scores, dtype=float)
    mean_f = float(scores.mean())
    std_f = float(scores.std(ddof=1)) if n > 1 else 0.0
    var_f = float(scores.var(ddof=1)) if n > 1 else 0.0

    by_dist: dict[int, list[float]] = {}
    max_d = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = typed_hamming_distance(population[i], population[j])
            max_d = max(max_d, d)
            diff = float((scores[i] - scores[j]) ** 2)
            by_dist.setdefault(d, []).append(diff)

    correlation_by_distance: dict[int, float] = {}
    for d, diffs in by_dist.items():
        if var_f > 0:
            corr = 1.0 - 0.5 * float(np.mean(diffs)) / var_f
        else:
            corr = 1.0
        correlation_by_distance[d] = max(-1.0, min(1.0, corr))

    sorted_ds = sorted(correlation_by_distance.keys())
    threshold = 1.0 / np.e
    auto_len: float = float(max_d)
    for d in sorted_ds:
        if correlation_by_distance[d] < threshold:
            auto_len = float(d)
            break

    normalized = 1.0 - (auto_len / max_d if max_d > 0 else 0.0)
    normalized = max(0.0, min(1.0, normalized))

    return RuggednessReport(
        n_samples=n,
        mean_fitness=mean_f,
        std_fitness=std_f,
        max_distance=max_d,
        correlation_by_distance=correlation_by_distance,
        autocorrelation_length=auto_len,
        normalized_ruggedness=normalized,
    )


# ---------------------------------------------------------------------------
# Effective dimensionality via weighted PCA
# ---------------------------------------------------------------------------


def effective_dimensionality_via_pca(
    feature_matrix: np.ndarray,
    fitness: np.ndarray,
    variance_threshold: float = 0.90,
) -> int:
    """Estimate the effective dimensionality of the fitness-explaining subspace.

    Given a feature matrix ``X`` of shape ``(n_samples, n_features)``,
    compute the fitness-weighted SVD of the centered matrix and return the
    smallest ``k`` such that the top-``k`` singular values explain at
    least ``variance_threshold`` of the weighted variance.
    """
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix must be 2-D")
    n_samples, n_features = feature_matrix.shape
    if n_samples != len(fitness):
        raise ValueError("feature_matrix rows must match fitness length")
    if n_samples < 2 or n_features < 1:
        return 0

    centered = feature_matrix - feature_matrix.mean(axis=0, keepdims=True)
    fitness_arr = np.asarray(fitness, dtype=float)
    weights = np.abs(fitness_arr - fitness_arr.mean()) + 1e-9
    weights = weights / weights.sum()
    weighted = centered * np.sqrt(weights[:, None])
    try:
        _, s, _ = np.linalg.svd(weighted, full_matrices=False)
    except np.linalg.LinAlgError:
        return n_features
    if s.sum() == 0:
        return 0
    variance_ratio = (s ** 2) / (s ** 2).sum()
    cumulative = np.cumsum(variance_ratio)
    k = int(np.searchsorted(cumulative, variance_threshold) + 1)
    return min(k, n_features)


# ---------------------------------------------------------------------------
# Schema preservation under crossover operators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemaPreservationReport:
    """Schema preservation statistics for a single crossover operator."""

    operator_name: str
    n_parent_pairs: int
    total_schemata: int
    preserved_schemata: int
    preservation_rate: float
    by_gene_pair: dict[tuple[str, str], float]

    def to_dict(self) -> dict:
        return {
            "operator_name": self.operator_name,
            "n_parent_pairs": self.n_parent_pairs,
            "total_schemata": self.total_schemata,
            "preserved_schemata": self.preserved_schemata,
            "preservation_rate": self.preservation_rate,
            "by_gene_pair": {f"{a}|{b}": v for (a, b), v in self.by_gene_pair.items()},
        }


def _shared_schemata(a: Genome, b: Genome) -> list[tuple[str, str]]:
    """Return the set of 2-gene schemata (pairs of equal gene values)
    that parents a and b share. Only ENUM and SET genes are considered
    because they have discrete equality semantics suitable for the
    schema-theorem framing."""
    shared: list[tuple[str, str]] = []
    names = sorted(set(a.genes.keys()) & set(b.genes.keys()))
    for i, name_i in enumerate(names):
        g_i_a = a.genes[name_i]
        g_i_b = b.genes[name_i]
        if g_i_a.type not in (GeneType.ENUM,):
            continue
        if g_i_a.value != g_i_b.value:
            continue
        for name_j in names[i + 1 :]:
            g_j_a = a.genes[name_j]
            g_j_b = b.genes[name_j]
            if g_j_a.type not in (GeneType.ENUM,):
                continue
            if g_j_a.value != g_j_b.value:
                continue
            shared.append((name_i, name_j))
    return shared


def _schema_present(child: Genome, schema: tuple[str, str], parent: Genome) -> bool:
    """Check whether the child preserves the value pair that the parent
    had for the given schema (pair of gene names)."""
    name_a, name_b = schema
    if name_a not in child.genes or name_b not in child.genes:
        return False
    return (
        child.genes[name_a].value == parent.genes[name_a].value
        and child.genes[name_b].value == parent.genes[name_b].value
    )


def schema_preservation_rate(
    operator_name: str,
    operator: Callable[[Genome, Genome, int | None], Genome],
    parents: Sequence[tuple[Genome, Genome]],
    seed: int | None = None,
) -> SchemaPreservationReport:
    """Measure schema preservation for a crossover operator.

    For each parent pair, count the 2-gene schemata the parents share,
    produce a child, and count how many of those schemata survive in the
    child (in the specific parent value configuration). The per-operator
    preservation rate is the fraction preserved across all parent pairs.

    The ``operator`` callable must have signature ``(parent_a, parent_b,
    seed=int|None) -> Genome`` and return a child genome.
    """
    import random

    rng = random.Random(seed)
    total = 0
    preserved = 0
    by_pair_counts: dict[tuple[str, str], list[int]] = {}

    for parent_a, parent_b in parents:
        shared = _shared_schemata(parent_a, parent_b)
        if not shared:
            continue
        child_seed = rng.randint(0, 2**31 - 1)
        child = operator(parent_a, parent_b, child_seed)
        for schema in shared:
            total += 1
            if _schema_present(child, schema, parent_a):
                preserved += 1
                by_pair_counts.setdefault(schema, [0, 0])[0] += 1
            else:
                by_pair_counts.setdefault(schema, [0, 0])
            by_pair_counts[schema][1] += 1

    rate = preserved / total if total > 0 else 0.0
    by_pair_rate: dict[tuple[str, str], float] = {
        pair: (counts[0] / counts[1]) if counts[1] > 0 else 0.0
        for pair, counts in by_pair_counts.items()
    }

    return SchemaPreservationReport(
        operator_name=operator_name,
        n_parent_pairs=len(parents),
        total_schemata=total,
        preserved_schemata=preserved,
        preservation_rate=rate,
        by_gene_pair=by_pair_rate,
    )
