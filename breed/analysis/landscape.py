"""Landscape-level summaries that complement epistasis and schema analyses.

Provides:
- `LandscapeSummary`: single record describing mean, std, max, min of
  fitness plus ruggedness and effective dim.
- `summarize_landscape`: convenience wrapper that computes all the
  headline metrics from a sampled population.

The purpose of this module is to give a one-shot landscape diagnostic
that can be run on any (population, scores) pair — useful both for
reporting and for sanity checks during the pilot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from breed.analysis.schema_preservation import (
    RuggednessReport,
    compute_ruggedness,
    effective_dimensionality_via_pca,
)
from breed.genome import Gene, GeneType, Genome


@dataclass(frozen=True)
class LandscapeSummary:
    """One-shot summary of a sampled fitness landscape."""

    n_samples: int
    mean_fitness: float
    std_fitness: float
    min_fitness: float
    max_fitness: float
    median_fitness: float
    ruggedness: RuggednessReport
    effective_dimensionality: int
    feature_dimensionality: int

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "mean_fitness": self.mean_fitness,
            "std_fitness": self.std_fitness,
            "min_fitness": self.min_fitness,
            "max_fitness": self.max_fitness,
            "median_fitness": self.median_fitness,
            "ruggedness": self.ruggedness.to_dict(),
            "effective_dimensionality": self.effective_dimensionality,
            "feature_dimensionality": self.feature_dimensionality,
        }


def _gene_to_features(gene: Gene) -> list[float]:
    """Convert a single gene into a flat feature vector for PCA."""
    if gene.type == GeneType.ENUM:
        if gene.options is None:
            return [0.0]
        # One-hot
        one_hot = [0.0] * len(gene.options)
        if gene.value in gene.options:
            one_hot[gene.options.index(gene.value)] = 1.0
        return one_hot
    if gene.type == GeneType.FLOAT:
        lo, hi = gene.range if gene.range else (0.0, 1.0)
        span = hi - lo if hi > lo else 1.0
        return [(float(gene.value) - lo) / span]
    if gene.type == GeneType.SET:
        if gene.available is None:
            return [0.0]
        return [1.0 if item in gene.value else 0.0 for item in gene.available]
    if gene.type == GeneType.NUMERIC_VECTOR:
        return [float(v) for v in gene.value]
    # TEXT -> bag-of-words dimensionality is too variable; collapse to 0.
    return [0.0]


def genomes_to_feature_matrix(genomes: Sequence[Genome]) -> tuple[np.ndarray, list[str]]:
    """Convert a list of genomes into a dense feature matrix for PCA.

    Returns (matrix, feature_names). Skips TEXT genes to keep dimensionality
    finite. Feature names are of the form ``gene__subfield``.
    """
    if not genomes:
        return np.zeros((0, 0)), []

    # Determine feature layout from the first genome
    feature_names: list[str] = []
    ref = genomes[0]
    for name in sorted(ref.genes.keys()):
        gene = ref.genes[name]
        if gene.type == GeneType.TEXT:
            continue
        if gene.type == GeneType.ENUM:
            if gene.options:
                for opt in gene.options:
                    feature_names.append(f"{name}__{opt}")
        elif gene.type == GeneType.FLOAT:
            feature_names.append(f"{name}__value")
        elif gene.type == GeneType.SET:
            if gene.available:
                for item in gene.available:
                    feature_names.append(f"{name}__{item}")
        elif gene.type == GeneType.NUMERIC_VECTOR:
            for i in range(len(gene.value)):
                feature_names.append(f"{name}__{i}")

    n_features = len(feature_names)
    matrix = np.zeros((len(genomes), n_features), dtype=float)
    for row_idx, genome in enumerate(genomes):
        col = 0
        for name in sorted(genome.genes.keys()):
            gene = genome.genes[name]
            if gene.type == GeneType.TEXT:
                continue
            features = _gene_to_features(gene)
            for v in features:
                if col < n_features:
                    matrix[row_idx, col] = v
                    col += 1
    return matrix, feature_names


def summarize_landscape(
    population: Sequence[Genome],
    fitness_scores: Sequence[float],
    variance_threshold: float = 0.90,
) -> LandscapeSummary:
    """Compute a one-shot landscape summary from a sampled population.

    Combines fitness statistics, ruggedness (via fitness-distance
    correlation), and effective dimensionality (via PCA on one-hot
    encoded genome features) into a single record.
    """
    if len(population) != len(fitness_scores):
        raise ValueError("population and fitness_scores must be the same length")
    if not population:
        raise ValueError("population is empty")

    scores = np.asarray(fitness_scores, dtype=float)
    ruggedness = compute_ruggedness(population, fitness_scores)
    matrix, feature_names = genomes_to_feature_matrix(population)
    if matrix.size > 0:
        effective_dim = effective_dimensionality_via_pca(
            matrix, scores, variance_threshold=variance_threshold
        )
    else:
        effective_dim = 0

    return LandscapeSummary(
        n_samples=len(population),
        mean_fitness=float(scores.mean()),
        std_fitness=float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        min_fitness=float(scores.min()),
        max_fitness=float(scores.max()),
        median_fitness=float(np.median(scores)),
        ruggedness=ruggedness,
        effective_dimensionality=effective_dim,
        feature_dimensionality=len(feature_names),
    )
