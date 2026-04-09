"""Selection operators for choosing parents from a population.

Both operators return lists of selected Genomes (new references, same data).
"""

from __future__ import annotations

import random

from breed.genome import Genome


def tournament_select(
    population: list[Genome],
    fitness_scores: dict[str, float],
    k: int,
    tournament_size: int = 3,
    seed: int | None = None,
) -> list[Genome]:
    """Select *k* genomes via tournament selection.

    For each of the *k* winners, *tournament_size* genomes are sampled uniformly
    at random (with replacement) from the population, and the one with the
    highest fitness wins the tournament.

    Args:
        population: Full list of candidate genomes.
        fitness_scores: Mapping of genome_id -> fitness score (higher is better).
        k: Number of winners to select.
        tournament_size: Number of contestants per tournament.
        seed: Optional RNG seed.

    Returns:
        A list of *k* selected Genomes (deep copies).

    Raises:
        ValueError: If *population* is empty or *k* < 1.
    """
    if not population:
        raise ValueError("Cannot select from an empty population")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if tournament_size < 1:
        raise ValueError(f"tournament_size must be >= 1, got {tournament_size}")

    rng = random.Random(seed)
    winners: list[Genome] = []

    for _ in range(k):
        contestants = rng.choices(population, k=tournament_size)
        best = max(contestants, key=lambda g: fitness_scores.get(g.genome_id, 0.0))
        winners.append(best.model_copy(deep=True))

    return winners


def truncation_select(
    population: list[Genome],
    fitness_scores: dict[str, float],
    k: int,
) -> list[Genome]:
    """Select the top-*k* genomes by fitness score.

    Args:
        population: Full list of candidate genomes.
        fitness_scores: Mapping of genome_id -> fitness score (higher is better).
        k: Number of top genomes to keep.

    Returns:
        A sorted (descending by fitness) list of the top-*k* Genomes (deep copies).

    Raises:
        ValueError: If *population* is empty or *k* < 1.
    """
    if not population:
        raise ValueError("Cannot select from an empty population")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    ranked = sorted(
        population,
        key=lambda g: fitness_scores.get(g.genome_id, 0.0),
        reverse=True,
    )
    top_k = ranked[:k]
    return [g.model_copy(deep=True) for g in top_k]
