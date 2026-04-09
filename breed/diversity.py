"""Diversity preservation utilities for evolutionary populations.

Prevents premature convergence by measuring population novelty and injecting
random immigrants when diversity drops.
"""

from __future__ import annotations

from typing import Any

from breed.genome import Genome, create_genome_from_template, create_random_genome


def novelty_score(population: list[Genome]) -> dict[str, float]:
    """Compute the average genome distance from each individual to all others.

    A higher score means the genome is more *novel* (dissimilar to the rest of
    the population).  Scores are non-negative; a population of size 1 receives
    a score of 0.0.

    Args:
        population: The current generation of genomes.

    Returns:
        A mapping of genome_id -> novelty score.
    """
    if len(population) <= 1:
        return {g.genome_id: 0.0 for g in population}

    scores: dict[str, float] = {}
    n = len(population)

    for i, genome in enumerate(population):
        total_dist = 0.0
        for j, other in enumerate(population):
            if i != j:
                total_dist += genome.distance(other)
        scores[genome.genome_id] = total_dist / (n - 1)

    return scores


def inject_immigrants(
    population: list[Genome],
    count: int,
    seed: int | None = None,
    gene_template: dict[str, dict[str, Any]] | None = None,
) -> list[Genome]:
    """Add *count* random genomes to the population to boost diversity.

    The returned list is a **new** list containing all original genomes plus the
    newly generated immigrants.  The original list is not mutated.

    Args:
        population: Existing population of genomes.
        count: Number of random immigrants to add.
        seed: Optional base seed -- each immigrant uses seed + i for
            reproducibility.
        gene_template: Optional custom gene template dict.  If provided, new
            immigrants use :func:`create_genome_from_template` instead of
            the default template.

    Returns:
        A new list with the original population followed by *count* immigrants.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")

    new_population = list(population)
    for i in range(count):
        immigrant_seed = (seed + i) if seed is not None else None
        if gene_template is not None:
            new_population.append(
                create_genome_from_template(gene_template, seed=immigrant_seed)
            )
        else:
            new_population.append(create_random_genome(seed=immigrant_seed))

    return new_population
