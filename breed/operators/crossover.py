"""Crossover operators for combining parent genomes into offspring.

All operators return a *new* Genome with:
- A fresh UUID.
- Generation = max(parent_a, parent_b) + 1.
- Both parent IDs recorded.
- Gene values drawn or blended from the parents.
"""

from __future__ import annotations

import random
import uuid

from breed.genome import Gene, GeneType, Genome


def _child_scaffold(parent_a: Genome, parent_b: Genome) -> Genome:
    """Create an empty child genome with correct lineage metadata."""
    return Genome(
        genome_id=uuid.uuid4().hex,
        generation=max(parent_a.generation, parent_b.generation) + 1,
        parents=[parent_a.genome_id, parent_b.genome_id],
        genes={},
    )


# ---------------------------------------------------------------------------
# Uniform crossover
# ---------------------------------------------------------------------------

def uniform_crossover(
    parent_a: Genome,
    parent_b: Genome,
    seed: int | None = None,
) -> Genome:
    """Each gene is independently chosen from either parent with equal probability.

    Args:
        parent_a: First parent genome.
        parent_b: Second parent genome.
        seed: Optional RNG seed for reproducibility.

    Returns:
        A new child Genome.
    """
    rng = random.Random(seed)
    child = _child_scaffold(parent_a, parent_b)

    all_gene_names = set(parent_a.genes.keys()) | set(parent_b.genes.keys())

    genes: dict[str, Gene] = {}
    for name in sorted(all_gene_names):
        gene_a = parent_a.genes.get(name)
        gene_b = parent_b.genes.get(name)

        if gene_a is not None and gene_b is not None:
            chosen = rng.choice([gene_a, gene_b])
            genes[name] = chosen.model_copy(deep=True)
        elif gene_a is not None:
            genes[name] = gene_a.model_copy(deep=True)
        else:
            assert gene_b is not None
            genes[name] = gene_b.model_copy(deep=True)

    child = child.model_copy(update={"genes": genes})
    return child


# ---------------------------------------------------------------------------
# Component swap
# ---------------------------------------------------------------------------

_TEXT_TYPES = frozenset({GeneType.TEXT, GeneType.ENUM, GeneType.SET})
_NUMERIC_TYPES = frozenset({GeneType.FLOAT, GeneType.NUMERIC_VECTOR})


def component_swap(
    parent_a: Genome,
    parent_b: Genome,
    seed: int | None = None,
) -> Genome:
    """Text/categorical genes from one parent, numeric genes from the other.

    Which parent supplies which category is decided randomly.

    Args:
        parent_a: First parent genome.
        parent_b: Second parent genome.
        seed: Optional RNG seed for reproducibility.

    Returns:
        A new child Genome.
    """
    rng = random.Random(seed)
    child = _child_scaffold(parent_a, parent_b)

    text_donor, numeric_donor = (
        (parent_a, parent_b) if rng.random() < 0.5 else (parent_b, parent_a)
    )

    all_gene_names = set(parent_a.genes.keys()) | set(parent_b.genes.keys())

    genes: dict[str, Gene] = {}
    for name in sorted(all_gene_names):
        if name in text_donor.genes and text_donor.genes[name].type in _TEXT_TYPES:
            genes[name] = text_donor.genes[name].model_copy(deep=True)
        elif name in numeric_donor.genes and numeric_donor.genes[name].type in _NUMERIC_TYPES:
            genes[name] = numeric_donor.genes[name].model_copy(deep=True)
        else:
            # Fallback: pick from whichever parent has it
            donor = parent_a if name in parent_a.genes else parent_b
            genes[name] = donor.genes[name].model_copy(deep=True)

    child = child.model_copy(update={"genes": genes})
    return child


# ---------------------------------------------------------------------------
# Weighted crossover
# ---------------------------------------------------------------------------

def _blend_float(val_a: float, val_b: float, weight_a: float, gene_range: tuple[float, float]) -> float:
    blended = weight_a * val_a + (1.0 - weight_a) * val_b
    lo, hi = gene_range
    return max(lo, min(hi, round(blended, 6)))


def _blend_vector(vec_a: list[float], vec_b: list[float], weight_a: float) -> list[float]:
    blended = [weight_a * a + (1.0 - weight_a) * b for a, b in zip(vec_a, vec_b)]
    total = sum(blended)
    if total > 0:
        blended = [round(v / total, 6) for v in blended]
    return blended


def weighted_crossover(
    parent_a: Genome,
    parent_b: Genome,
    fitness_a: float,
    fitness_b: float,
    seed: int | None = None,
) -> Genome:
    """Bias gene selection toward the fitter parent; blend numeric genes.

    For discrete genes (TEXT, ENUM, SET) the fitter parent's allele is chosen
    with probability proportional to its relative fitness.  For numeric genes
    (FLOAT, NUMERIC_VECTOR) the values are linearly blended weighted by fitness.

    Args:
        parent_a: First parent genome.
        parent_b: Second parent genome.
        fitness_a: Fitness score of parent_a (higher is better).
        fitness_b: Fitness score of parent_b (higher is better).
        seed: Optional RNG seed for reproducibility.

    Returns:
        A new child Genome.
    """
    rng = random.Random(seed)
    child = _child_scaffold(parent_a, parent_b)

    total_fitness = fitness_a + fitness_b
    weight_a = fitness_a / total_fitness if total_fitness > 0 else 0.5

    all_gene_names = set(parent_a.genes.keys()) | set(parent_b.genes.keys())

    genes: dict[str, Gene] = {}
    for name in sorted(all_gene_names):
        gene_a = parent_a.genes.get(name)
        gene_b = parent_b.genes.get(name)

        if gene_a is not None and gene_b is not None:
            if gene_a.type == GeneType.FLOAT and gene_a.range is not None:
                blended_val = _blend_float(gene_a.value, gene_b.value, weight_a, gene_a.range)
                genes[name] = gene_a.model_copy(update={"value": blended_val}, deep=True)

            elif gene_a.type == GeneType.NUMERIC_VECTOR:
                blended_vec = _blend_vector(gene_a.value, gene_b.value, weight_a)
                genes[name] = gene_a.model_copy(update={"value": blended_vec}, deep=True)

            else:
                # Discrete gene: probabilistic selection biased toward fitter parent
                chosen = gene_a if rng.random() < weight_a else gene_b
                genes[name] = chosen.model_copy(deep=True)

        elif gene_a is not None:
            genes[name] = gene_a.model_copy(deep=True)
        else:
            assert gene_b is not None
            genes[name] = gene_b.model_copy(deep=True)

    child = child.model_copy(update={"genes": genes})
    return child
