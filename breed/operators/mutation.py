"""Mutation operators for introducing variation into agent genomes.

Every operator returns a **new** Genome (deep copy + mutation).  The original
genome is never modified.
"""

from __future__ import annotations

import copy
import math
import random
import uuid

from typing import Any

from breed.genome import (
    DEFAULT_GENE_TEMPLATES,
    Gene,
    GeneType,
    Genome,
)


def _clone_genome(genome: Genome) -> Genome:
    """Return a deep copy of *genome* with a fresh UUID."""
    cloned = genome.model_copy(deep=True)
    cloned = cloned.model_copy(update={"genome_id": uuid.uuid4().hex})
    return cloned


# ---------------------------------------------------------------------------
# 1. parameter_jitter  (FLOAT gene)
# ---------------------------------------------------------------------------

def parameter_jitter(
    genome: Genome,
    gene_name: str,
    strength: float = 0.1,
    seed: int | None = None,
) -> Genome:
    """Apply Gaussian noise to a FLOAT gene and clamp to its valid range.

    Args:
        genome: Source genome (not modified).
        gene_name: Name of the FLOAT gene to jitter.
        strength: Standard deviation of the noise, expressed as a fraction of
            the gene's range.
        seed: Optional RNG seed.

    Returns:
        A new Genome with the jittered gene.

    Raises:
        ValueError: If the gene is not FLOAT or does not exist.
    """
    if gene_name not in genome.genes:
        raise ValueError(f"Gene '{gene_name}' not found in genome")
    gene = genome.genes[gene_name]
    if gene.type != GeneType.FLOAT:
        raise ValueError(f"Gene '{gene_name}' is {gene.type}, expected FLOAT")
    if gene.range is None:
        raise ValueError(f"Gene '{gene_name}' has no range defined")

    rng = random.Random(seed)
    lo, hi = gene.range
    sigma = strength * (hi - lo)
    new_val = gene.value + rng.gauss(0, sigma)
    new_val = max(lo, min(hi, round(new_val, 6)))

    mutated = _clone_genome(genome)
    mutated.genes[gene_name] = gene.model_copy(update={"value": new_val}, deep=True)
    return mutated


# ---------------------------------------------------------------------------
# 2. set_swap  (generic SET gene mutation)
# ---------------------------------------------------------------------------

def set_swap(
    genome: Genome,
    gene_name: str,
    seed: int | None = None,
) -> Genome:
    """Add, remove, or swap one item in any SET gene.

    The operation is chosen uniformly at random from the three possibilities
    (add / remove / swap), subject to feasibility (e.g. cannot remove from an
    empty set, cannot add when all items are already present).

    Args:
        genome: Source genome (not modified).
        gene_name: Name of the SET gene to mutate.
        seed: Optional RNG seed.

    Returns:
        A new Genome with a modified SET gene.

    Raises:
        ValueError: If the gene is missing, not a SET, or has no available list.
    """
    if gene_name not in genome.genes:
        raise ValueError(f"Gene '{gene_name}' not found in genome")
    gene = genome.genes[gene_name]
    if gene.type != GeneType.SET:
        raise ValueError(f"Gene '{gene_name}' is {gene.type}, expected SET")
    if gene.available is None:
        raise ValueError(f"Gene '{gene_name}' has no 'available' list defined")

    rng = random.Random(seed)
    current: list[str] = list(gene.value)
    available_set = set(gene.available)
    current_set = set(current)
    not_present = sorted(available_set - current_set)

    ops: list[str] = []
    if not_present:
        ops.append("add")
    if current:
        ops.append("remove")
    if current and not_present:
        ops.append("swap")

    if not ops:
        # Edge case: no valid operation (empty set, no available items)
        return _clone_genome(genome)

    op = rng.choice(ops)

    if op == "add":
        current.append(rng.choice(not_present))
    elif op == "remove":
        current.remove(rng.choice(current))
    elif op == "swap":
        current.remove(rng.choice(current))
        # Recalculate not_present after removal
        not_present_now = sorted(available_set - set(current))
        if not_present_now:
            current.append(rng.choice(not_present_now))

    new_value = sorted(current)
    mutated = _clone_genome(genome)
    mutated.genes[gene_name] = gene.model_copy(update={"value": new_value}, deep=True)
    return mutated


def tool_swap(
    genome: Genome,
    seed: int | None = None,
) -> Genome:
    """Add, remove, or swap one tool in the ``tool_policy`` SET gene.

    Backward-compatible wrapper around :func:`set_swap`.

    Args:
        genome: Source genome (not modified).
        seed: Optional RNG seed.

    Returns:
        A new Genome with a modified tool_policy.

    Raises:
        ValueError: If ``tool_policy`` gene is missing or not a SET.
    """
    return set_swap(genome, gene_name="tool_policy", seed=seed)


# ---------------------------------------------------------------------------
# 3. strategy_mutation  (ENUM gene)
# ---------------------------------------------------------------------------

def strategy_mutation(
    genome: Genome,
    gene_name: str,
    seed: int | None = None,
) -> Genome:
    """Swap an ENUM gene to a random alternative value.

    Args:
        genome: Source genome (not modified).
        gene_name: Name of the ENUM gene.
        seed: Optional RNG seed.

    Returns:
        A new Genome with the mutated ENUM gene.

    Raises:
        ValueError: If the gene is not ENUM or does not exist.
    """
    if gene_name not in genome.genes:
        raise ValueError(f"Gene '{gene_name}' not found in genome")
    gene = genome.genes[gene_name]
    if gene.type != GeneType.ENUM:
        raise ValueError(f"Gene '{gene_name}' is {gene.type}, expected ENUM")
    if gene.options is None or len(gene.options) < 2:
        raise ValueError(f"Gene '{gene_name}' needs at least 2 options to mutate")

    rng = random.Random(seed)
    alternatives = [o for o in gene.options if o != gene.value]
    new_val = rng.choice(alternatives)

    mutated = _clone_genome(genome)
    mutated.genes[gene_name] = gene.model_copy(update={"value": new_val}, deep=True)
    return mutated


# ---------------------------------------------------------------------------
# 4. text_swap  (generic TEXT gene mutation)
# ---------------------------------------------------------------------------

def text_swap(
    genome: Genome,
    gene_name: str,
    templates: list[str] | None = None,
    seed: int | None = None,
) -> Genome:
    """Swap a TEXT gene to a different template.

    If *templates* is provided, chooses an alternative from that list.
    Otherwise falls back to :data:`DEFAULT_GENE_TEMPLATES` if the gene name
    exists there.  As a last resort, shuffles the sentences in the current
    value to produce a variant.

    Args:
        genome: Source genome (not modified).
        gene_name: Name of the TEXT gene to mutate.
        templates: Optional list of alternative text values.
        seed: Optional RNG seed.

    Returns:
        A new Genome with a different TEXT gene value.

    Raises:
        ValueError: If the gene is missing or not a TEXT type.
    """
    if gene_name not in genome.genes:
        raise ValueError(f"Gene '{gene_name}' not found in genome")
    gene = genome.genes[gene_name]
    if gene.type != GeneType.TEXT:
        raise ValueError(f"Gene '{gene_name}' is {gene.type}, expected TEXT")

    rng = random.Random(seed)

    # Resolve the template list to pick from
    candidate_templates = templates
    if candidate_templates is None and gene_name in DEFAULT_GENE_TEMPLATES:
        default_spec = DEFAULT_GENE_TEMPLATES[gene_name]
        if "templates" in default_spec:
            candidate_templates = default_spec["templates"]

    if candidate_templates is not None:
        alternatives = [t for t in candidate_templates if t != gene.value]
        if alternatives:
            new_val = rng.choice(alternatives)
            mutated = _clone_genome(genome)
            mutated.genes[gene_name] = gene.model_copy(
                update={"value": new_val}, deep=True
            )
            return mutated

    # Last resort: shuffle sentences in the current value
    sentences = [s.strip() for s in str(gene.value).split(".") if s.strip()]
    if len(sentences) > 1:
        rng.shuffle(sentences)
        new_val = ". ".join(sentences) + "."
    else:
        # Shuffle words as final fallback
        words = str(gene.value).split()
        rng.shuffle(words)
        new_val = " ".join(words)

    mutated = _clone_genome(genome)
    mutated.genes[gene_name] = gene.model_copy(update={"value": new_val}, deep=True)
    return mutated


def prompt_perturb_simple(
    genome: Genome,
    seed: int | None = None,
) -> Genome:
    """Replace the ``prompt_template`` TEXT gene with a different template.

    Backward-compatible wrapper around :func:`text_swap`.

    Args:
        genome: Source genome (not modified).
        seed: Optional RNG seed.

    Returns:
        A new Genome with a different prompt_template.

    Raises:
        ValueError: If ``prompt_template`` gene is missing.
    """
    return text_swap(genome, gene_name="prompt_template", seed=seed)


# ---------------------------------------------------------------------------
# 5. vector_jitter  (NUMERIC_VECTOR gene)
# ---------------------------------------------------------------------------

def vector_jitter(
    genome: Genome,
    gene_name: str,
    strength: float = 0.1,
    seed: int | None = None,
) -> Genome:
    """Apply Gaussian noise to a NUMERIC_VECTOR gene and re-normalise to sum=1.

    Args:
        genome: Source genome (not modified).
        gene_name: Name of the NUMERIC_VECTOR gene.
        strength: Standard deviation of per-element noise.
        seed: Optional RNG seed.

    Returns:
        A new Genome with the jittered, re-normalised vector.

    Raises:
        ValueError: If the gene is not NUMERIC_VECTOR or does not exist.
    """
    if gene_name not in genome.genes:
        raise ValueError(f"Gene '{gene_name}' not found in genome")
    gene = genome.genes[gene_name]
    if gene.type != GeneType.NUMERIC_VECTOR:
        raise ValueError(f"Gene '{gene_name}' is {gene.type}, expected NUMERIC_VECTOR")

    rng = random.Random(seed)
    vec: list[float] = list(gene.value)

    # Add noise and clamp to non-negative
    noisy = [max(0.0, v + rng.gauss(0, strength)) for v in vec]

    # Re-normalise to sum = 1.0
    total = sum(noisy)
    if total > 0:
        normalised = [round(v / total, 6) for v in noisy]
    else:
        # Fallback: uniform distribution
        n = len(noisy)
        normalised = [round(1.0 / n, 6) for _ in noisy]

    mutated = _clone_genome(genome)
    mutated.genes[gene_name] = gene.model_copy(update={"value": normalised}, deep=True)
    return mutated


# ---------------------------------------------------------------------------
# 6. calibration_adjustment  (TEXT gene -- calibration_rule)
# ---------------------------------------------------------------------------

def calibration_adjustment(
    genome: Genome,
    seed: int | None = None,
) -> Genome:
    """Replace the ``calibration_rule`` TEXT gene with a different template.

    Backward-compatible wrapper around :func:`text_swap`.

    Args:
        genome: Source genome (not modified).
        seed: Optional RNG seed.

    Returns:
        A new Genome with a different calibration_rule.

    Raises:
        ValueError: If ``calibration_rule`` gene is missing.
    """
    return text_swap(genome, gene_name="calibration_rule", seed=seed)


# ---------------------------------------------------------------------------
# 7. mutate_gene  (generic mutation by gene type)
# ---------------------------------------------------------------------------

def mutate_gene(
    genome: Genome,
    gene_name: str,
    templates: list[Any] | None = None,
    strength: float = 0.1,
    seed: int | None = None,
) -> Genome:
    """Generic mutation that works on ANY gene by dispatching on its type.

    Behaviour per gene type:

    * **TEXT**: If *templates* is provided, swap to a random alternative.
      Otherwise, shuffle words in the current value.
    * **ENUM**: Swap to a random alternative option.
    * **FLOAT**: Apply Gaussian jitter (delegates to :func:`parameter_jitter`).
    * **SET**: Add/remove/swap one item (delegates to :func:`set_swap`).
    * **NUMERIC_VECTOR**: Apply noise and renormalise (delegates to
      :func:`vector_jitter`).

    Args:
        genome: Source genome (not modified).
        gene_name: Name of the gene to mutate.
        templates: Optional alternatives for TEXT genes.
        strength: Jitter strength for FLOAT / NUMERIC_VECTOR genes.
        seed: Optional RNG seed.

    Returns:
        A new Genome with the mutated gene.

    Raises:
        ValueError: If the gene does not exist.
    """
    if gene_name not in genome.genes:
        raise ValueError(f"Gene '{gene_name}' not found in genome")

    gene = genome.genes[gene_name]

    if gene.type == GeneType.TEXT:
        return text_swap(genome, gene_name, templates=templates, seed=seed)

    if gene.type == GeneType.ENUM:
        return strategy_mutation(genome, gene_name, seed=seed)

    if gene.type == GeneType.FLOAT:
        return parameter_jitter(genome, gene_name, strength=strength, seed=seed)

    if gene.type == GeneType.SET:
        return set_swap(genome, gene_name, seed=seed)

    if gene.type == GeneType.NUMERIC_VECTOR:
        return vector_jitter(genome, gene_name, strength=strength, seed=seed)

    raise ValueError(f"Unsupported gene type: {gene.type}")
