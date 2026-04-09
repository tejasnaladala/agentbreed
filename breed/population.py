"""Population management for agent evolution.

A ``Population`` wraps a list of ``Genome`` instances and exposes the
evolutionary operators (select, breed, mutate, immigrate) needed for a
single generation step.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

from breed.genome import Genome, create_random_genome
from breed.operators.crossover import (
    component_swap,
    uniform_crossover,
    weighted_crossover,
)
from breed.operators.mutation import (
    calibration_adjustment,
    parameter_jitter,
    prompt_perturb_simple,
    strategy_mutation,
    tool_swap,
    vector_jitter,
)
from breed.operators.selection import tournament_select, truncation_select
from breed.diversity import inject_immigrants

# Crossover operators that take (parent_a, parent_b, seed=) only
_SIMPLE_CROSSOVER_OPS: tuple[Callable[..., Genome], ...] = (
    uniform_crossover,
    component_swap,
)

# Mutation operators that need only (genome, seed=)
_GENOME_ONLY_MUTATION_OPS: tuple[Callable[..., Genome], ...] = (
    tool_swap,
    prompt_perturb_simple,
    calibration_adjustment,
)

# Mutation operators that need (genome, gene_name, seed=)
# Paired with the gene names they target.
_GENE_TARGETED_MUTATIONS: tuple[tuple[Callable[..., Genome], str], ...] = (
    (parameter_jitter, "temperature"),
    (parameter_jitter, "risk_threshold"),
    (vector_jitter, "evidence_weighting"),
    (strategy_mutation, "memory_structure"),
    (strategy_mutation, "decomposition_style"),
    (strategy_mutation, "answer_format"),
)


def _build_fitness_dict(
    members: list[Genome], scores: Sequence[float]
) -> dict[str, float]:
    """Convert positional scores into a genome_id -> score mapping."""
    return {m.genome_id: s for m, s in zip(members, scores)}


@dataclass
class Population:
    """A generation of agent genomes.

    Attributes
    ----------
    members : list[Genome]
        The current genomes in this population.
    generation : int
        Zero-indexed generation counter.  Incremented after ``breed``.
    """

    members: list[Genome] = field(default_factory=list)
    generation: int = 0

    # -- convenience property ------------------------------------------------

    @property
    def size(self) -> int:
        """Number of genomes currently in the population."""
        return len(self.members)

    # -- factory -------------------------------------------------------------

    @classmethod
    def spawn(cls, size: int, seed: int | None = None) -> Population:
        """Create a new random population of *size* genomes.

        Parameters
        ----------
        size:
            Number of genomes to create.
        seed:
            Optional RNG seed for reproducibility.
        """
        if size < 1:
            raise ValueError("Population size must be >= 1")
        rng = random.Random(seed)
        members = [
            create_random_genome(seed=rng.randint(0, 2**63)) for _ in range(size)
        ]
        return cls(members=members, generation=0)

    # -- selection -----------------------------------------------------------

    def select(
        self,
        fitness_scores: Sequence[float],
        top_k: int,
        method: Literal["tournament", "truncation"] = "tournament",
        tournament_size: int = 3,
        seed: int | None = None,
    ) -> None:
        """Select survivors **in-place**, reducing the population to *top_k*.

        Parameters
        ----------
        fitness_scores:
            One score per member (same order as ``self.members``).
        top_k:
            How many survivors to keep.
        method:
            ``"tournament"`` or ``"truncation"``.
        tournament_size:
            Only relevant when *method* is ``"tournament"``.
        seed:
            Optional RNG seed.
        """
        if len(fitness_scores) != self.size:
            raise ValueError(
                f"fitness_scores length ({len(fitness_scores)}) != population size ({self.size})"
            )
        if top_k < 1 or top_k > self.size:
            raise ValueError(f"top_k ({top_k}) must be in [1, {self.size}]")

        score_dict = _build_fitness_dict(self.members, fitness_scores)

        if method == "tournament":
            selected = tournament_select(
                self.members, score_dict, top_k,
                tournament_size=tournament_size, seed=seed,
            )
        elif method == "truncation":
            selected = truncation_select(self.members, score_dict, top_k)
        else:
            raise ValueError(f"Unknown selection method: {method!r}")

        self.members = list(selected)

    # -- breeding ------------------------------------------------------------

    def breed(
        self,
        target_size: int,
        seed: int | None = None,
        fitness_scores: Sequence[float] | None = None,
    ) -> None:
        """Breed offspring from the current (selected) population.

        The existing members are treated as *elites* and survive into the
        next generation unchanged.  Children are produced via random
        crossover operators until the population reaches *target_size*.
        After breeding the ``generation`` counter is incremented.

        Parameters
        ----------
        target_size:
            Desired population size after breeding.  Must be >= current size.
        seed:
            Optional RNG seed.
        fitness_scores:
            Optional fitness scores for the current elites (same order).
            Passed through to the weighted crossover operator.
        """
        if target_size < self.size:
            raise ValueError(
                f"target_size ({target_size}) must be >= current size ({self.size})"
            )

        rng = random.Random(seed)
        elites = list(self.members)  # snapshot
        children: list[Genome] = []

        # Pre-build score lookup for weighted crossover
        elite_score: dict[str, float] = {}
        if fitness_scores is not None:
            elite_score = _build_fitness_dict(elites, fitness_scores)

        while len(elites) + len(children) < target_size:
            parent_a = rng.choice(elites)
            parent_b = rng.choice(elites)
            # Avoid selfing when possible
            if len(elites) > 1:
                while parent_b.genome_id == parent_a.genome_id:
                    parent_b = rng.choice(elites)

            child_seed = rng.randint(0, 2**63)

            # Choose whether to use weighted crossover (if scores available)
            if fitness_scores is not None and rng.random() < 0.33:
                child = weighted_crossover(
                    parent_a,
                    parent_b,
                    fitness_a=elite_score[parent_a.genome_id],
                    fitness_b=elite_score[parent_b.genome_id],
                    seed=child_seed,
                )
            else:
                crossover_op = rng.choice(_SIMPLE_CROSSOVER_OPS)
                child = crossover_op(parent_a, parent_b, seed=child_seed)

            children.append(child)

        self.members = elites + children
        self.generation += 1

    # -- mutation ------------------------------------------------------------

    def mutate(self, rate: float = 0.15, seed: int | None = None) -> None:
        """Apply random mutation operators to members at *rate*.

        Each member independently has a *rate* probability of being
        mutated.  The mutation operator is chosen uniformly at random
        from the available genome-only and gene-targeted operators.

        Parameters
        ----------
        rate:
            Per-genome mutation probability in [0, 1].
        seed:
            Optional RNG seed.
        """
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")

        rng = random.Random(seed)
        mutated: list[Genome] = []

        for genome in self.members:
            if rng.random() < rate:
                mutated.append(_apply_random_mutation(genome, rng))
            else:
                mutated.append(genome)

        self.members = mutated

    # -- immigration ---------------------------------------------------------

    def add_immigrants(self, count: int, seed: int | None = None) -> None:
        """Inject *count* random immigrants into the population.

        Parameters
        ----------
        count:
            Number of new random genomes to add.
        seed:
            Optional RNG seed.
        """
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")

        result = inject_immigrants(list(self.members), count, seed=seed)
        # inject_immigrants returns original + new; we only want the new tail
        self.members = result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_random_mutation(genome: Genome, rng: random.Random) -> Genome:
    """Pick a random mutation operator and apply it, handling signature differences."""
    mut_seed = rng.randint(0, 2**63)

    # Build a combined list of (callable, extra_kwargs) entries
    candidates: list[tuple[Callable[..., Genome], dict[str, Any]]] = []

    for op in _GENOME_ONLY_MUTATION_OPS:
        candidates.append((op, {}))

    for op, gene_name in _GENE_TARGETED_MUTATIONS:
        if gene_name in genome.genes:
            candidates.append((op, {"gene_name": gene_name}))

    if not candidates:
        # Fallback: no applicable mutations, return unchanged
        return genome

    op, kwargs = rng.choice(candidates)
    try:
        return op(genome, seed=mut_seed, **kwargs)
    except (ValueError, KeyError):
        # If the mutation is inapplicable (e.g. gene removed), skip gracefully
        return genome
