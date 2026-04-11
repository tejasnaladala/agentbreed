"""Search-method abstractions shared across all 8 confirmatory methods."""

from __future__ import annotations

import abc
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from ..genome import Genome, genome_content_hash


# A FitnessFn takes a genome and returns (search_set_fitness, n_llm_calls_used).
# It MUST report the number of LLM calls it consumed so the runner can enforce
# the budget exactly.
FitnessFn = Callable[[Genome], Awaitable[Tuple[float, int]]]


@dataclass
class SearchContext:
    """Everything a search method needs that is not the fitness function itself."""
    benchmark: str
    seed: int
    budget_llm_calls: int
    population_size: int
    generations: int


@dataclass
class EvaluatedGenome:
    genome: Genome
    fitness: float
    hash: str


@dataclass
class SearchResult:
    """Output of a search run."""
    champion: Genome
    champion_fitness_search: float
    all_evaluated: List[EvaluatedGenome] = field(default_factory=list)
    n_llm_calls: int = 0
    per_generation_best: List[float] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)


class SearchMethod(abc.ABC):
    """Abstract search method interface."""

    method_name: str = ""

    @abc.abstractmethod
    async def run(
        self,
        *,
        ctx: SearchContext,
        fitness_fn: FitnessFn,
    ) -> SearchResult:
        """Run the search method and return a SearchResult.

        The fitness_fn is async so it can parallelize LLM calls internally.
        The method must stop as soon as ctx.budget_llm_calls is reached.
        """


def make_evaluated(genome: Genome, fitness: float) -> EvaluatedGenome:
    return EvaluatedGenome(
        genome=genome,
        fitness=fitness,
        hash=genome_content_hash(genome),
    )


def select_champion(evaluated: List[EvaluatedGenome]) -> EvaluatedGenome:
    """Pick the argmax-fitness evaluated genome, breaking ties deterministically by hash."""
    if not evaluated:
        raise ValueError("cannot select champion from empty list")
    best = evaluated[0]
    for e in evaluated[1:]:
        if e.fitness > best.fitness:
            best = e
        elif e.fitness == best.fitness and e.hash < best.hash:
            best = e
    return best
