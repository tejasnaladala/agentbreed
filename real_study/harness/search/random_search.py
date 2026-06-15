"""Random search baseline.

Samples `budget_llm_calls` genomes uniformly at random from the legal space,
evaluates each, and returns the best. This is the null operator — any other
method that cannot beat random search is suspect.

Note: one "evaluation" here = one genome-level fitness call, which the fitness
function internally implements as `|search_set|` LLM calls. So for our main
matrix, budget_llm_calls = 300 genome-level evaluations, not 300 LLM calls.
This matches the pilot and the preregistration; the term "LLM call" in the
search layer is slightly abused to mean "search-set evaluation."
"""

from __future__ import annotations

import random
from typing import List

from ..genome import random_genome_for
from .base import (
    EvaluatedGenome,
    FitnessFn,
    SearchContext,
    SearchMethod,
    SearchResult,
    make_evaluated,
    select_champion,
)


class RandomSearch(SearchMethod):
    method_name = "random_search"

    async def run(
        self,
        *,
        ctx: SearchContext,
        fitness_fn: FitnessFn,
    ) -> SearchResult:
        rng = random.Random(ctx.seed)

        # Budget: population_size * generations total search-set evaluations.
        n_evals = ctx.population_size * ctx.generations  # = 300 in the main matrix

        evaluated: List[EvaluatedGenome] = []
        per_generation_best: List[float] = []
        running_best = -float("inf")

        for i in range(n_evals):
            g = random_genome_for(ctx.benchmark, rng)
            fitness, _ = await fitness_fn(g)
            e = make_evaluated(g, fitness)
            evaluated.append(e)
            running_best = max(running_best, fitness)
            # Log per-generation-best: once every population_size evaluations
            if (i + 1) % ctx.population_size == 0:
                per_generation_best.append(running_best)

        champion = select_champion(evaluated)
        return SearchResult(
            champion=champion.genome,
            champion_fitness_search=champion.fitness,
            all_evaluated=evaluated,
            n_llm_calls=n_evals,
            per_generation_best=per_generation_best,
            notes={"method": "random_search"},
        )
