"""Search methods for the real study.

The 8 confirmatory methods (see docs/05_methods.md):
  full_evolution, mutation_only, crossover_only, random_search,
  coordinate_descent, bayesian_opt, successive_halving, prompt_only_evolution

All methods are budget-matched to `population_size * generations = 300` LLM
evaluations on the search set, plus `|test_set|` at the end.

Every method exposes the same entry point:
    async def run(...) -> SearchResult
so the runner can dispatch dynamically.
"""

from .base import SearchResult, FitnessFn, SearchContext
from .random_search import RandomSearch

__all__ = ["SearchResult", "FitnessFn", "SearchContext", "RandomSearch"]
