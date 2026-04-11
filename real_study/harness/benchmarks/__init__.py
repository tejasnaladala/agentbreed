"""Benchmark loaders for the real study.

Each benchmark is one module that:
1. Downloads (or loads from cache) a snapshot.
2. Exposes train/test splits as tuples of BenchmarkItem.
3. Provides a scorer from (item, agent_response) -> float in [0, 1].
4. Provides the prompt-construction logic given a Genome.
5. Exposes contamination-probe helpers.

All loaders inherit from Benchmark and implement the abstract methods.
"""

from .base import Benchmark, BenchmarkItem, BenchmarkScore

__all__ = ["Benchmark", "BenchmarkItem", "BenchmarkScore"]
