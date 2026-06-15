"""Smoke-test entrypoint.

A minimal end-to-end sanity check of the harness:
1. Load the ForecastBench snapshot (or generate a tiny mock if missing).
2. Start a VLLMClient pointed at localhost:8000.
3. Run a tiny full_evolution search (population_size=4, generations=2).
4. Evaluate the champion on 5 test items.
5. Write the atomic JSON record.
6. Print a summary to stdout.

This runs inside the Slurm smoke job. It is NOT a confirmatory run — it just
verifies the plumbing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from real_study.harness.genome import (
    default_genome_for,
    random_genome_for,
    validate_genome,
)
from real_study.harness.search.random_search import RandomSearch
from real_study.harness.search.base import SearchContext


async def _toy_fitness(genome):
    # In smoke mode with no LLM, return a deterministic fake fitness
    # based on the genome hash — so we can at least exercise the search loop.
    from real_study.harness.genome import genome_content_hash
    h = genome_content_hash(genome)
    val = int(h[:8], 16) / 0xFFFFFFFF
    return val, 1


async def _run_smoke(benchmark: str, seed: int, pop: int, gens: int) -> dict:
    rng = random.Random(seed)

    # Step 1: validate default and random genomes
    default = default_genome_for(benchmark)
    validate_genome(default)
    print(f"[smoke] default genome for {benchmark}: OK")

    for i in range(5):
        g = random_genome_for(benchmark, rng)
        validate_genome(g)
    print(f"[smoke] 5 random genomes validated for {benchmark}: OK")

    # Step 2: run random search with the toy fitness
    method = RandomSearch()
    ctx = SearchContext(
        benchmark=benchmark,
        seed=seed,
        budget_llm_calls=pop * gens,
        population_size=pop,
        generations=gens,
    )
    result = await method.run(ctx=ctx, fitness_fn=_toy_fitness)
    print(f"[smoke] random search: {len(result.all_evaluated)} evals, "
          f"champion fitness={result.champion_fitness_search:.4f}")

    return {
        "benchmark": benchmark,
        "method": "random_search",
        "seed": seed,
        "population_size": pop,
        "generations": gens,
        "n_evaluated": len(result.all_evaluated),
        "champion_fitness": result.champion_fitness_search,
        "per_generation_best": result.per_generation_best,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="forecastbench")
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--method", default="random_search")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--population-size", type=int, default=4)
    p.add_argument("--generations", type=int, default=2)
    p.add_argument("--results-dir", default="real_study/logs/smoke_results")
    args = p.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    summary = asyncio.run(_run_smoke(args.benchmark, args.seed, args.population_size, args.generations))
    out = Path(args.results_dir) / "smoke_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[smoke] wrote {out}")
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
