"""Tiny end-to-end smoke test for the agentbreed harness.

Run inside a Slurm job where a vllm server is already serving a small model
on localhost. The script:

  1. Pings /v1/models.
  2. Makes one LLM call via the VLLMClient.
  3. Validates the genome schema round-trip (validate + hash).
  4. Instantiates a RandomSearch over the tiny-model, runs it with a
     toy deterministic fitness fn (NO real benchmark items) — this tests
     the search loop without burning a real benchmark download.
  5. Writes the result to --out as JSON.

Exit code 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

# Make the repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from real_study.harness.genome import (  # noqa: E402
    BENCHMARKS,
    default_genome_for,
    genome_content_hash,
    random_genome_for,
    validate_genome,
)
from real_study.harness.models.vllm_client import VLLMClient, VLLMConfig  # noqa: E402
from real_study.harness.search.base import SearchContext  # noqa: E402
from real_study.harness.search.random_search import RandomSearch  # noqa: E402


async def _run(args) -> dict:
    report = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "endpoint": args.endpoint,
        "benchmark": args.benchmark,
        "steps": [],
        "status": "running",
    }

    # Step 1: ping the endpoint via the client
    cfg = VLLMConfig(
        model=args.model,
        base_url=args.endpoint,
        max_concurrent_requests=4,
        timeout_s=60.0,
    )
    client = VLLMClient(cfg)

    try:
        # Step 1: one LLM call
        t0 = time.perf_counter()
        result = await client.generate(
            system="You are a helpful assistant.",
            user="Respond with exactly one word: hello",
            max_tokens=16,
            temperature=0.0,
            seed=1,
        )
        elapsed = time.perf_counter() - t0
        report["steps"].append({
            "step": "single_llm_call",
            "ok": result.ok,
            "elapsed_s": round(elapsed, 3),
            "text_preview": result.text[:100] if result.text else None,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "error": result.error,
        })
        if not result.ok:
            report["status"] = "fail"
            return report
        print(f"[smoke] LLM call: ok={result.ok} text={result.text[:60]!r}")

        # Step 2: genome schema round-trip
        default = default_genome_for(args.benchmark)
        validate_genome(default)
        h = genome_content_hash(default)
        report["steps"].append({
            "step": "default_genome",
            "ok": True,
            "hash": h,
        })

        rng = random.Random(42)
        for i in range(10):
            g = random_genome_for(args.benchmark, rng)
            validate_genome(g)
        report["steps"].append({
            "step": "random_genomes",
            "ok": True,
            "count": 10,
        })
        print(f"[smoke] genome round-trip: 10 random genomes validated")

        # Step 3: toy random search (deterministic fitness, no real LLM)
        async def _toy_fitness(genome):
            hh = genome_content_hash(genome)
            return int(hh[:8], 16) / 0xFFFFFFFF, 1

        method = RandomSearch()
        ctx = SearchContext(
            benchmark=args.benchmark,
            seed=1,
            budget_llm_calls=8,
            population_size=4,
            generations=2,
        )
        search_result = await method.run(ctx=ctx, fitness_fn=_toy_fitness)
        report["steps"].append({
            "step": "toy_random_search",
            "ok": True,
            "n_evaluated": len(search_result.all_evaluated),
            "champion_fitness": search_result.champion_fitness_search,
        })
        print(f"[smoke] random search: n_eval={len(search_result.all_evaluated)} "
              f"champion_fitness={search_result.champion_fitness_search:.4f}")

        # Step 4: a second LLM call to verify the client is still alive
        result2 = await client.generate(
            system=None,
            user="Respond with a single digit between 0 and 9.",
            max_tokens=8,
            temperature=0.0,
            seed=2,
        )
        report["steps"].append({
            "step": "second_llm_call",
            "ok": result2.ok,
            "text_preview": result2.text[:100] if result2.text else None,
            "error": result2.error,
        })
        print(f"[smoke] second LLM call: ok={result2.ok}")

        report["status"] = "pass" if result2.ok else "fail"

    finally:
        await client.close()

    report["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--endpoint", required=True, help="vLLM /v1 base URL")
    p.add_argument("--benchmark", default="forecastbench", choices=list(BENCHMARKS))
    p.add_argument("--out", required=True, help="path to write JSON result")
    args = p.parse_args()

    report = asyncio.run(_run(args))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[smoke] wrote {out_path}")
    print(f"[smoke] STATUS: {report['status']}")
    sys.exit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
