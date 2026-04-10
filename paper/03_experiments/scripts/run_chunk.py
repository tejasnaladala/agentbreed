"""Run a small chunk of experiments at a time, skipping existing files.

This script is designed to work around a Windows-specific subprocess stall
observed when running ~500 experiments in a single Python process. Each
invocation runs up to ``--max-runs`` fresh experiments, skipping
intermediate files that already exist, then exits cleanly. Call it in a
loop until all runs are done.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from run_main_track_experiments import (  # noqa: E402
    GENE_TEMPLATE_V1,
    run_one,
)


EXPERIMENTS = {
    "E_decisive": {
        "dir": "paper/04_results/E_decisive/runs",
        "K_values": [1, 2, 3, 5, 7, 9],
        "domains": ["forecasting", "coding"],
        "methods": ["full_evolution", "random_search"],
        "seeds": list(range(1, 21)),
    },
    "E1": {
        "dir": "paper/04_results/E1/runs",
        "K_values": [9],  # full template
        "domains": ["forecasting", "coding", "knowledge"],
        "methods": [
            "full_evolution", "mutation_only", "crossover_only",
            "random_search", "static_best", "static_ensemble",
            "prompt_only_evolution", "bayesian_opt",
            "successive_halving", "coordinate_descent",
        ],
        "seeds": list(range(1, 21)),
    },
    "E2": {
        "dir": "paper/04_results/E2/runs",
        "K_values": [9, 8, 6, 4, 2],  # 0, 1, 3, 5, 7 dropped
        "domains": ["forecasting", "coding"],
        "methods": ["full_evolution"],
        "seeds": list(range(1, 11)),
    },
}


def e_decisive_path(runs_dir: Path, K: int, domain: str, method: str, seed: int) -> Path:
    return runs_dir / f"K{K}__{domain}__{method}__seed{seed}.json"


def e1_path(runs_dir: Path, K: int, domain: str, method: str, seed: int) -> Path:
    return runs_dir / f"{domain}__{method}__seed{seed}.json"


def e2_path(runs_dir: Path, K: int, domain: str, method: str, seed: int) -> Path:
    # K is actually n_active genes; k_dropped = 9 - K
    k_dropped = 9 - K
    return runs_dir / f"dropout{k_dropped}__{domain}__seed{seed}.json"


PATH_FUNCTIONS = {
    "E_decisive": e_decisive_path,
    "E1": e1_path,
    "E2": e2_path,
}


def build_subset_template(K: int) -> dict:
    all_genes = list(GENE_TEMPLATE_V1.keys())
    if K >= len(all_genes):
        return GENE_TEMPLATE_V1
    subset_names = ["prompt_template"]
    for g in all_genes:
        if g not in subset_names and len(subset_names) < K:
            subset_names.append(g)
    return {name: GENE_TEMPLATE_V1[name] for name in subset_names}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--max-runs", type=int, default=40)
    args = parser.parse_args()

    spec = EXPERIMENTS[args.experiment]
    runs_dir = Path(spec["dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    path_fn = PATH_FUNCTIONS[args.experiment]

    # Enumerate all expected runs in deterministic order
    todo = []
    for K in spec["K_values"]:
        for domain in spec["domains"]:
            for method in spec["methods"]:
                for seed in spec["seeds"]:
                    p = path_fn(runs_dir, K, domain, method, seed)
                    if not p.exists():
                        todo.append((K, domain, method, seed, p))

    if not todo:
        print(f"[{args.experiment}] All runs already exist.", flush=True)
        return

    print(f"[{args.experiment}] {len(todo)} runs to go, running up to "
          f"{args.max_runs} in this chunk.", flush=True)

    start = time.monotonic()
    completed = 0

    for K, domain, method, seed, inter_path in todo[: args.max_runs]:
        sub_template = build_subset_template(K)
        t0 = time.monotonic()
        try:
            r = await run_one(
                method, seed, domain,
                gene_template=sub_template,
            )
        except Exception as exc:
            print(f"  ERROR on K={K} {domain} {method} seed={seed}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            continue
        dt = time.monotonic() - t0
        test_mean = (
            sum(r.per_question_scores) / max(len(r.per_question_scores), 1)
            if r.per_question_scores else 0.0
        )
        record = {"K": K, "domain": domain, **r.to_dict()}
        if args.experiment == "E2":
            record["k_dropped"] = 9 - K
        inter_path.write_text(json.dumps(record), encoding="utf-8")
        completed += 1
        print(
            f"  [{completed}/{args.max_runs}] K={K} {domain:11} {method:22} "
            f"seed={seed:2d} test={test_mean:.4f} ({dt:4.1f}s)",
            flush=True,
        )

    elapsed = time.monotonic() - start
    print(f"[{args.experiment}] Chunk done: {completed} runs in {elapsed:.1f}s",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
