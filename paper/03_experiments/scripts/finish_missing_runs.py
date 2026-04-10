"""Complete the missing prompt_only_evolution seeds from the v2 experiment."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from breed.datasets import load_builtin_questions, time_based_split
from breed.experiment import ExperimentConfig, ExperimentResult, ExperimentRunner, RunResult
from synthetic_agent import make_synthetic_agent
from run_main_experiment import GENE_TEMPLATE, brier_fitness_scorer, questions_to_tasks


async def main():
    output_dir = Path("paper/04_results/logs/main_experiment_v2")
    runs_dir = output_dir / "runs"

    full = load_builtin_questions()
    train_ds, test_ds = time_based_split(full, train_cutoff="2024-06-01")
    truth_lookup = {q.prompt: q.outcome for q in full}
    agent_fn = make_synthetic_agent(truth_lookup, noise_level=0.05)
    train_tasks = questions_to_tasks(train_ds)
    test_tasks = questions_to_tasks(test_ds)

    cfg = ExperimentConfig(
        name="main_experiment",
        seeds=[1, 2, 3, 4, 5, 6, 7, 8],
        population_size=16,
        generations=12,
        elite_count=4,
        mutation_rate=0.20,
        immigration_rate=0.10,
        selection_method="tournament",
        tournament_size=3,
        output_dir=str(output_dir),
        methods=[
            "full_evolution", "mutation_only", "crossover_only",
            "random_search", "static_best", "static_ensemble",
            "prompt_only_evolution",
        ],
    )

    runner = ExperimentRunner(
        config=cfg, agent_fn=agent_fn,
        train_tasks=train_tasks, test_tasks=test_tasks,
        scorer=brier_fitness_scorer, gene_template=GENE_TEMPLATE,
    )

    # Run only the missing prompt_only_evolution seeds 3-8
    missing = [("prompt_only_evolution", s) for s in [3, 4, 5, 6, 7, 8]]
    for method, seed in missing:
        inter_path = runs_dir / f"{method}__seed{seed}.json"
        if inter_path.exists():
            print(f"  skip {method} seed={seed}")
            continue
        t0 = time.monotonic()
        r = await runner.run_method(method, seed)
        inter_path.parent.mkdir(parents=True, exist_ok=True)
        inter_path.write_text(json.dumps(r.to_dict()), encoding="utf-8")
        print(
            f"  [finish] {method} seed={seed} done in {time.monotonic()-t0:.2f}s "
            f"train={r.champion_fitness:.4f}",
            flush=True,
        )

    # Aggregate all intermediate files into a results.jsonl
    all_runs: list[RunResult] = []
    for p in sorted(runs_dir.glob("*.json")):
        data = json.loads(p.read_text())
        all_runs.append(RunResult.from_dict(data))
    print(f"\nAggregated {len(all_runs)} runs.")

    result = ExperimentResult(config=cfg, runs=all_runs)
    out_path = output_dir / "results.jsonl"
    result.to_jsonl(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
