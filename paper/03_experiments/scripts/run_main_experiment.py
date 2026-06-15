"""Main experiment for the agentbreed paper.

Runs the complete multi-method, multi-seed experiment on the built-in
forecasting dataset using the synthetic agent. Produces:
- A JSONL file of all RunResults at paper/04_results/logs/main_experiment.jsonl
- A summary table at paper/04_results/tables/main_summary.json
- Per-method aggregated curves and scores

This is THE experiment that the paper's main figures are built from.
Run with::

    python paper/03_experiments/scripts/run_main_experiment.py

The synthetic agent has known cross-component interactions, so we can
test whether crossover-based search exploits them more than mutation-only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Make sibling files importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from breed.datasets import load_builtin_questions, time_based_split
from breed.experiment import ExperimentConfig, ExperimentRunner
from breed.arenas.base import Task
from synthetic_agent import make_synthetic_agent


# ---------------------------------------------------------------------------
# Gene template -- the search space the paper actually evolves
# ---------------------------------------------------------------------------

GENE_TEMPLATE = {
    "prompt_template": {
        "type": "enum",
        "options": [
            "superforecaster_decompose",
            "calibrated_bayesian",
            "pro_con_analytical",
            "outside_view_first",
            "market_trader",
        ],
        "mutation_rate": 0.20,
    },
    "decision_heuristic": {
        "type": "enum",
        "options": [
            "expected_value",
            "minimax_regret",
            "recognition_primed",
            "fermi_decomposition",
            "competing_hypotheses",
        ],
        "mutation_rate": 0.15,
    },
    "calibration_rule": {
        "type": "enum",
        "options": [
            "shrink_to_base_rate",
            "tetlock_curve_check",
            "extremeness_aversion",
            "three_interval_check",
        ],
        "mutation_rate": 0.15,
    },
    "decomposition_style": {
        "type": "enum",
        "options": [
            "reference_class",
            "causal_chain",
            "scenario_tree",
            "fermi",
            "analogy",
        ],
        "mutation_rate": 0.15,
    },
    "memory_structure": {
        "type": "enum",
        "options": ["sliding_window", "episodic", "semantic", "graph", "none"],
        "mutation_rate": 0.10,
    },
    "answer_format": {
        "type": "enum",
        "options": [
            "probability_only",
            "probability_with_reasoning",
            "structured_json",
        ],
        "mutation_rate": 0.10,
    },
    "tool_policy": {
        "type": "set",
        "available": [
            "web_search",
            "web_fetch",
            "calculator",
            "code_exec",
            "read",
            "write",
        ],
        "mutation_rate": 0.10,
    },
    "temperature": {
        "type": "float",
        "range": (0.0, 1.5),
        "mutation_rate": 0.10,
    },
    "risk_threshold": {
        "type": "float",
        "range": (0.01, 0.5),
        "mutation_rate": 0.10,
    },
}


def brier_fitness_scorer(output: str, expected) -> float:
    """Convert agent probability output to fitness in [0, 1].

    Higher is better. Uses 1 - Brier so it can be maximized like other
    fitness metrics in breed's API.
    """
    try:
        p = float(output.strip())
    except (ValueError, AttributeError):
        return 0.0
    try:
        y = float(expected)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 - (p - y) ** 2


def questions_to_tasks(questions) -> list[Task]:
    """Convert ForecastingQuestion objects into Task objects for the runner."""
    return [
        Task(
            task_id=q.question_id,
            prompt=q.prompt,
            expected=q.outcome,
            metadata={"category": q.category, "source": q.source},
        )
        for q in questions
    ]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="main_experiment", help="Experiment name")
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--mutation-rate", type=float, default=0.20)
    parser.add_argument("--immigration-rate", type=float, default=0.10)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5])
    parser.add_argument("--methods", nargs="+", default=[
        "full_evolution",
        "mutation_only",
        "crossover_only",
        "random_search",
        "static_best",
        "static_ensemble",
        "prompt_only_evolution",
    ])
    parser.add_argument("--train-cutoff", default="2024-06-01",
                        help="ISO date splitting train (before) from test (after)")
    parser.add_argument("--output-dir", default="paper/04_results/logs/main_experiment")
    args = parser.parse_args()

    # Load and split the dataset
    print("Loading built-in forecasting dataset...")
    full = load_builtin_questions()
    print(f"  Total: {len(full)} questions")
    print(f"  Categories: {full.category_counts()}")
    print(f"  Base rate: {full.base_rate():.3f}")

    train_ds, test_ds = time_based_split(full, train_cutoff=args.train_cutoff)
    print(f"  Train (resolved before {args.train_cutoff}): {len(train_ds)}")
    print(f"  Test  (resolved on/after {args.train_cutoff}): {len(test_ds)}")

    if len(train_ds) < 10 or len(test_ds) < 5:
        print("WARNING: small dataset; consider adjusting train_cutoff")

    # Convert to runner Task format
    train_tasks = questions_to_tasks(train_ds)
    test_tasks = questions_to_tasks(test_ds)

    # Build the synthetic agent (knows the truth via lookup)
    truth_lookup = {q.prompt: q.outcome for q in full}
    agent_fn = make_synthetic_agent(truth_lookup, noise_level=0.05)

    # Configure experiment
    cfg = ExperimentConfig(
        name=args.name,
        seeds=args.seeds,
        population_size=args.population,
        generations=args.generations,
        elite_count=args.elite,
        mutation_rate=args.mutation_rate,
        immigration_rate=args.immigration_rate,
        selection_method="tournament",
        tournament_size=3,
        output_dir=args.output_dir,
        methods=args.methods,
        metadata={
            "synthetic": True,
            "train_cutoff": args.train_cutoff,
            "n_train": len(train_tasks),
            "n_test": len(test_tasks),
            "agent_noise_level": 0.05,
            "fitness_function": "1 - Brier",
        },
    )

    print(f"\nRunning experiment '{cfg.name}'")
    print(f"  Methods: {cfg.methods}")
    print(f"  Seeds: {cfg.seeds}")
    print(f"  Pop x Gen budget: {cfg.population_size} x {cfg.generations} = {cfg.compute_budget}")
    print(f"  Total agent calls (rough): {len(cfg.methods) * len(cfg.seeds) * cfg.compute_budget * (len(train_tasks) + len(test_tasks))}")

    runner = ExperimentRunner(
        config=cfg,
        agent_fn=agent_fn,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        scorer=brier_fitness_scorer,
        gene_template=GENE_TEMPLATE,
    )

    from breed.experiment import ExperimentResult
    runs = []
    start = time.monotonic()
    print("\n[run] starting method/seed loop", flush=True)
    for method in cfg.methods:
        for seed in cfg.seeds:
            t0 = time.monotonic()
            r = await runner.run_method(method, seed)
            runs.append(r)
            inter_path = Path(args.output_dir) / "runs" / f"{method}__seed{seed}.json"
            inter_path.parent.mkdir(parents=True, exist_ok=True)
            inter_path.write_text(json.dumps(r.to_dict()), encoding="utf-8")
            elapsed_run = time.monotonic() - t0
            test_mean = sum(r.per_question_scores) / max(len(r.per_question_scores), 1)
            print(
                f"  [run] {method:<24} seed={seed} done in {elapsed_run:5.2f}s "
                f"train={r.champion_fitness:.4f} test={test_mean:.4f}",
                flush=True,
            )
    result = ExperimentResult(config=cfg, runs=runs)
    elapsed = time.monotonic() - start

    # Save full results
    output_path = Path(args.output_dir) / "results.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_jsonl(output_path)
    print(f"\nFull results saved to: {output_path}")
    print(f"Total runs: {len(result.runs)} (expected: {len(cfg.methods) * len(cfg.seeds)})")
    print(f"Wall clock: {elapsed:.1f}s")

    # Print quick summary
    print("\n" + "=" * 70)
    print("PER-METHOD SUMMARY (mean +/- std across seeds)")
    print("=" * 70)
    print(f"{'Method':<28} {'TrainFit':>10} {'TestFit':>10} {'Wall(s)':>10}")
    print("-" * 70)
    for method in cfg.methods:
        runs = result.runs_for_method(method)
        if not runs:
            continue
        train_fits = [r.champion_fitness for r in runs]
        test_fits = [
            sum(r.per_question_scores) / len(r.per_question_scores)
            if r.per_question_scores else 0.0
            for r in runs
        ]
        walls = [r.wall_clock_seconds for r in runs]
        import statistics
        train_mean = statistics.mean(train_fits)
        train_std = statistics.stdev(train_fits) if len(train_fits) > 1 else 0.0
        test_mean = statistics.mean(test_fits)
        test_std = statistics.stdev(test_fits) if len(test_fits) > 1 else 0.0
        wall_mean = statistics.mean(walls)
        print(
            f"{method:<28} "
            f"{train_mean:.3f}±{train_std:.3f} "
            f"{test_mean:.3f}±{test_std:.3f} "
            f"{wall_mean:>8.1f}"
        )

    print("=" * 70)
    print("\nDone. Use paper/03_experiments/scripts/analyze_results.py to generate figures.")


if __name__ == "__main__":
    asyncio.run(main())
