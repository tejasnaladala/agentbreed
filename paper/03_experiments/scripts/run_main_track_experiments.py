"""Main-track experiment driver.

Runs all five preregistered experiments:

- E1: 9-method matrix x 3 domains x 20 seeds (primary evidence)
- E_Decisive: K in {1,2,3,5,7,9} dimensionality sweep (centerpiece)
- E2: gene dropout + mutation rate ablations
- E3: Sobol epistasis decomposition
- E_Transfer: champion transfer matrix across domains

Writes results to paper/04_results/{experiment}/ as JSONL and JSON.
Fully deterministic given seeds.

Usage:
    python paper/03_experiments/scripts/run_main_track_experiments.py [--which all|E1|...]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import yaml

from breed.arenas.base import Task
from breed.datasets import load_builtin_questions, time_based_split
from breed.experiment import ExperimentConfig, ExperimentResult, ExperimentRunner, RunResult
from multi_domain_agent import (
    accuracy_scorer,
    build_coding_tasks,
    build_knowledge_tasks,
    forecasting_scorer,
    make_coding_agent,
    make_forecasting_agent,
    make_knowledge_agent,
)


# ---------------------------------------------------------------------------
# Gene template loading
# ---------------------------------------------------------------------------

def load_gene_template_v1() -> dict[str, dict[str, Any]]:
    """Load the locked gene template from the YAML config."""
    path = Path("paper/03_experiments/configs/gene_template_v1.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Gene template v1 not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    template: dict[str, dict[str, Any]] = {}
    for name, spec in raw["genes"].items():
        parsed: dict[str, Any] = dict(spec)
        if parsed.get("type") == "float" and "range" in parsed:
            parsed["range"] = tuple(parsed["range"])
        template[name] = parsed
    return template


GENE_TEMPLATE_V1 = load_gene_template_v1()


# ---------------------------------------------------------------------------
# Domain setup helpers
# ---------------------------------------------------------------------------

def build_forecasting_domain():
    ds = load_builtin_questions()
    train, test = time_based_split(ds, train_cutoff="2024-06-01")
    train_tasks = [
        Task(task_id=q.question_id, prompt=q.prompt, expected=q.outcome)
        for q in train
    ]
    test_tasks = [
        Task(task_id=q.question_id, prompt=q.prompt, expected=q.outcome)
        for q in test
    ]
    truth = {q.prompt: q.outcome for q in ds}
    agent = make_forecasting_agent(truth, noise_level=0.05)
    return agent, train_tasks, test_tasks, forecasting_scorer


def build_coding_domain():
    # 40 train, 40 test (non-overlapping ids)
    train_tasks = build_coding_tasks(40, prefix="code-train")
    test_tasks = build_coding_tasks(40, prefix="code-test")
    agent = make_coding_agent(n_test_cases_per_task=5)
    def scorer(output, expected):
        try:
            return max(0.0, min(1.0, float(output.strip())))
        except (ValueError, AttributeError):
            return 0.0
    return agent, train_tasks, test_tasks, scorer


def build_knowledge_domain():
    train_tasks = build_knowledge_tasks(40, prefix="know-train")
    test_tasks = build_knowledge_tasks(40, prefix="know-test")
    agent = make_knowledge_agent()
    return agent, train_tasks, test_tasks, accuracy_scorer


DOMAINS = {
    "forecasting": build_forecasting_domain,
    "coding": build_coding_domain,
    "knowledge": build_knowledge_domain,
}


# ---------------------------------------------------------------------------
# Generic runner for a method / seed / domain triple
# ---------------------------------------------------------------------------

async def run_one(
    method: str,
    seed: int,
    domain: str,
    population_size: int = 20,
    generations: int = 15,
    gene_template: dict | None = None,
    verbose: bool = True,
) -> RunResult:
    agent, train_tasks, test_tasks, scorer = DOMAINS[domain]()
    cfg = ExperimentConfig(
        name=f"{domain}_{method}_s{seed}",
        seeds=[seed],
        population_size=population_size,
        generations=generations,
        elite_count=4,
        mutation_rate=0.20,
        immigration_rate=0.10,
        selection_method="tournament",
        tournament_size=3,
        output_dir=f"paper/04_results/_scratch/{domain}",
        methods=[method],
    )
    runner = ExperimentRunner(
        config=cfg, agent_fn=agent,
        train_tasks=train_tasks, test_tasks=test_tasks,
        scorer=scorer, gene_template=gene_template or GENE_TEMPLATE_V1,
    )
    return await runner.run_method(method, seed)


# ---------------------------------------------------------------------------
# Experiment E1: primary matrix
# ---------------------------------------------------------------------------

async def run_E1(
    domains: list[str] = None,
    methods: list[str] = None,
    seeds: list[int] = None,
    population_size: int = 20,
    generations: int = 15,
    output_dir: str = "paper/04_results/E1",
) -> None:
    domains = domains or ["forecasting", "coding", "knowledge"]
    methods = methods or [
        "full_evolution",
        "mutation_only",
        "crossover_only",
        "random_search",
        "static_best",
        "static_ensemble",
        "prompt_only_evolution",
        "bayesian_opt",
        "successive_halving",
        "coordinate_descent",
    ]
    seeds = seeds or list(range(1, 21))  # 20 seeds

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    runs_dir = out_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    total = len(domains) * len(methods) * len(seeds)
    print(f"\n[E1] Running {total} runs: {len(domains)} domains x "
          f"{len(methods)} methods x {len(seeds)} seeds")
    print(f"[E1] Output: {out_path}")

    t_start = time.monotonic()
    completed = 0
    runs: list[dict] = []

    for domain in domains:
        for method in methods:
            for seed in seeds:
                inter_path = runs_dir / f"{domain}__{method}__seed{seed}.json"
                if inter_path.exists():
                    data = json.loads(inter_path.read_text())
                    runs.append({"domain": domain, **data})
                    completed += 1
                    continue
                t0 = time.monotonic()
                r = await run_one(
                    method, seed, domain,
                    population_size=population_size,
                    generations=generations,
                )
                runs.append({"domain": domain, **r.to_dict()})
                inter_path.write_text(json.dumps(r.to_dict()), encoding="utf-8")
                completed += 1
                dt = time.monotonic() - t0
                test_mean = (
                    sum(r.per_question_scores) / max(len(r.per_question_scores), 1)
                    if r.per_question_scores else 0.0
                )
                print(
                    f"  [{completed:4d}/{total}] {domain:11} {method:22} "
                    f"seed={seed:2d} train={r.champion_fitness:.4f} "
                    f"test={test_mean:.4f} ({dt:5.1f}s)",
                    flush=True,
                )

    elapsed = time.monotonic() - t_start
    print(f"[E1] Done in {elapsed:.1f}s")

    # Aggregate JSONL
    agg_path = out_path / "E1_results.jsonl"
    with agg_path.open("w", encoding="utf-8") as f:
        header = {
            "__header__": True,
            "experiment": "E1",
            "domains": domains,
            "methods": methods,
            "seeds": seeds,
            "population_size": population_size,
            "generations": generations,
        }
        f.write(json.dumps(header) + "\n")
        for run in runs:
            f.write(json.dumps(run) + "\n")
    print(f"[E1] Aggregated -> {agg_path}")


# ---------------------------------------------------------------------------
# Experiment E_Decisive: dimensionality sweep
# ---------------------------------------------------------------------------

async def run_E_decisive(
    K_values: list[int] = None,
    domains: list[str] = None,
    methods: list[str] = None,
    seeds: list[int] = None,
    output_dir: str = "paper/04_results/E_decisive",
) -> None:
    K_values = K_values or [1, 2, 3, 5, 7, 9]
    domains = domains or ["forecasting", "coding"]
    methods = methods or ["full_evolution", "random_search"]
    seeds = seeds or list(range(1, 21))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    runs_dir = out_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Fixed gene-subset order: pick the first K genes from the full template
    all_genes = list(GENE_TEMPLATE_V1.keys())
    total = len(K_values) * len(domains) * len(methods) * len(seeds)
    print(f"\n[E_decisive] Running {total} runs: "
          f"K in {K_values} x {len(domains)} domains x {len(methods)} methods "
          f"x {len(seeds)} seeds")

    t_start = time.monotonic()
    completed = 0
    runs: list[dict] = []

    for K in K_values:
        if K > len(all_genes):
            print(f"  Skip K={K}: only {len(all_genes)} genes available")
            continue
        # Always include prompt_template (the most impactful gene) as the first
        # axis, then add in order until K genes are reached.
        subset_names = ["prompt_template"]
        for g in all_genes:
            if g not in subset_names and len(subset_names) < K:
                subset_names.append(g)
        sub_template = {name: GENE_TEMPLATE_V1[name] for name in subset_names}

        for domain in domains:
            for method in methods:
                for seed in seeds:
                    inter_path = runs_dir / f"K{K}__{domain}__{method}__seed{seed}.json"
                    if inter_path.exists():
                        data = json.loads(inter_path.read_text())
                        runs.append({
                            "K": K, "domain": domain, "subset": subset_names, **data
                        })
                        completed += 1
                        continue
                    t0 = time.monotonic()
                    r = await run_one(
                        method, seed, domain,
                        gene_template=sub_template,
                    )
                    record = {
                        "K": K, "domain": domain, "subset": subset_names,
                        **r.to_dict(),
                    }
                    runs.append(record)
                    inter_path.write_text(json.dumps(record), encoding="utf-8")
                    completed += 1
                    dt = time.monotonic() - t0
                    test_mean = (
                        sum(r.per_question_scores) / max(len(r.per_question_scores), 1)
                        if r.per_question_scores else 0.0
                    )
                    print(
                        f"  [{completed:4d}/{total}] K={K} {domain:11} "
                        f"{method:16} seed={seed:2d} test={test_mean:.4f} "
                        f"({dt:4.1f}s)",
                        flush=True,
                    )

    elapsed = time.monotonic() - t_start
    print(f"[E_decisive] Done in {elapsed:.1f}s")

    agg_path = out_path / "E_decisive_results.jsonl"
    with agg_path.open("w", encoding="utf-8") as f:
        header = {
            "__header__": True,
            "experiment": "E_decisive",
            "K_values": K_values,
            "domains": domains,
            "methods": methods,
            "seeds": seeds,
        }
        f.write(json.dumps(header) + "\n")
        for run in runs:
            f.write(json.dumps(run) + "\n")
    print(f"[E_decisive] Aggregated -> {agg_path}")


# ---------------------------------------------------------------------------
# Experiment E2: ablations (gene dropout + mutation rate)
# ---------------------------------------------------------------------------

async def run_E2(
    output_dir: str = "paper/04_results/E2",
) -> None:
    """Gene-dropout ablation: freeze k random genes and see how much fitness drops."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    runs_dir = out_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Gene dropout: drop k = 0, 1, 3, 5, 7 (keep 9 - k active)
    # To "drop" a gene we freeze it at a fixed baseline by omitting it from the template.
    dropout_ks = [0, 1, 3, 5, 7]
    domains = ["forecasting", "coding"]
    method = "full_evolution"
    seeds = list(range(1, 11))  # 10 seeds for ablations

    all_genes = list(GENE_TEMPLATE_V1.keys())
    total = len(dropout_ks) * len(domains) * len(seeds)
    print(f"\n[E2] Gene dropout: {total} runs")
    t_start = time.monotonic()
    completed = 0
    runs: list[dict] = []

    for k in dropout_ks:
        # Drop the last k genes from all_genes (deterministic ordering)
        active = all_genes[: len(all_genes) - k] if k > 0 else all_genes
        sub_template = {name: GENE_TEMPLATE_V1[name] for name in active}
        for domain in domains:
            for seed in seeds:
                inter_path = runs_dir / f"dropout{k}__{domain}__seed{seed}.json"
                if inter_path.exists():
                    data = json.loads(inter_path.read_text())
                    runs.append({"k_dropped": k, "domain": domain, **data})
                    completed += 1
                    continue
                t0 = time.monotonic()
                r = await run_one(
                    method, seed, domain, gene_template=sub_template,
                )
                record = {"k_dropped": k, "domain": domain, **r.to_dict()}
                runs.append(record)
                inter_path.write_text(json.dumps(record), encoding="utf-8")
                completed += 1
                dt = time.monotonic() - t0
                test_mean = (
                    sum(r.per_question_scores) / max(len(r.per_question_scores), 1)
                    if r.per_question_scores else 0.0
                )
                print(
                    f"  [{completed:3d}/{total}] dropout={k} {domain:11} "
                    f"seed={seed:2d} test={test_mean:.4f} ({dt:4.1f}s)",
                    flush=True,
                )

    elapsed = time.monotonic() - t_start
    print(f"[E2] Done in {elapsed:.1f}s")

    agg_path = out_path / "E2_results.jsonl"
    with agg_path.open("w", encoding="utf-8") as f:
        header = {
            "__header__": True,
            "experiment": "E2_gene_dropout",
            "k_values": dropout_ks,
            "domains": domains,
            "method": method,
            "seeds": seeds,
        }
        f.write(json.dumps(header) + "\n")
        for run in runs:
            f.write(json.dumps(run) + "\n")
    print(f"[E2] Aggregated -> {agg_path}")


# ---------------------------------------------------------------------------
# Experiment E3: Sobol epistasis decomposition
# ---------------------------------------------------------------------------

async def run_E3(
    n_base: int = 256,
    output_dir: str = "paper/04_results/E3",
) -> None:
    from breed.analysis.epistasis import functional_anova_decomposition
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[E3] Running Sobol epistasis decomposition (n_base={n_base})")

    reports = {}
    for domain in ["forecasting", "coding", "knowledge"]:
        print(f"[E3] Domain: {domain}")
        agent, train_tasks, _, scorer = DOMAINS[domain]()
        t0 = time.monotonic()
        report = await functional_anova_decomposition(
            gene_template=GENE_TEMPLATE_V1,
            agent_fn=agent,
            tasks=train_tasks,
            scorer=scorer,
            n_base=n_base,
            seed=42,
        )
        dt = time.monotonic() - t0
        report_dict = report.to_dict()
        report_dict["domain"] = domain
        report_dict["wall_clock_seconds"] = dt
        reports[domain] = report_dict
        sum_first = report.indices.sum_first_order
        sum_pair = report.indices.sum_pairwise
        print(
            f"  {domain}: sum_first={sum_first:.3f} sum_pairwise={sum_pair:.3f} "
            f"interaction_frac={report.indices.interaction_fraction:.3f} "
            f"H3={'YES' if report.h3_confirmed else 'no'} ({dt:.1f}s)"
        )

    out_file = out_path / "E3_epistasis.json"
    out_file.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print(f"[E3] Saved -> {out_file}")


# ---------------------------------------------------------------------------
# Experiment E_Transfer: champion transfer across domains
# ---------------------------------------------------------------------------

async def run_E_transfer(
    output_dir: str = "paper/04_results/E_transfer",
) -> None:
    """For each (source_domain, target_domain), take the full_evolution
    champion from source and evaluate it on target's test set."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n[E_transfer] Building champions from E1 on each domain")

    # Reuse E1 results if they exist
    champs: dict[str, dict] = {}
    e1_file = Path("paper/04_results/E1/E1_results.jsonl")
    if e1_file.exists():
        print("[E_transfer] Loading champions from E1 results...")
        for line in e1_file.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("__header__"):
                continue
            if rec["method"] == "full_evolution":
                domain = rec["domain"]
                # Keep the one with best train fitness per domain
                if domain not in champs or rec["champion_fitness"] > champs[domain]["champion_fitness"]:
                    champs[domain] = rec
    else:
        print("[E_transfer] No E1 results; running fresh champions on each domain")
        for domain in ["forecasting", "coding", "knowledge"]:
            r = await run_one("full_evolution", seed=1, domain=domain)
            champs[domain] = {"domain": domain, **r.to_dict()}

    # Transfer matrix
    print("\n[E_transfer] Evaluating champions on each target domain test set")
    transfer_matrix: dict[str, dict[str, float]] = {}
    for src_domain, champ in champs.items():
        transfer_matrix[src_domain] = {}
        # Reconstruct the champion genome
        from breed.genome import Gene, GeneType, Genome

        champ_genes_raw = champ["metadata"]["champion_genes"]
        # Build full genes dict using template metadata
        reconstructed_genes: dict[str, Gene] = {}
        for name, value in champ_genes_raw.items():
            spec = GENE_TEMPLATE_V1.get(name, {})
            gtype_str = str(spec.get("type", "")).lower()
            if gtype_str == "enum":
                reconstructed_genes[name] = Gene(
                    name=name, type=GeneType.ENUM, value=value,
                    options=list(spec.get("options", [])),
                )
            elif gtype_str == "float":
                reconstructed_genes[name] = Gene(
                    name=name, type=GeneType.FLOAT, value=value,
                    range=tuple(spec.get("range", (0.0, 1.0))),
                )
            elif gtype_str == "set":
                reconstructed_genes[name] = Gene(
                    name=name, type=GeneType.SET, value=value,
                    available=list(spec.get("available", [])),
                )

        champ_genome = Genome(
            genome_id=f"transfer-champ-{src_domain}",
            generation=0, parents=[], genes=reconstructed_genes,
        )

        for tgt_domain in ["forecasting", "coding", "knowledge"]:
            agent, _, test_tasks, scorer = DOMAINS[tgt_domain]()
            total = 0.0
            for task in test_tasks:
                out = await agent(champ_genome.to_dict(), task.prompt)
                total += scorer(out, task.expected)
            mean_score = total / max(len(test_tasks), 1)
            transfer_matrix[src_domain][tgt_domain] = mean_score
            marker = "*" if src_domain == tgt_domain else " "
            print(
                f"  {marker} {src_domain:11} -> {tgt_domain:11}: {mean_score:.4f}",
                flush=True,
            )

    out_file = out_path / "E_transfer_results.json"
    out_file.write_text(
        json.dumps({"transfer_matrix": transfer_matrix}, indent=2),
        encoding="utf-8",
    )
    print(f"[E_transfer] Saved -> {out_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", default="all",
                        choices=["all", "E1", "E_decisive", "E2", "E3", "E_transfer"])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--gens", type=int, default=15)
    parser.add_argument("--sobol-n-base", type=int, default=256)
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))
    t_all = time.monotonic()

    if args.which in ("all", "E1"):
        await run_E1(seeds=seeds, population_size=args.pop, generations=args.gens)
    if args.which in ("all", "E_decisive"):
        await run_E_decisive(seeds=seeds)
    if args.which in ("all", "E2"):
        await run_E2()
    if args.which in ("all", "E3"):
        await run_E3(n_base=args.sobol_n_base)
    if args.which in ("all", "E_transfer"):
        await run_E_transfer()

    print(f"\nAll experiments done in {time.monotonic() - t_all:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
