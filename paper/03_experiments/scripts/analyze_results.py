"""Analyze main experiment results and generate paper figures + tables.

Reads the JSONL produced by run_main_experiment.py and produces:
- Per-method aggregated statistics (mean, std, 95% CI)
- Paired statistical tests (full vs each baseline)
- Holm-Bonferroni multiple-comparison correction
- All paper figures (fitness curves, ablation bars, calibration, lineage stub)
- A markdown summary table for the paper

Outputs land in:
- paper/04_results/figures/  -- PDF + PNG figures
- paper/04_results/tables/   -- markdown + JSON tables
- paper/04_results/logs/main_experiment/analysis_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make breed importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from breed.experiment import ExperimentResult, RunResult
from breed.paper_figures import (
    METHOD_LABELS,
    aggregate_curves_across_seeds,
    aggregate_scores_across_seeds,
    fig_ablation_bars,
    fig_cost_performance,
    fig_fitness_curves,
)
from breed.stats import (
    aggregate_seed_runs,
    bootstrap_ci,
    cohens_d,
    cohens_d_paired,
    holm_bonferroni,
    paired_t_test,
    wilcoxon_signed_rank,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(jsonl_path: Path) -> tuple[dict, list[RunResult]]:
    """Read a results.jsonl file. Returns (header_dict, runs)."""
    header: dict = {}
    runs: list[RunResult] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("__header__"):
                header = obj
                continue
            runs.append(RunResult.from_dict(obj))
    return header, runs


def runs_for_method(runs: list[RunResult], method: str) -> list[RunResult]:
    return [r for r in runs if r.method == method]


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def per_seed_test_means(runs: list[RunResult]) -> list[float]:
    """For each seed, compute the mean of per_question_scores. Returns one value per seed."""
    return [
        sum(r.per_question_scores) / max(len(r.per_question_scores), 1)
        for r in sorted(runs, key=lambda r: r.seed)
    ]


def per_seed_train_fitness(runs: list[RunResult]) -> list[float]:
    return [r.champion_fitness for r in sorted(runs, key=lambda r: r.seed)]


def paired_comparison(
    method_a_runs: list[RunResult],
    method_b_runs: list[RunResult],
) -> dict:
    """Paired comparison of two methods at the seed level (test fitness)."""
    a_by_seed = {r.seed: per_seed_test_means([r])[0] for r in method_a_runs}
    b_by_seed = {r.seed: per_seed_test_means([r])[0] for r in method_b_runs}
    common = sorted(a_by_seed.keys() & b_by_seed.keys())
    a = [a_by_seed[s] for s in common]
    b = [b_by_seed[s] for s in common]

    if len(a) < 2:
        return {"insufficient_seeds": True, "n": len(a)}

    t_result = paired_t_test(a, b)
    try:
        w_result = wilcoxon_signed_rank(a, b)
    except ValueError:
        w_result = None

    diff = np.array(a) - np.array(b)
    diff_ci = bootstrap_ci(diff.tolist(), n_bootstrap=10_000, seed=42)

    return {
        "method_a_mean": float(np.mean(a)),
        "method_b_mean": float(np.mean(b)),
        "mean_diff": float(np.mean(diff)),
        "diff_ci_low": diff_ci.lower,
        "diff_ci_high": diff_ci.upper,
        "paired_t": {
            "t": t_result.statistic,
            "p": t_result.p_value,
            "d_z": t_result.effect_size,
            "n": t_result.n,
            "significant_alpha_05": t_result.significant,
        },
        "wilcoxon": (
            {
                "W": w_result.statistic,
                "p": w_result.p_value,
                "n": w_result.n,
                "significant_alpha_05": w_result.significant,
            }
            if w_result is not None
            else None
        ),
        "n_seeds": len(a),
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def make_all_figures(
    runs: list[RunResult],
    methods_in_order: list[str],
    fig_dir: Path,
) -> dict[str, str]:
    """Generate all paper figures and return paths."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    # ---- fitness curves ----
    method_curves = []
    for method in methods_in_order:
        m_runs = runs_for_method(runs, method)
        if not m_runs:
            continue
        per_seed = [r.fitness_curve for r in sorted(m_runs, key=lambda r: r.seed)]
        # Pad uneven curves to same length
        max_len = max(len(c) for c in per_seed)
        padded = []
        for c in per_seed:
            if len(c) < max_len:
                c = list(c) + [c[-1]] * (max_len - len(c))
            padded.append(c)
        curve = aggregate_curves_across_seeds(
            method=method, per_seed_curves=padded,
            n_bootstrap=500, seed=42,
        )
        method_curves.append(curve)

    if method_curves:
        path = fig_dir / "fig1_fitness_curves.pdf"
        fig_fitness_curves(
            method_curves, path,
            title="Train fitness over generations (mean +/- 95% CI, n=8 seeds)",
        )
        paths["fitness_curves"] = str(path)
        # Also PNG for appendix
        png_path = fig_dir / "fig1_fitness_curves.png"
        fig_fitness_curves(method_curves, png_path)

    # ---- ablation bars (test fitness) ----
    method_scores = []
    for method in methods_in_order:
        m_runs = runs_for_method(runs, method)
        if not m_runs:
            continue
        scores = per_seed_test_means(m_runs)
        method_scores.append(
            aggregate_scores_across_seeds(
                method=method, per_seed_scores=scores,
                n_bootstrap=10_000, seed=42,
            )
        )

    if method_scores:
        path = fig_dir / "fig2_ablation_bars.pdf"
        fig_ablation_bars(
            method_scores, path,
            title="Test fitness by method (mean +/- 95% CI)",
        )
        paths["ablation_bars"] = str(path)
        png_path = fig_dir / "fig2_ablation_bars.png"
        fig_ablation_bars(method_scores, png_path)

    # ---- cost-performance ----
    cost_data: dict[str, tuple[float, float, float, float]] = {}
    for method in methods_in_order:
        m_runs = runs_for_method(runs, method)
        if not m_runs:
            continue
        costs = [r.api_calls for r in m_runs]
        scores = per_seed_test_means(m_runs)
        cost_data[method] = (
            float(np.mean(costs)),
            float(np.mean(scores)),
            float(np.std(costs)) if len(costs) > 1 else 0.0,
            float(np.std(scores)) if len(scores) > 1 else 0.0,
        )
    if cost_data:
        path = fig_dir / "fig3_cost_performance.pdf"
        fig_cost_performance(
            cost_data, path,
            title="Cost vs test fitness (mean across seeds)",
        )
        paths["cost_performance"] = str(path)

    return paths


# ---------------------------------------------------------------------------
# Markdown table generation
# ---------------------------------------------------------------------------

def make_summary_table(runs: list[RunResult], methods_in_order: list[str]) -> str:
    """Generate a markdown table of method results."""
    header = "| Method | Train fit (mean ± std) | Test fit (mean ± std) | Test 95% CI | n seeds | Wall (s) |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]

    for method in methods_in_order:
        m_runs = runs_for_method(runs, method)
        if not m_runs:
            continue
        train_fits = per_seed_train_fitness(m_runs)
        test_fits = per_seed_test_means(m_runs)
        walls = [r.wall_clock_seconds for r in m_runs]

        train_mean = np.mean(train_fits)
        train_std = np.std(train_fits, ddof=1) if len(train_fits) > 1 else 0.0
        test_mean = np.mean(test_fits)
        test_std = np.std(test_fits, ddof=1) if len(test_fits) > 1 else 0.0
        test_ci = bootstrap_ci(test_fits, n_bootstrap=10_000, seed=42)
        wall_mean = np.mean(walls)

        label = METHOD_LABELS.get(method, method)
        lines.append(
            f"| {label} | {train_mean:.4f} ± {train_std:.4f} | "
            f"{test_mean:.4f} ± {test_std:.4f} | "
            f"[{test_ci.lower:.4f}, {test_ci.upper:.4f}] | "
            f"{len(m_runs)} | {wall_mean:.2f} |"
        )

    return "\n".join(lines)


def make_comparison_table(
    runs: list[RunResult],
    primary: str,
    baselines: list[str],
) -> tuple[str, dict]:
    """Paired comparisons of primary method vs each baseline.

    Returns (markdown_table, dict_of_results).
    """
    primary_runs = runs_for_method(runs, primary)
    if not primary_runs:
        return "", {}

    header = "| Baseline | Primary mean | Baseline mean | Diff | 95% CI | t | p (raw) | p (Holm) | d_z | Sig? |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]

    raw_results: dict[str, dict] = {}
    p_values: list[float] = []

    for baseline in baselines:
        baseline_runs = runs_for_method(runs, baseline)
        if not baseline_runs:
            continue
        cmp = paired_comparison(primary_runs, baseline_runs)
        raw_results[baseline] = cmp
        p_values.append(cmp["paired_t"]["p"])

    # Holm-Bonferroni correction
    holm_flags = holm_bonferroni(p_values)
    holm_p_adjusted: list[float] = []
    # Compute adjusted p-values for the table (Holm-style: rank-weighted)
    sorted_idx = np.argsort(p_values)
    n = len(p_values)
    raw_sorted = np.array(p_values)[sorted_idx]
    adj = np.zeros(n)
    running_max = 0.0
    for k, p in enumerate(raw_sorted):
        adjusted = (n - k) * p
        running_max = max(running_max, adjusted)
        adj[sorted_idx[k]] = min(running_max, 1.0)
    holm_p_adjusted = adj.tolist()

    for i, baseline in enumerate([b for b in baselines if b in raw_results]):
        cmp = raw_results[baseline]
        raw_p = cmp["paired_t"]["p"]
        rows.append(
            f"| {METHOD_LABELS.get(baseline, baseline)} | "
            f"{cmp['method_a_mean']:.4f} | "
            f"{cmp['method_b_mean']:.4f} | "
            f"{cmp['mean_diff']:+.4f} | "
            f"[{cmp['diff_ci_low']:+.4f}, {cmp['diff_ci_high']:+.4f}] | "
            f"{cmp['paired_t']['t']:.3f} | "
            f"{raw_p:.4f} | "
            f"{holm_p_adjusted[i]:.4f} | "
            f"{cmp['paired_t']['d_z']:.3f} | "
            f"{'YES' if holm_flags[i] else 'no'} |"
        )

    return "\n".join(rows), raw_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-jsonl",
        default="paper/04_results/logs/main_experiment/results.jsonl",
    )
    parser.add_argument(
        "--fig-dir",
        default="paper/04_results/figures",
    )
    parser.add_argument(
        "--table-dir",
        default="paper/04_results/tables",
    )
    parser.add_argument(
        "--summary-out",
        default="paper/04_results/analysis_summary.json",
    )
    parser.add_argument("--primary-method", default="full_evolution")
    args = parser.parse_args()

    results_path = Path(args.results_jsonl)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run paper/03_experiments/scripts/run_main_experiment.py first.")
        return 1

    print(f"Loading results from {results_path}...")
    header, runs = load_results(results_path)
    print(f"  Loaded {len(runs)} runs")

    methods_in_order = header.get("config", {}).get("methods", [])
    if not methods_in_order:
        methods_in_order = sorted({r.method for r in runs})

    primary = args.primary_method
    baselines = [m for m in methods_in_order if m != primary]

    # ---- Figures ----
    fig_dir = Path(args.fig_dir)
    print(f"Generating figures in {fig_dir}/...")
    fig_paths = make_all_figures(runs, methods_in_order, fig_dir)
    for name, path in fig_paths.items():
        print(f"  {name}: {path}")

    # ---- Tables ----
    table_dir = Path(args.table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    summary_md = make_summary_table(runs, methods_in_order)
    (table_dir / "table1_summary.md").write_text(summary_md, encoding="utf-8")
    print(f"\nTable 1 (per-method summary) saved to {table_dir / 'table1_summary.md'}")
    print()
    print(summary_md)

    comparison_md, comparison_data = make_comparison_table(
        runs, primary=primary, baselines=baselines,
    )
    (table_dir / "table2_paired_comparisons.md").write_text(comparison_md, encoding="utf-8")
    print(f"\nTable 2 (paired comparisons) saved to {table_dir / 'table2_paired_comparisons.md'}")
    print()
    print(comparison_md)

    # ---- JSON summary ----
    summary = {
        "n_runs": len(runs),
        "methods": methods_in_order,
        "primary_method": primary,
        "header": header,
        "comparisons": comparison_data,
        "figure_paths": fig_paths,
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nFull summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
