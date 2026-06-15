"""Main-track analysis pipeline.

Loads all per-run JSON files from paper/04_results/{E1,E_decisive,E2},
computes aggregated statistics, runs paired bootstrap tests with Holm
correction, generates publication-quality figures, and writes markdown
tables for paper.md.

Outputs:
- paper/04_results/figures/main_track/*.pdf
- paper/04_results/tables/main_track/*.md
- paper/04_results/analysis_main_track.json  (full stats)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from breed.paper_figures import (
    METHOD_LABELS,
    aggregate_scores_across_seeds,
    fig_ablation_bars,
)
from breed.stats import bootstrap_ci, holm_bonferroni, paired_t_test


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_runs(runs_dir: Path) -> list[dict]:
    records: list[dict] = []
    if not runs_dir.exists():
        return records
    for p in sorted(runs_dir.glob("*.json")):
        try:
            records.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def _test_mean(rec: dict) -> float:
    scores = rec.get("per_question_scores") or []
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def _seed(rec: dict) -> int:
    return int(rec.get("seed", -1))


# ---------------------------------------------------------------------------
# Per-method aggregation on E1
# ---------------------------------------------------------------------------


def aggregate_e1(runs_dir: Path) -> tuple[dict[str, dict], list[dict]]:
    """Return (per_method_stats, raw_records).

    per_method_stats maps (domain, method) -> aggregated stats dict.
    """
    records = _load_runs(runs_dir)
    by_key: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        domain = r.get("domain", "unknown")
        method = r.get("method", "unknown")
        by_key.setdefault((domain, method), []).append(r)

    stats: dict[str, dict] = {}
    for (domain, method), rs in sorted(by_key.items()):
        rs_sorted = sorted(rs, key=_seed)
        test_scores = [_test_mean(r) for r in rs_sorted]
        train_scores = [float(r.get("champion_fitness", 0.0)) for r in rs_sorted]
        wall = [float(r.get("wall_clock_seconds", 0.0)) for r in rs_sorted]
        if not test_scores:
            continue
        ci = bootstrap_ci(test_scores, n_bootstrap=10_000, seed=42)
        stats[f"{domain}__{method}"] = {
            "domain": domain,
            "method": method,
            "n_seeds": len(test_scores),
            "train_mean": float(np.mean(train_scores)),
            "train_std": float(np.std(train_scores, ddof=1)) if len(train_scores) > 1 else 0.0,
            "test_mean": float(np.mean(test_scores)),
            "test_std": float(np.std(test_scores, ddof=1)) if len(test_scores) > 1 else 0.0,
            "test_ci_low": ci.lower,
            "test_ci_high": ci.upper,
            "wall_mean": float(np.mean(wall)),
            "per_seed_test_scores": test_scores,
        }
    return stats, records


# ---------------------------------------------------------------------------
# Paired comparisons (Holm-corrected)
# ---------------------------------------------------------------------------


def paired_compare(a: list[float], b: list[float]) -> dict:
    if len(a) != len(b) or len(a) < 2:
        return {"insufficient": True, "n": len(a)}
    t = paired_t_test(a, b)
    diff = np.asarray(a) - np.asarray(b)
    ci = bootstrap_ci(diff.tolist(), n_bootstrap=10_000, seed=42)
    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "diff": float(np.mean(diff)),
        "ci_low": ci.lower,
        "ci_high": ci.upper,
        "t": t.statistic,
        "p_raw": t.p_value,
        "d_z": t.effect_size,
        "n": t.n,
    }


def compare_against_primary(
    e1_stats: dict[str, dict],
    domain: str,
    primary: str = "full_evolution",
    baselines: list[str] | None = None,
) -> list[dict]:
    baselines = baselines or [
        "mutation_only", "crossover_only", "random_search",
        "static_best", "static_ensemble", "prompt_only_evolution",
        "bayesian_opt", "successive_halving", "coordinate_descent",
    ]
    primary_key = f"{domain}__{primary}"
    if primary_key not in e1_stats:
        return []
    p_scores = e1_stats[primary_key]["per_seed_test_scores"]
    comparisons: list[dict] = []
    p_values: list[float] = []
    for base in baselines:
        key = f"{domain}__{base}"
        if key not in e1_stats:
            continue
        b_scores = e1_stats[key]["per_seed_test_scores"]
        cmp = paired_compare(p_scores, b_scores)
        cmp["baseline"] = base
        cmp["domain"] = domain
        comparisons.append(cmp)
        p_values.append(cmp.get("p_raw", 1.0))
    # Holm-Bonferroni
    holm_flags = holm_bonferroni(p_values)
    # Holm-adjusted p-values
    n = len(p_values)
    if n > 0:
        sorted_idx = np.argsort(p_values)
        adj = np.zeros(n)
        running_max = 0.0
        for k, idx in enumerate(sorted_idx):
            adjusted = (n - k) * p_values[idx]
            running_max = max(running_max, adjusted)
            adj[idx] = min(running_max, 1.0)
        for i, cmp in enumerate(comparisons):
            cmp["p_holm"] = float(adj[i])
            cmp["significant_holm"] = bool(holm_flags[i])
    return comparisons


# ---------------------------------------------------------------------------
# Mixed-effects-style aggregate using bootstrap over (domain, seed)
# ---------------------------------------------------------------------------


def aggregate_across_domains(
    e1_stats: dict[str, dict],
    domains: list[str],
    method_a: str,
    method_b: str,
) -> dict:
    """Paired bootstrap of (method_a - method_b) across (domain, seed) pairs."""
    deltas: list[float] = []
    for d in domains:
        a_key = f"{d}__{method_a}"
        b_key = f"{d}__{method_b}"
        if a_key not in e1_stats or b_key not in e1_stats:
            continue
        a = np.asarray(e1_stats[a_key]["per_seed_test_scores"])
        b = np.asarray(e1_stats[b_key]["per_seed_test_scores"])
        n = min(len(a), len(b))
        for i in range(n):
            deltas.append(float(a[i] - b[i]))
    if len(deltas) < 2:
        return {"insufficient": True, "n": len(deltas)}
    ci = bootstrap_ci(deltas, n_bootstrap=10_000, seed=42)
    from scipy import stats as sstats

    arr = np.asarray(deltas)
    t_stat, p_val = sstats.ttest_1samp(arr, 0.0)
    d = float(arr.mean() / arr.std(ddof=1)) if arr.std(ddof=1) > 0 else 0.0
    return {
        "method_a": method_a,
        "method_b": method_b,
        "n_pairs": len(deltas),
        "mean_diff": float(arr.mean()),
        "ci_low": ci.lower,
        "ci_high": ci.upper,
        "t": float(t_stat),
        "p_raw": float(p_val),
        "d_z": d,
    }


# ---------------------------------------------------------------------------
# E_decisive aggregation
# ---------------------------------------------------------------------------


def aggregate_e_decisive(runs_dir: Path) -> dict[tuple[int, str, str], dict]:
    """Return (K, domain, method) -> per-seed test scores list + stats."""
    records = _load_runs(runs_dir)
    by_key: dict[tuple[int, str, str], list[dict]] = {}
    for r in records:
        K = int(r.get("K", -1))
        domain = r.get("domain", "unknown")
        method = r.get("method", "unknown")
        by_key.setdefault((K, domain, method), []).append(r)
    stats: dict[tuple[int, str, str], dict] = {}
    for key, rs in sorted(by_key.items()):
        rs_sorted = sorted(rs, key=_seed)
        scores = [_test_mean(r) for r in rs_sorted]
        train = [float(r.get("champion_fitness", 0.0)) for r in rs_sorted]
        if not scores:
            continue
        ci = bootstrap_ci(scores, n_bootstrap=5_000, seed=42)
        stats[key] = {
            "K": key[0],
            "domain": key[1],
            "method": key[2],
            "n_seeds": len(scores),
            "mean_test": float(np.mean(scores)),
            "std_test": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "ci_low": ci.lower,
            "ci_high": ci.upper,
            "mean_train": float(np.mean(train)),
            "per_seed_test_scores": scores,
        }
    return stats


# ---------------------------------------------------------------------------
# E2 dropout
# ---------------------------------------------------------------------------


def aggregate_e2(runs_dir: Path) -> dict[tuple[int, str], dict]:
    records = _load_runs(runs_dir)
    by_key: dict[tuple[int, str], list[dict]] = {}
    for r in records:
        k_dropped = int(r.get("k_dropped", -1))
        domain = r.get("domain", "unknown")
        by_key.setdefault((k_dropped, domain), []).append(r)
    stats: dict[tuple[int, str], dict] = {}
    for key, rs in sorted(by_key.items()):
        scores = [_test_mean(r) for r in rs]
        if not scores:
            continue
        stats[key] = {
            "k_dropped": key[0],
            "domain": key[1],
            "n_seeds": len(scores),
            "mean_test": float(np.mean(scores)),
            "std_test": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "per_seed_test_scores": scores,
        }
    return stats


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def make_figures(
    e1_stats: dict[str, dict],
    e_decisive_stats: dict[tuple[int, str, str], dict],
    e2_stats: dict[tuple[int, str], dict],
    e3_reports: dict,
    transfer_matrix: dict,
    fig_dir: Path,
) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    domains = sorted({v["domain"] for v in e1_stats.values()})
    all_methods = [
        "full_evolution", "mutation_only", "crossover_only",
        "random_search", "static_best", "static_ensemble",
        "prompt_only_evolution", "bayesian_opt",
        "successive_halving", "coordinate_descent",
    ]

    # ---- Figure: ablation bars per domain ----
    for domain in domains:
        method_scores = []
        for method in all_methods:
            key = f"{domain}__{method}"
            if key not in e1_stats:
                continue
            scores = e1_stats[key]["per_seed_test_scores"]
            method_scores.append(
                aggregate_scores_across_seeds(
                    method=method,
                    per_seed_scores=scores,
                    n_bootstrap=5_000,
                    seed=42,
                )
            )
        if method_scores:
            out = fig_dir / f"fig_ablation_{domain}.pdf"
            fig_ablation_bars(
                method_scores, out,
                title=f"{domain.title()} test fitness by method",
            )
            paths.append(str(out))
            # PNG too
            fig_ablation_bars(method_scores, fig_dir / f"fig_ablation_{domain}.png")

    # ---- Figure: decisive dimensionality sweep per domain ----
    for domain in sorted({k[1] for k in e_decisive_stats}):
        K_values = sorted({k[0] for k in e_decisive_stats if k[1] == domain})
        methods_in_sweep = sorted({k[2] for k in e_decisive_stats if k[1] == domain})
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6.5, 3.5))
        for method in methods_in_sweep:
            x = []
            mean = []
            ci_low = []
            ci_high = []
            for K in K_values:
                stats = e_decisive_stats.get((K, domain, method))
                if not stats:
                    continue
                x.append(K)
                mean.append(stats["mean_test"])
                ci_low.append(stats["ci_low"])
                ci_high.append(stats["ci_high"])
            if x:
                color = {
                    "full_evolution": "#0072B2",
                    "random_search": "#999999",
                }.get(method, "#000000")
                label = METHOD_LABELS.get(method, method)
                plt.plot(x, mean, "o-", color=color, label=label)
                plt.fill_between(x, ci_low, ci_high, color=color, alpha=0.2)
        plt.xlabel("Search space dimensionality K (number of gene axes)")
        plt.ylabel("Test fitness")
        plt.title(f"Decisive dimensionality sweep -- {domain}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="best", frameon=True)
        out = fig_dir / f"fig_decisive_{domain}.pdf"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.savefig(fig_dir / f"fig_decisive_{domain}.png", dpi=300, bbox_inches="tight")
        plt.close()
        paths.append(str(out))

    # ---- Figure: E2 gene dropout ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.5, 3.5))
    for domain in sorted({k[1] for k in e2_stats}):
        xs = sorted({k[0] for k in e2_stats if k[1] == domain})
        ys = [e2_stats[(k, domain)]["mean_test"] for k in xs]
        errs = [e2_stats[(k, domain)]["std_test"] for k in xs]
        plt.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=domain)
    plt.xlabel("Genes dropped from template")
    plt.ylabel("Test fitness (full_evolution)")
    plt.title("Gene dropout ablation (fewer genes -> lower ceiling)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", frameon=True)
    out = fig_dir / "fig_e2_dropout.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir / "fig_e2_dropout.png", dpi=300, bbox_inches="tight")
    plt.close()
    paths.append(str(out))

    # ---- Figure: transfer heatmap ----
    if transfer_matrix:
        plt.figure(figsize=(5.0, 4.0))
        src = list(transfer_matrix.keys())
        tgt = list(transfer_matrix[src[0]].keys()) if src else []
        if src and tgt:
            mat = np.array([[transfer_matrix[s][t] for t in tgt] for s in src])
            im = plt.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=1)
            plt.colorbar(im, label="Test fitness")
            plt.xticks(range(len(tgt)), tgt, rotation=45, ha="right")
            plt.yticks(range(len(src)), src)
            plt.xlabel("Target domain")
            plt.ylabel("Source (trained) domain")
            plt.title("Champion cross-domain transfer")
            for i in range(len(src)):
                for j in range(len(tgt)):
                    plt.text(
                        j, i, f"{mat[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if mat[i, j] < 0.5 else "black",
                    )
            out = fig_dir / "fig_transfer_matrix.pdf"
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.savefig(fig_dir / "fig_transfer_matrix.png", dpi=300, bbox_inches="tight")
            plt.close()
            paths.append(str(out))

    return paths


# ---------------------------------------------------------------------------
# Markdown tables
# ---------------------------------------------------------------------------


def make_e1_summary_table(e1_stats: dict[str, dict]) -> str:
    lines = [
        "| Domain | Method | Train (mean±std) | Test (mean±std) | 95% CI | n |",
        "|---|---|---|---|---|---|",
    ]
    for key, s in sorted(e1_stats.items()):
        label = METHOD_LABELS.get(s["method"], s["method"])
        lines.append(
            f"| {s['domain']} | {label} | "
            f"{s['train_mean']:.4f} ± {s['train_std']:.4f} | "
            f"{s['test_mean']:.4f} ± {s['test_std']:.4f} | "
            f"[{s['test_ci_low']:.4f}, {s['test_ci_high']:.4f}] | "
            f"{s['n_seeds']} |"
        )
    return "\n".join(lines)


def make_comparison_table(comparisons_per_domain: dict[str, list[dict]]) -> str:
    lines = [
        "| Domain | Baseline | Diff | 95% CI | t | p (raw) | p (Holm) | d_z | Sig? |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for domain, comps in sorted(comparisons_per_domain.items()):
        for cmp in comps:
            if cmp.get("insufficient"):
                continue
            label = METHOD_LABELS.get(cmp["baseline"], cmp["baseline"])
            sig = "YES" if cmp.get("significant_holm", False) else "no"
            lines.append(
                f"| {domain} | {label} | "
                f"{cmp['diff']:+.4f} | "
                f"[{cmp['ci_low']:+.4f}, {cmp['ci_high']:+.4f}] | "
                f"{cmp['t']:.3f} | "
                f"{cmp['p_raw']:.4f} | "
                f"{cmp['p_holm']:.4f} | "
                f"{cmp['d_z']:.3f} | "
                f"{sig} |"
            )
    return "\n".join(lines)


def make_decisive_table(e_decisive_stats: dict[tuple[int, str, str], dict]) -> str:
    lines = [
        "| K | Domain | Method | Test (mean±std) | 95% CI | n |",
        "|---|---|---|---|---|---|",
    ]
    for key, s in sorted(e_decisive_stats.items()):
        lines.append(
            f"| {s['K']} | {s['domain']} | {s['method']} | "
            f"{s['mean_test']:.4f} ± {s['std_test']:.4f} | "
            f"[{s['ci_low']:.4f}, {s['ci_high']:.4f}] | "
            f"{s['n_seeds']} |"
        )
    return "\n".join(lines)


def make_transfer_table(transfer_matrix: dict) -> str:
    if not transfer_matrix:
        return ""
    src = list(transfer_matrix.keys())
    tgt = list(transfer_matrix[src[0]].keys())
    lines = ["| Source\\Target | " + " | ".join(tgt) + " |"]
    lines.append("|" + "---|" * (len(tgt) + 1))
    for s in src:
        row = [s]
        for t in tgt:
            v = transfer_matrix[s][t]
            cell = f"**{v:.4f}**" if s == t else f"{v:.4f}"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    results_root = Path("paper/04_results")
    fig_dir = results_root / "figures" / "main_track"
    table_dir = results_root / "tables" / "main_track"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    print("Loading E1 results...", flush=True)
    e1_stats, e1_raw = aggregate_e1(results_root / "E1" / "runs")
    print(f"  {len(e1_raw)} runs across {len(e1_stats)} (domain, method) pairs")

    print("Loading E_decisive results...", flush=True)
    e_decisive_stats = aggregate_e_decisive(results_root / "E_decisive" / "runs")
    print(f"  {len(e_decisive_stats)} (K, domain, method) cells")

    print("Loading E2 results...", flush=True)
    e2_stats = aggregate_e2(results_root / "E2" / "runs")
    print(f"  {len(e2_stats)} (k_dropped, domain) cells")

    print("Loading E3 epistasis...", flush=True)
    e3_file = results_root / "E3" / "E3_epistasis.json"
    e3_reports = json.loads(e3_file.read_text(encoding="utf-8")) if e3_file.exists() else {}

    print("Loading E_transfer...", flush=True)
    transfer_file = results_root / "E_transfer" / "E_transfer_results.json"
    transfer_matrix = {}
    if transfer_file.exists():
        data = json.loads(transfer_file.read_text(encoding="utf-8"))
        transfer_matrix = data.get("transfer_matrix", {})

    # Paired comparisons per domain
    print("Running paired comparisons...", flush=True)
    domains = sorted({v["domain"] for v in e1_stats.values()})
    comparisons_per_domain: dict[str, list[dict]] = {}
    for domain in domains:
        comparisons_per_domain[domain] = compare_against_primary(e1_stats, domain)
        sig_count = sum(
            1 for c in comparisons_per_domain[domain]
            if not c.get("insufficient") and c.get("significant_holm", False)
        )
        print(f"  {domain}: {sig_count} / {len(comparisons_per_domain[domain])} significant")

    # Aggregate cross-domain dimensionality test
    print("Cross-domain full vs prompt_only...", flush=True)
    cross_full_vs_prompt = aggregate_across_domains(
        e1_stats, domains, "full_evolution", "prompt_only_evolution"
    )
    cross_full_vs_mutation = aggregate_across_domains(
        e1_stats, domains, "full_evolution", "mutation_only"
    )
    cross_full_vs_random = aggregate_across_domains(
        e1_stats, domains, "full_evolution", "random_search"
    )
    print(
        f"  full vs prompt_only: diff={cross_full_vs_prompt.get('mean_diff', 0):.4f} "
        f"CI=[{cross_full_vs_prompt.get('ci_low', 0):.4f}, "
        f"{cross_full_vs_prompt.get('ci_high', 0):.4f}] "
        f"d_z={cross_full_vs_prompt.get('d_z', 0):.3f} "
        f"p={cross_full_vs_prompt.get('p_raw', 0):.4g}",
        flush=True,
    )
    print(
        f"  full vs mutation_only: diff={cross_full_vs_mutation.get('mean_diff', 0):.4f} "
        f"CI=[{cross_full_vs_mutation.get('ci_low', 0):.4f}, "
        f"{cross_full_vs_mutation.get('ci_high', 0):.4f}] "
        f"d_z={cross_full_vs_mutation.get('d_z', 0):.3f} "
        f"p={cross_full_vs_mutation.get('p_raw', 0):.4g}",
        flush=True,
    )
    print(
        f"  full vs random_search: diff={cross_full_vs_random.get('mean_diff', 0):.4f} "
        f"CI=[{cross_full_vs_random.get('ci_low', 0):.4f}, "
        f"{cross_full_vs_random.get('ci_high', 0):.4f}] "
        f"d_z={cross_full_vs_random.get('d_z', 0):.3f} "
        f"p={cross_full_vs_random.get('p_raw', 0):.4g}",
        flush=True,
    )

    # Figures
    print("Generating figures...", flush=True)
    fig_paths = make_figures(
        e1_stats, e_decisive_stats, e2_stats, e3_reports, transfer_matrix, fig_dir,
    )
    print(f"  {len(fig_paths)} figures")

    # Tables
    print("Writing tables...", flush=True)
    (table_dir / "table_E1_summary.md").write_text(
        make_e1_summary_table(e1_stats), encoding="utf-8"
    )
    (table_dir / "table_E1_comparisons.md").write_text(
        make_comparison_table(comparisons_per_domain), encoding="utf-8"
    )
    (table_dir / "table_E_decisive.md").write_text(
        make_decisive_table(e_decisive_stats), encoding="utf-8"
    )
    (table_dir / "table_transfer.md").write_text(
        make_transfer_table(transfer_matrix), encoding="utf-8"
    )

    # Full JSON summary
    summary = {
        "e1_stats": e1_stats,
        "e_decisive_stats": {
            f"K{k[0]}__{k[1]}__{k[2]}": v for k, v in e_decisive_stats.items()
        },
        "e2_stats": {f"drop{k[0]}__{k[1]}": v for k, v in e2_stats.items()},
        "e3_epistasis": e3_reports,
        "transfer_matrix": transfer_matrix,
        "comparisons_per_domain": comparisons_per_domain,
        "cross_domain_tests": {
            "full_vs_prompt_only": cross_full_vs_prompt,
            "full_vs_mutation_only": cross_full_vs_mutation,
            "full_vs_random_search": cross_full_vs_random,
        },
        "figure_paths": fig_paths,
    }
    (results_root / "analysis_main_track.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print("\nDone. Summary -> paper/04_results/analysis_main_track.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
