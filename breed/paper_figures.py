"""Publication-quality figure generation for the agentbreed paper.

Designed for:
- NeurIPS / ICML / ACL style guidelines (single column, 3.5 inches wide
  for main figures, 7 inches for two-column)
- Vector PDF output for the paper, PNG for the appendix
- Colorblind-friendly palette (Wong 2011)
- Honest error bars (95% CI from multiple seeds)
- No bullshit dark themes -- white background, black text, professional

All figures are produced from the experiment.ExperimentResult / RunResult
data structures, NOT from screen-scraping or live demos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# Guard matplotlib import
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

# Wong 2011 colorblind-safe palette
PALETTE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "gray": "#999999",
}

# Method -> color mapping (consistent across all figures)
METHOD_COLORS = {
    "full_evolution": PALETTE["blue"],
    "mutation_only": PALETTE["orange"],
    "crossover_only": PALETTE["bluish_green"],
    "random_search": PALETTE["gray"],
    "static_best": PALETTE["vermillion"],
    "static_ensemble": PALETTE["reddish_purple"],
    "prompt_only_evolution": PALETTE["sky_blue"],
}

METHOD_LABELS = {
    "full_evolution": "Full evolution (this work)",
    "mutation_only": "Mutation only",
    "crossover_only": "Crossover only",
    "random_search": "Random search",
    "static_best": "Best random init",
    "static_ensemble": "Static ensemble",
    "prompt_only_evolution": "Prompt-only evolution",
}


def _require_matplotlib() -> None:
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for paper figures. "
            "Install with: pip install agentbreed[charts]"
        )


def _setup_paper_style() -> None:
    """Apply publication-quality matplotlib settings."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,  # Editable text in PDF
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# Data containers (decoupled from experiment.py to avoid circular imports)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MethodCurve:
    """Aggregated fitness curves across seeds for a single method."""
    method: str
    generations: list[int]
    mean: list[float]
    sem: list[float]   # standard error of mean
    ci_low: list[float]
    ci_high: list[float]
    n_seeds: int


@dataclass(frozen=True)
class MethodScores:
    """Aggregated test-set scores across seeds for a single method."""
    method: str
    scores_per_seed: list[float]  # one score per seed
    mean: float
    std: float
    sem: float
    ci_low: float
    ci_high: float


# ---------------------------------------------------------------------------
# Figure 1: Fitness over generations (the hero figure)
# ---------------------------------------------------------------------------

def fig_fitness_curves(
    curves: Sequence[MethodCurve],
    output_path: str | Path,
    title: str | None = None,
    ylabel: str = "Fitness (1 - Brier score)",
    width_inches: float = 7.0,
    height_inches: float = 3.5,
) -> str:
    """Plot mean +/- 95% CI fitness curves for multiple methods.

    This is the hero figure for the paper. Each method gets its own
    color (from METHOD_COLORS). Shaded bands show the 95% bootstrap CI
    across seeds.
    """
    _require_matplotlib()
    _setup_paper_style()

    fig, ax = plt.subplots(figsize=(width_inches, height_inches))

    for curve in curves:
        color = METHOD_COLORS.get(curve.method, PALETTE["gray"])
        label = METHOD_LABELS.get(curve.method, curve.method)
        gens = np.asarray(curve.generations)
        mean = np.asarray(curve.mean)
        ci_low = np.asarray(curve.ci_low)
        ci_high = np.asarray(curve.ci_high)

        ax.plot(gens, mean, label=label, color=color, marker="o", markersize=3)
        ax.fill_between(gens, ci_low, ci_high, color=color, alpha=0.18, linewidth=0)

    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=True, edgecolor="black", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


# ---------------------------------------------------------------------------
# Figure 2: Reliability diagram (calibration plot)
# ---------------------------------------------------------------------------

def fig_reliability_diagram(
    methods_data: dict[str, tuple[Sequence[float], Sequence[float]]],
    output_path: str | Path,
    n_bins: int = 10,
    title: str | None = None,
    width_inches: float = 4.0,
    height_inches: float = 4.0,
) -> str:
    """Reliability diagram comparing calibration across methods.

    methods_data: dict of method_name -> (predictions, outcomes).
    Plots the perfect calibration diagonal and each method's binned curve.
    """
    _require_matplotlib()
    _setup_paper_style()
    from breed.stats import reliability_diagram_data

    fig, ax = plt.subplots(figsize=(width_inches, height_inches))

    # Diagonal: perfect calibration
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8,
            label="Perfect calibration", zorder=1)

    for method, (preds, outs) in methods_data.items():
        centers, accs, counts = reliability_diagram_data(preds, outs, n_bins=n_bins)
        # Skip empty bins
        mask = counts > 0
        color = METHOD_COLORS.get(method, PALETTE["gray"])
        label = METHOD_LABELS.get(method, method)
        ax.plot(centers[mask], accs[mask], color=color, marker="o",
                markersize=5, label=label, zorder=2)

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, edgecolor="black", framealpha=0.9, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


# ---------------------------------------------------------------------------
# Figure 3: Ablation bar chart
# ---------------------------------------------------------------------------

def fig_ablation_bars(
    method_scores: Sequence[MethodScores],
    output_path: str | Path,
    title: str | None = None,
    ylabel: str = "Test fitness (1 - Brier score)",
    width_inches: float = 4.5,
    height_inches: float = 3.5,
    sort_by_mean: bool = True,
) -> str:
    """Bar chart of test fitness across methods with 95% CI error bars.

    Each method's bar height is the mean across seeds; error bars show
    the 95% bootstrap CI.
    """
    _require_matplotlib()
    _setup_paper_style()

    if sort_by_mean:
        method_scores = sorted(method_scores, key=lambda m: m.mean, reverse=True)

    methods = [m.method for m in method_scores]
    means = [m.mean for m in method_scores]
    errs_low = [m.mean - m.ci_low for m in method_scores]
    errs_high = [m.ci_high - m.mean for m in method_scores]
    colors = [METHOD_COLORS.get(m, PALETTE["gray"]) for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(width_inches, height_inches))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=[errs_low, errs_high],
                  color=colors, edgecolor="black", linewidth=0.8,
                  capsize=3, error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)

    # Annotate each bar with the mean value
    for i, (m, mean_val) in enumerate(zip(method_scores, means)):
        ax.text(i, mean_val + errs_high[i] + 0.005, f"{mean_val:.3f}",
                ha="center", va="bottom", fontsize=7)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


# ---------------------------------------------------------------------------
# Figure 4: Cost-performance Pareto plot
# ---------------------------------------------------------------------------

def fig_cost_performance(
    method_scores: dict[str, tuple[float, float, float, float]],
    output_path: str | Path,
    title: str | None = None,
    width_inches: float = 4.5,
    height_inches: float = 3.5,
) -> str:
    """Cost-performance scatter: x = total tokens (or evals), y = fitness.

    method_scores: dict of method -> (cost, fitness, cost_ci, fitness_ci).
    Useful for showing that evolution is more sample-efficient than baselines.
    """
    _require_matplotlib()
    _setup_paper_style()

    fig, ax = plt.subplots(figsize=(width_inches, height_inches))

    for method, (cost, fit, cost_err, fit_err) in method_scores.items():
        color = METHOD_COLORS.get(method, PALETTE["gray"])
        label = METHOD_LABELS.get(method, method)
        ax.errorbar(cost, fit, xerr=cost_err, yerr=fit_err,
                    color=color, marker="o", markersize=8,
                    linestyle="None", capsize=3, label=label,
                    markeredgecolor="black", markeredgewidth=0.5)

    ax.set_xlabel("Total evaluations (cost proxy)")
    ax.set_ylabel("Test fitness (1 - Brier score)")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=True, edgecolor="black", framealpha=0.9, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


# ---------------------------------------------------------------------------
# Figure 5: Gene trait persistence heatmap
# ---------------------------------------------------------------------------

def fig_gene_persistence(
    persistence_matrix: np.ndarray,  # shape (n_genes, n_generations)
    gene_names: Sequence[str],
    output_path: str | Path,
    title: str | None = None,
    width_inches: float = 6.0,
    height_inches: float = 3.5,
) -> str:
    """Heatmap showing how strongly each gene's value persists across generations.

    Each row is a gene; each column is a generation. The cell value is
    the fraction of the population that shares the modal gene value at
    that generation.
    """
    _require_matplotlib()
    _setup_paper_style()

    fig, ax = plt.subplots(figsize=(width_inches, height_inches))

    im = ax.imshow(persistence_matrix, aspect="auto", cmap="viridis",
                   vmin=0, vmax=1, origin="lower")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Gene")
    ax.set_yticks(np.arange(len(gene_names)))
    ax.set_yticklabels(gene_names)
    if title:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Modal-value frequency", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


# ---------------------------------------------------------------------------
# Figure 6: Champion lineage tree (compact ASCII for paper, optional viz)
# ---------------------------------------------------------------------------

def fig_lineage_tree(
    lineage_records: Sequence[dict],  # records from LineageTracker.load
    champion_id: str,
    output_path: str | Path,
    title: str | None = None,
    width_inches: float = 6.0,
    height_inches: float = 4.0,
) -> str:
    """Render the ancestry of a champion genome as a tree.

    Walks parent pointers backward from the champion. Nodes are colored
    by fitness (red = low, green = high). The champion path is highlighted.
    """
    _require_matplotlib()
    _setup_paper_style()

    # Build parent lookup
    by_id = {r["genome_id"]: r for r in lineage_records}

    # Walk ancestry from champion
    ancestry: list[dict] = []
    seen: set[str] = set()
    queue: list[str] = [champion_id]
    while queue:
        gid = queue.pop(0)
        if gid in seen:
            continue
        seen.add(gid)
        rec = by_id.get(gid)
        if rec is None:
            continue
        ancestry.append(rec)
        for parent_id in rec.get("parents", []):
            if parent_id not in seen:
                queue.append(parent_id)

    if not ancestry:
        raise ValueError(f"No ancestry found for champion {champion_id}")

    # Layout: x = generation, y = position within generation
    by_gen: dict[int, list[dict]] = {}
    for rec in ancestry:
        gen = rec["generation"]
        by_gen.setdefault(gen, []).append(rec)

    fig, ax = plt.subplots(figsize=(width_inches, height_inches))

    # Position nodes
    positions: dict[str, tuple[float, float]] = {}
    for gen in sorted(by_gen):
        records = by_gen[gen]
        for i, rec in enumerate(records):
            y = i - len(records) / 2 + 0.5
            positions[rec["genome_id"]] = (float(gen), y)

    # Draw edges from parents to children
    for rec in ancestry:
        child_pos = positions[rec["genome_id"]]
        for parent_id in rec.get("parents", []):
            if parent_id in positions:
                parent_pos = positions[parent_id]
                ax.plot([parent_pos[0], child_pos[0]],
                        [parent_pos[1], child_pos[1]],
                        color="gray", linewidth=0.5, alpha=0.5, zorder=1)

    # Draw nodes
    fitnesses = [rec["fitness_score"] for rec in ancestry]
    f_min, f_max = min(fitnesses), max(fitnesses)
    for rec in ancestry:
        x, y = positions[rec["genome_id"]]
        f = rec["fitness_score"]
        norm = (f - f_min) / (f_max - f_min) if f_max > f_min else 0.5
        # Red -> green via viridis
        color = plt.cm.RdYlGn(norm)
        is_champion = rec["genome_id"] == champion_id
        size = 80 if is_champion else 30
        edge = "black" if is_champion else "none"
        edge_width = 1.2 if is_champion else 0.0
        ax.scatter(x, y, s=size, c=[color], edgecolors=edge,
                   linewidths=edge_width, zorder=2)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Lineage position")
    if title:
        ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


# ---------------------------------------------------------------------------
# Helpers for building MethodCurve / MethodScores from raw data
# ---------------------------------------------------------------------------

def aggregate_curves_across_seeds(
    method: str,
    per_seed_curves: Sequence[Sequence[float]],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> MethodCurve:
    """Aggregate per-seed fitness curves into mean +/- bootstrap CI."""
    from breed.stats import bootstrap_ci

    arr = np.asarray(per_seed_curves)  # shape (n_seeds, n_generations)
    n_seeds, n_gens = arr.shape
    means = arr.mean(axis=0)
    sems = arr.std(axis=0, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else np.zeros(n_gens)

    ci_low = []
    ci_high = []
    for g in range(n_gens):
        ci = bootstrap_ci(
            arr[:, g].tolist(),
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
        )
        ci_low.append(ci.lower)
        ci_high.append(ci.upper)

    return MethodCurve(
        method=method,
        generations=list(range(n_gens)),
        mean=means.tolist(),
        sem=sems.tolist(),
        ci_low=ci_low,
        ci_high=ci_high,
        n_seeds=n_seeds,
    )


def aggregate_scores_across_seeds(
    method: str,
    per_seed_scores: Sequence[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int | None = None,
) -> MethodScores:
    """Aggregate per-seed scalar scores into mean +/- bootstrap CI."""
    from breed.stats import bootstrap_ci

    arr = np.asarray(per_seed_scores)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sem = std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    ci = bootstrap_ci(
        per_seed_scores,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    return MethodScores(
        method=method,
        scores_per_seed=list(per_seed_scores),
        mean=mean,
        std=std,
        sem=sem,
        ci_low=ci.lower,
        ci_high=ci.upper,
    )
