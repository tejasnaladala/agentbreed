"""Publication-quality matplotlib charts for breeding results.

All charts use a dark theme inspired by GitHub's dark mode. Matplotlib
is lazily imported so the rest of the library stays lightweight --
users who do not need charts are never forced to install it.

Install the optional dependency with::

    pip install agentbreed[charts]
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from breed.lineage import LineageTracker

# ---------------------------------------------------------------------------
# Lazy matplotlib import
# ---------------------------------------------------------------------------

_MPL_MISSING_MSG = (
    "matplotlib is required for chart generation. "
    "Install it with:  pip install agentbreed[charts]"
)


def _import_matplotlib():
    """Import matplotlib and return (matplotlib, pyplot) or raise ImportError.

    Forces the non-interactive ``Agg`` backend so chart generation works
    in headless environments and CI pipelines.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(_MPL_MISSING_MSG) from exc
    return matplotlib, plt


# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------

BG_COLOR = "#0d1117"
TEXT_COLOR = "#c9d1d9"
GRID_COLOR = "#21262d"

BLUE = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
PURPLE = "#d2a8ff"
CYAN = "#79c0ff"

LABEL_SIZE = 12
TITLE_SIZE = 14
CURVE_FIGSIZE = (12, 6)
HEATMAP_FIGSIZE = (10, 8)
TREE_FIGSIZE = (10, 8)
DPI = 150


def _apply_dark_theme(ax, fig):
    """Apply the dark theme to a figure and axes."""
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=LABEL_SIZE - 2)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)


# ---------------------------------------------------------------------------
# 1. Fitness curve
# ---------------------------------------------------------------------------


def fitness_curve(tracker: "LineageTracker", output_path: str) -> str:
    """Plot best / average / worst fitness per generation.

    Parameters
    ----------
    tracker:
        A :class:`LineageTracker` with logged records.
    output_path:
        Filesystem path for the output PNG file.

    Returns
    -------
    str
        The *output_path* that was written.
    """
    _matplotlib, plt = _import_matplotlib()

    records = tracker.load()
    if not records:
        raise ValueError("No records in tracker -- nothing to plot.")

    # Group fitness scores by generation
    gen_scores: dict[int, list[float]] = defaultdict(list)
    for rec in records:
        gen_scores[rec.generation].append(rec.fitness_score)

    generations = sorted(gen_scores.keys())
    best = [max(gen_scores[g]) for g in generations]
    avg = [sum(gen_scores[g]) / len(gen_scores[g]) for g in generations]
    worst = [min(gen_scores[g]) for g in generations]

    fig, ax = plt.subplots(figsize=CURVE_FIGSIZE)
    _apply_dark_theme(ax, fig)

    ax.plot(generations, best, color=GREEN, linewidth=2, marker="o",
            markersize=4, label="Best")
    ax.plot(generations, avg, color=CYAN, linewidth=2, marker="s",
            markersize=4, label="Average")
    ax.plot(generations, worst, color=RED, linewidth=2, marker="^",
            markersize=4, label="Worst")

    ax.fill_between(generations, worst, best, color=BLUE, alpha=0.08)

    ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
    ax.set_ylabel("Fitness Score", fontsize=LABEL_SIZE)
    ax.set_title("Fitness Over Generations", fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
              fontsize=LABEL_SIZE - 1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 2. Population distribution
# ---------------------------------------------------------------------------


def population_distribution(
    tracker: "LineageTracker",
    generation: int,
    output_path: str,
) -> str:
    """Histogram of fitness scores for a single generation.

    Parameters
    ----------
    tracker:
        A :class:`LineageTracker` with logged records.
    generation:
        The generation number to plot.
    output_path:
        Filesystem path for the output PNG file.

    Returns
    -------
    str
        The *output_path* that was written.
    """
    _matplotlib, plt = _import_matplotlib()

    records = tracker.load()
    scores = [r.fitness_score for r in records if r.generation == generation]
    if not scores:
        raise ValueError(f"No records for generation {generation}.")

    fig, ax = plt.subplots(figsize=CURVE_FIGSIZE)
    _apply_dark_theme(ax, fig)

    bin_count = max(5, min(30, len(scores) // 2))
    ax.hist(scores, bins=bin_count, color=BLUE, edgecolor=CYAN, alpha=0.85)

    ax.set_xlabel("Fitness Score", fontsize=LABEL_SIZE)
    ax.set_ylabel("Count", fontsize=LABEL_SIZE)
    ax.set_title(
        f"Population Distribution -- Generation {generation}",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 3. Gene persistence heatmap
# ---------------------------------------------------------------------------


def gene_persistence(tracker: "LineageTracker", output_path: str) -> str:
    """Heatmap showing which gene values persist across generations.

    For ENUM genes the chart tracks which option dominates each generation.
    The cell value is the fraction of the population carrying the most
    common option for that gene in that generation.

    Parameters
    ----------
    tracker:
        A :class:`LineageTracker` with logged records.
    output_path:
        Filesystem path for the output PNG file.

    Returns
    -------
    str
        The *output_path* that was written.
    """
    _matplotlib, plt = _import_matplotlib()
    import numpy as np

    records = tracker.load()
    if not records:
        raise ValueError("No records in tracker -- nothing to plot.")

    # Identify ENUM genes from the first record's snapshot
    enum_gene_names: list[str] = []
    sample_snapshot = records[0].genome_snapshot
    genes_data = sample_snapshot.get("genes", {})
    for gene_name, gene_info in genes_data.items():
        gene_type = gene_info.get("type", "")
        if gene_type in ("enum", "ENUM"):
            enum_gene_names.append(gene_name)

    if not enum_gene_names:
        # Fall back: treat all genes as present, show fitness as proxy
        raise ValueError("No ENUM genes found in genome snapshots.")

    # Group records by generation
    gen_records: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        gen_records[rec.generation].append(rec.genome_snapshot)

    generations = sorted(gen_records.keys())

    # Build the heatmap matrix: rows = genes, cols = generations
    # Value = fraction of population with the dominant option
    matrix = np.zeros((len(enum_gene_names), len(generations)))
    annotations: list[list[str]] = [
        [""] * len(generations) for _ in enum_gene_names
    ]

    for col_idx, gen in enumerate(generations):
        snapshots = gen_records[gen]
        for row_idx, gene_name in enumerate(enum_gene_names):
            values: list[str] = []
            for snap in snapshots:
                gene_info = snap.get("genes", {}).get(gene_name, {})
                val = gene_info.get("value")
                if val is not None:
                    values.append(str(val))

            if values:
                # Find the dominant value and its frequency
                counts: dict[str, int] = defaultdict(int)
                for v in values:
                    counts[v] += 1
                dominant = max(counts, key=lambda k: counts[k])
                fraction = counts[dominant] / len(values)
                matrix[row_idx, col_idx] = fraction
                annotations[row_idx][col_idx] = dominant

    fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)
    _apply_dark_theme(ax, fig)

    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(generations)))
    ax.set_xticklabels([str(g) for g in generations])
    ax.set_yticks(range(len(enum_gene_names)))
    ax.set_yticklabels(enum_gene_names)

    # Add text annotations
    for row_idx in range(len(enum_gene_names)):
        for col_idx in range(len(generations)):
            val = matrix[row_idx, col_idx]
            text_color = "#000000" if val > 0.6 else TEXT_COLOR
            ax.text(
                col_idx,
                row_idx,
                annotations[row_idx][col_idx],
                ha="center",
                va="center",
                color=text_color,
                fontsize=max(6, LABEL_SIZE - 3),
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    cbar.outline.set_edgecolor(GRID_COLOR)
    for label in cbar.ax.get_yticklabels():
        label.set_color(TEXT_COLOR)
    cbar.set_label("Dominance Fraction", color=TEXT_COLOR, fontsize=LABEL_SIZE)

    ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
    ax.set_ylabel("Gene", fontsize=LABEL_SIZE)
    ax.set_title("Gene Value Persistence", fontsize=TITLE_SIZE, fontweight="bold")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 4. Static lineage tree
# ---------------------------------------------------------------------------


def lineage_tree_static(tracker: "LineageTracker", output_path: str) -> str:
    """Simple matplotlib tree visualisation with fitness-coloured nodes.

    Nodes are positioned by generation (Y) and index within generation (X).
    The champion path (highest fitness per generation) is highlighted.

    Parameters
    ----------
    tracker:
        A :class:`LineageTracker` with logged records.
    output_path:
        Filesystem path for the output PNG file.

    Returns
    -------
    str
        The *output_path* that was written.
    """
    _matplotlib, plt = _import_matplotlib()
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    import numpy as np

    tree = tracker.get_tree()
    if not tree:
        raise ValueError("No records in tracker -- nothing to plot.")

    # Group nodes by generation
    gen_nodes: dict[int, list[str]] = defaultdict(list)
    for nid, node in tree.items():
        gen_nodes[node.generation].append(nid)

    generations = sorted(gen_nodes.keys())

    # Sort nodes within each generation by fitness (best on left)
    for gen in generations:
        gen_nodes[gen].sort(
            key=lambda nid: tree[nid].fitness_score, reverse=True
        )

    # Assign positions
    positions: dict[str, tuple[float, float]] = {}
    for gen in generations:
        nodes_in_gen = gen_nodes[gen]
        n = len(nodes_in_gen)
        for idx, nid in enumerate(nodes_in_gen):
            x = (idx - (n - 1) / 2.0) if n > 1 else 0.0
            y = -gen  # Generations go downward
            positions[nid] = (x, y)

    # Find champion path
    champion_ids: set[str] = set()
    for gen in generations:
        nodes_in_gen = gen_nodes[gen]
        if nodes_in_gen:
            best_nid = max(nodes_in_gen, key=lambda nid: tree[nid].fitness_score)
            champion_ids.add(best_nid)

    # Collect fitness values for colour mapping
    all_fitness = [node.fitness_score for node in tree.values()]
    f_min = min(all_fitness)
    f_max = max(all_fitness)
    norm = Normalize(vmin=f_min, vmax=f_max) if f_max > f_min else Normalize(0, 1)

    # Red-to-green colormap
    cmap = cm.RdYlGn

    fig, ax = plt.subplots(figsize=TREE_FIGSIZE)
    _apply_dark_theme(ax, fig)
    ax.grid(False)

    # Draw edges
    for nid, node in tree.items():
        if nid not in positions:
            continue
        x1, y1 = positions[nid]
        for parent_id in node.parents:
            if parent_id in positions:
                x0, y0 = positions[parent_id]
                is_champ_edge = nid in champion_ids and parent_id in champion_ids
                edge_color = GREEN if is_champ_edge else GRID_COLOR
                edge_width = 2.0 if is_champ_edge else 0.8
                edge_alpha = 1.0 if is_champ_edge else 0.5
                ax.plot(
                    [x0, x1], [y0, y1],
                    color=edge_color,
                    linewidth=edge_width,
                    alpha=edge_alpha,
                    zorder=1,
                )

    # Draw nodes
    for nid, (x, y) in positions.items():
        fitness = tree[nid].fitness_score
        color = cmap(norm(fitness))
        size = 120 if nid in champion_ids else 60
        edge_color = GREEN if nid in champion_ids else "none"
        ax.scatter(
            x, y,
            c=[color],
            s=size,
            edgecolors=edge_color,
            linewidths=1.5 if nid in champion_ids else 0.5,
            zorder=2,
        )

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(np.array([]))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    cbar.outline.set_edgecolor(GRID_COLOR)
    for label in cbar.ax.get_yticklabels():
        label.set_color(TEXT_COLOR)
    cbar.set_label("Fitness", color=TEXT_COLOR, fontsize=LABEL_SIZE)

    ax.set_xlabel("Population Index", fontsize=LABEL_SIZE)
    ax.set_ylabel("Generation", fontsize=LABEL_SIZE)
    ax.set_title("Lineage Tree", fontsize=TITLE_SIZE, fontweight="bold")

    # Make Y-axis labels show positive generation numbers
    if generations:
        y_ticks = [-g for g in generations]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(g) for g in generations])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 5. Generate all charts
# ---------------------------------------------------------------------------


def generate_all_charts(
    tracker: "LineageTracker",
    output_dir: str,
) -> list[str]:
    """Generate all available charts and return the list of file paths.

    Parameters
    ----------
    tracker:
        A :class:`LineageTracker` with logged records.
    output_dir:
        Directory where chart PNGs will be saved.  Created if needed.

    Returns
    -------
    list[str]
        Paths to all generated chart files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []

    # Fitness curve
    paths.append(fitness_curve(tracker, str(out / "fitness_curve.png")))

    # Population distribution for every generation
    records = tracker.load()
    generations = sorted({r.generation for r in records})
    for gen in generations:
        paths.append(
            population_distribution(
                tracker, gen, str(out / f"population_gen{gen}.png")
            )
        )

    # Gene persistence (may fail if no ENUM genes)
    try:
        paths.append(gene_persistence(tracker, str(out / "gene_persistence.png")))
    except ValueError:
        pass  # No ENUM genes -- skip gracefully

    # Lineage tree
    paths.append(lineage_tree_static(tracker, str(out / "lineage_tree.png")))

    return paths
