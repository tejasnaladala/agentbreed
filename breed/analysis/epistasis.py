"""Epistasis / functional ANOVA decomposition of agent fitness landscapes.

This module implements the theoretical centerpiece of the agentbreed
research paper: a variance-based decomposition of an LLM agent fitness
landscape into contributions from (a) single genes (first-order Sobol
indices), (b) pairwise gene interactions (second-order Sobol indices),
and (c) higher-order interactions (captured by total-order indices).

The decomposition follows the classical functional ANOVA approach of
Sobol (2001), using the Saltelli (2010) sampling scheme with the
Jansen (1999) estimator for first-order indices and the Sobol (2007) /
Jansen (1999) estimator for total-order indices.

References
----------
Sobol, I. M. (2001). Global sensitivity indices for nonlinear mathematical
    models and their Monte Carlo estimates. Mathematics and Computers in
    Simulation, 55(1-3), 271-280.
Jansen, M. J. W. (1999). Analysis of variance designs for model output.
    Computer Physics Communications, 117(1-2), 35-43.
Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., &
    Tarantola, S. (2010). Variance based sensitivity analysis of model
    output. Design and estimator for the total sensitivity index.
    Computer Physics Communications, 181(2), 259-270.

Notes
-----
Implementation details worth knowing:

* Gene decoding is fully deterministic given a [0, 1]^d sample, so
  re-running this pipeline with the same seed produces the same
  design points and the same indices.
* Total variance of zero (constant score) is handled by returning
  all-zero indices with a note. Sum/fraction properties on
  :class:`SobolIndices` and :class:`EpistasisReport` are therefore
  well-defined in that degenerate case.
* Sobol indices from finite samples can be slightly negative due to
  noise. Raw values are preserved in the indices dataclass; the
  :meth:`EpistasisReport.to_dict` and :meth:`to_markdown` methods
  clip negative values to zero for display.
* SET genes are encoded as one Boolean dimension per available item;
  each bit is thresholded at 0.5. NUMERIC_VECTOR genes get one
  dimension per element plus a post-hoc softmax normalization.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

import numpy as np
from scipy.stats import qmc

from breed.arenas.base import Task
from breed.genome import Gene, GeneType, Genome, _resolve_gene_type

_LOG = logging.getLogger(__name__)

_H3_THRESHOLD: float = 0.10
"""H3 confirmation threshold for the epistasis hypothesis."""

_FLOAT_BINS: int = 10
"""Number of bins used when discretising FLOAT genes for pairwise stratification."""


# ---------------------------------------------------------------------------
# Gene dimension layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _GeneLayout:
    """Layout of a single gene within the [0, 1]^d Sobol design matrix.

    Attributes
    ----------
    name:
        Gene name.
    gene_type:
        Resolved :class:`GeneType`.
    start:
        Starting index (inclusive) of this gene's slice in the design matrix.
    width:
        Number of contiguous design-matrix columns consumed by the gene.
    spec:
        The original gene-template specification dict (kept verbatim so
        decoding has full access to options / range / available / size).
    """

    name: str
    gene_type: GeneType
    start: int
    width: int
    spec: dict[str, Any]

    @property
    def stop(self) -> int:
        """Exclusive stop index of this gene's slice."""
        return self.start + self.width


def _build_layouts(gene_template: dict[str, dict[str, Any]]) -> list[_GeneLayout]:
    """Compute the column layout for a Sobol design matrix given a gene template.

    Parameters
    ----------
    gene_template:
        Mapping of gene name to gene specification dict. The specification
        schema mirrors the one used by :func:`breed.genome.create_genome_from_template`.

    Returns
    -------
    list[_GeneLayout]
        Ordered list of :class:`_GeneLayout` entries. Iteration order is
        deterministic (follows ``dict.items()`` insertion order, which is
        stable in Python 3.7+).
    """
    layouts: list[_GeneLayout] = []
    cursor = 0
    for name, spec in gene_template.items():
        gene_type = _resolve_gene_type(spec["type"])
        if gene_type == GeneType.ENUM:
            width = 1
        elif gene_type == GeneType.FLOAT:
            width = 1
        elif gene_type == GeneType.SET:
            available = spec.get("available")
            if not available:
                raise ValueError(
                    f"Gene '{name}': SET gene requires a non-empty 'available' list"
                )
            width = len(available)
        elif gene_type == GeneType.NUMERIC_VECTOR:
            # Prefer explicit 'size' for determinism. Fall back to the length
            # of the first template vector if size is absent.
            size = spec.get("size")
            if size is None:
                templates = spec.get("templates")
                if not templates:
                    raise ValueError(
                        f"Gene '{name}': NUMERIC_VECTOR gene needs 'size' or 'templates'"
                    )
                size = len(templates[0])
            width = int(size)
        elif gene_type == GeneType.TEXT:
            # TEXT treated as a discrete choice over provided templates.
            templates = spec.get("templates")
            if not templates:
                raise ValueError(
                    f"Gene '{name}': TEXT gene requires a non-empty 'templates' list"
                )
            width = 1
        else:
            raise ValueError(f"Gene '{name}': unsupported gene type {gene_type}")

        layouts.append(
            _GeneLayout(
                name=name,
                gene_type=gene_type,
                start=cursor,
                width=width,
                spec=spec,
            )
        )
        cursor += width
    return layouts


def _effective_dim(layouts: Sequence[_GeneLayout]) -> int:
    """Return the total number of design-matrix columns across all genes."""
    return sum(layout.width for layout in layouts)


# ---------------------------------------------------------------------------
# Decoding: [0,1]^d -> concrete gene values
# ---------------------------------------------------------------------------

def _decode_row(row: np.ndarray, layouts: Sequence[_GeneLayout]) -> dict[str, Any]:
    """Decode one row of the design matrix into a ``{gene_name: value}`` dict.

    Parameters
    ----------
    row:
        1-D float array of shape ``(d,)`` with values in [0, 1].
    layouts:
        Output of :func:`_build_layouts`.

    Returns
    -------
    dict[str, Any]
        Concrete gene values compatible with
        :class:`breed.genome.Gene` constraints.
    """
    out: dict[str, Any] = {}
    for layout in layouts:
        slice_vals = row[layout.start : layout.stop]
        out[layout.name] = _decode_slice(slice_vals, layout)
    return out


def _decode_slice(slice_vals: np.ndarray, layout: _GeneLayout) -> Any:
    """Decode a single gene's slice of the design row into a concrete value.

    Parameters
    ----------
    slice_vals:
        1-D float array of length ``layout.width`` with values in [0, 1].
    layout:
        Layout describing the gene type, spec, and width.

    Returns
    -------
    Any
        Gene value suitable for use in a :class:`Gene` instance.
    """
    spec = layout.spec
    gtype = layout.gene_type

    if gtype == GeneType.ENUM:
        options: list[str] = spec["options"]
        idx = min(int(math.floor(float(slice_vals[0]) * len(options))), len(options) - 1)
        return options[idx]

    if gtype == GeneType.FLOAT:
        lo, hi = spec["range"]
        lo_f = float(lo)
        hi_f = float(hi)
        return lo_f + float(slice_vals[0]) * (hi_f - lo_f)

    if gtype == GeneType.SET:
        available: list[str] = spec["available"]
        chosen = [available[i] for i, v in enumerate(slice_vals) if float(v) >= 0.5]
        # Preserve the 'available' ordering; empty sets are legal so do not coerce.
        return chosen

    if gtype == GeneType.NUMERIC_VECTOR:
        arr = np.asarray(slice_vals, dtype=float)
        # Softmax normalization so the result is a probability-like vector.
        shifted = arr - float(arr.max())
        exp = np.exp(shifted)
        total = float(exp.sum())
        if total <= 0.0 or not math.isfinite(total):
            normalized = np.full_like(arr, 1.0 / max(1, arr.size))
        else:
            normalized = exp / total
        return [float(x) for x in normalized]

    if gtype == GeneType.TEXT:
        templates: list[str] = spec["templates"]
        idx = min(int(math.floor(float(slice_vals[0]) * len(templates))), len(templates) - 1)
        return templates[idx]

    raise ValueError(f"Unsupported gene type: {gtype}")


def _build_genome_from_dict(
    gene_values: dict[str, Any], gene_template: dict[str, dict[str, Any]]
) -> Genome:
    """Build a :class:`Genome` instance from a decoded ``{name: value}`` dict.

    Parameters
    ----------
    gene_values:
        Concrete gene values produced by :func:`_decode_row`.
    gene_template:
        The original gene template (used to attach constraint metadata).

    Returns
    -------
    Genome
        A newly-constructed genome populated with :class:`Gene` instances.
    """
    genes: dict[str, Gene] = {}
    for name, spec in gene_template.items():
        gene_type = _resolve_gene_type(spec["type"])
        value = gene_values[name]
        kwargs: dict[str, Any] = {
            "name": name,
            "type": gene_type,
            "value": value,
            "mutation_rate": float(spec.get("mutation_rate", 0.1)),
        }
        if gene_type == GeneType.ENUM:
            kwargs["options"] = list(spec["options"])
        elif gene_type == GeneType.FLOAT:
            lo, hi = spec["range"]
            kwargs["range"] = (float(lo), float(hi))
        elif gene_type == GeneType.SET:
            kwargs["available"] = list(spec["available"])
        genes[name] = Gene(**kwargs)
    return Genome(genes=genes)


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SobolIndices:
    """Variance-based sensitivity indices for a gene template.

    Attributes
    ----------
    gene_names:
        Tuple of gene names in design-matrix order.
    first_order:
        Mapping of gene name -> first-order Sobol index :math:`S_i`.
    total_order:
        Mapping of gene name -> total-order Sobol index :math:`S_{T_i}`.
    pairwise:
        Mapping of (gene_a, gene_b) -> pairwise interaction index
        :math:`S_{ij}` (a canonicalised sorted tuple is used).
    total_variance:
        Total output variance :math:`\\mathrm{Var}(Y)` of the design.
    n_samples:
        Number of Saltelli samples :math:`N \\cdot (d + 2)` used to
        estimate the indices.
    method:
        Estimator label: currently always ``"saltelli_jansen"``.
    """

    gene_names: tuple[str, ...]
    first_order: dict[str, float]
    total_order: dict[str, float]
    pairwise: dict[tuple[str, str], float]
    total_variance: float
    n_samples: int
    method: str

    @property
    def sum_first_order(self) -> float:
        """Sum of first-order indices, clipped to [0, 1]."""
        if not self.first_order:
            return 0.0
        total = sum(max(0.0, float(v)) for v in self.first_order.values())
        return float(min(1.0, total))

    @property
    def sum_pairwise(self) -> float:
        """Sum of pairwise indices, clipped at zero per pair."""
        if not self.pairwise:
            return 0.0
        return float(sum(max(0.0, float(v)) for v in self.pairwise.values()))

    @property
    def interaction_fraction(self) -> float:
        """Fraction of variance explained by interactions of any order.

        Computed as ``sum(total_order) - sum(first_order)`` with per-term
        non-negativity clipping. Returns a float in approximately [0, d].
        """
        if not self.total_order:
            return 0.0
        total_t = sum(max(0.0, float(v)) for v in self.total_order.values())
        total_1 = sum(max(0.0, float(v)) for v in self.first_order.values())
        return float(max(0.0, total_t - total_1))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the indices to a plain nested dict."""
        return {
            "gene_names": list(self.gene_names),
            "first_order": dict(self.first_order),
            "total_order": dict(self.total_order),
            "pairwise": {f"{a}|{b}": v for (a, b), v in self.pairwise.items()},
            "total_variance": float(self.total_variance),
            "n_samples": int(self.n_samples),
            "method": self.method,
            "sum_first_order": self.sum_first_order,
            "sum_pairwise": self.sum_pairwise,
            "interaction_fraction": self.interaction_fraction,
        }


@dataclass(frozen=True)
class EpistasisReport:
    """Human-readable epistasis decomposition report.

    Attributes
    ----------
    indices:
        The underlying :class:`SobolIndices`.
    top_interactions:
        List of ``(gene_a, gene_b, strength)`` triples sorted by
        ``strength`` descending. Up to 20 entries are retained.
    h3_confirmed:
        ``True`` when ``sum_pairwise >= 0.10`` (the preregistered
        threshold).
    notes:
        Free-form diagnostics (negative-index warnings, degenerate
        cases, etc.).
    """

    indices: SobolIndices
    top_interactions: list[tuple[str, str, float]]
    h3_confirmed: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a plain nested dict."""
        return {
            "indices": self.indices.to_dict(),
            "top_interactions": [
                {"gene_a": a, "gene_b": b, "strength": float(s)}
                for (a, b, s) in self.top_interactions
            ],
            "h3_confirmed": bool(self.h3_confirmed),
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        """Render the report as a compact markdown string.

        Returns
        -------
        str
            Markdown suitable for embedding in a paper or a log.
        """
        lines: list[str] = []
        lines.append("# Epistasis / Functional ANOVA Report")
        lines.append("")
        lines.append(f"- Total variance: {self.indices.total_variance:.6f}")
        lines.append(f"- Samples (Saltelli): {self.indices.n_samples}")
        lines.append(f"- Method: {self.indices.method}")
        lines.append(f"- Sum of first-order indices: {self.indices.sum_first_order:.4f}")
        lines.append(f"- Sum of pairwise indices: {self.indices.sum_pairwise:.4f}")
        lines.append(f"- Interaction fraction: {self.indices.interaction_fraction:.4f}")
        lines.append(f"- H3 confirmed (>= {_H3_THRESHOLD:.2f}): {self.h3_confirmed}")
        lines.append("")
        lines.append("## First-order (S_i)")
        for name in self.indices.gene_names:
            val = max(0.0, float(self.indices.first_order.get(name, 0.0)))
            lines.append(f"- {name}: {val:.4f}")
        lines.append("")
        lines.append("## Total-order (S_Ti)")
        for name in self.indices.gene_names:
            val = max(0.0, float(self.indices.total_order.get(name, 0.0)))
            lines.append(f"- {name}: {val:.4f}")
        lines.append("")
        lines.append("## Top pairwise interactions")
        if not self.top_interactions:
            lines.append("- (none)")
        else:
            for a, b, s in self.top_interactions[:20]:
                lines.append(f"- {a} x {b}: {max(0.0, float(s)):.4f}")
        if self.notes:
            lines.append("")
            lines.append("## Notes")
            lines.append(self.notes)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sobol_sample(
    gene_template: dict[str, dict[str, Any]],
    n_base: int = 1024,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate a Saltelli sequence for Sobol-index estimation.

    Produces the canonical Saltelli design ``[A; B; A^B_1; ...; A^B_d]``
    of shape ``(N * (d + 2), d)`` where ``d`` is the effective dimensionality
    of the gene template. Each row is decoded into a concrete
    ``{gene_name: value}`` dict using type-specific mappings:

    * **ENUM / TEXT** -- ``floor(u * len(options))`` clipped to
      ``[0, len - 1]``.
    * **FLOAT** -- linear map ``u -> [lo, hi]``.
    * **SET** -- one Boolean dimension per available item thresholded
      at 0.5; empty sets are allowed.
    * **NUMERIC_VECTOR** -- one dimension per vector element plus a
      softmax normalization so the decoded vector sums to 1.

    Parameters
    ----------
    gene_template:
        Mapping of gene name to gene spec dict.
    n_base:
        The base sample count ``N``. Must be positive. Powers of two are
        preferred for Sobol balance but not required.
    seed:
        Optional RNG seed. Same seed yields identical design points.

    Returns
    -------
    list[dict[str, Any]]
        ``N * (d + 2)`` decoded design points.

    Raises
    ------
    ValueError
        If ``n_base`` is non-positive or the gene template decodes
        to zero effective dimensions.
    """
    if n_base <= 0:
        raise ValueError("n_base must be a positive integer")

    layouts = _build_layouts(gene_template)
    d_eff = _effective_dim(layouts)
    if d_eff == 0:
        raise ValueError("Gene template has zero effective dimensionality")

    sobol_engine = qmc.Sobol(d=2 * d_eff, seed=seed, scramble=True)
    base = sobol_engine.random(n_base)
    base = np.asarray(base, dtype=float)

    matrix_a = base[:, :d_eff]
    matrix_b = base[:, d_eff : 2 * d_eff]

    # Build the (N*(d+2)) x d Saltelli design.
    design = np.empty((n_base * (d_eff + 2), d_eff), dtype=float)
    design[0:n_base] = matrix_a
    design[n_base : 2 * n_base] = matrix_b
    for i in range(d_eff):
        ab_i = matrix_a.copy()
        ab_i[:, i] = matrix_b[:, i]
        start = (2 + i) * n_base
        stop = start + n_base
        design[start:stop] = ab_i

    return [_decode_row(design[row_idx], layouts) for row_idx in range(design.shape[0])]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

async def evaluate_sobol_sample(
    samples: list[dict[str, Any]],
    agent_fn: Callable[[dict[str, Any], str], Awaitable[str]],
    tasks: list[Task],
    scorer: Callable[[str, Any], float],
    gene_template: dict[str, dict[str, Any]],
) -> list[float]:
    """Evaluate every Saltelli design point and return mean score per point.

    For each ``sample`` dict the function constructs a valid
    :class:`Genome`, invokes ``agent_fn(genome.to_dict(), task.prompt)``
    for every task, scores each output with ``scorer(output, task.expected)``,
    and returns the per-sample mean.

    Parameters
    ----------
    samples:
        Output of :func:`sobol_sample`.
    agent_fn:
        Async callable ``(genome_dict, task_str) -> str``.
    tasks:
        Tasks to evaluate against. At least one is required.
    scorer:
        Function mapping ``(output, expected) -> float``.
    gene_template:
        The gene template used to build samples (needed to attach
        constraint metadata when constructing :class:`Gene` objects).

    Returns
    -------
    list[float]
        One mean score per sample, in input order.

    Raises
    ------
    ValueError
        If ``tasks`` is empty or ``samples`` is empty.
    """
    if not samples:
        raise ValueError("samples must not be empty")
    if not tasks:
        raise ValueError("tasks must not be empty")

    scores: list[float] = []
    for sample in samples:
        genome = _build_genome_from_dict(sample, gene_template)
        genome_dict = genome.to_dict()
        task_scores: list[float] = []
        for task in tasks:
            try:
                output = await agent_fn(genome_dict, task.prompt)
            except Exception as exc:  # pragma: no cover - defensive
                _LOG.warning("agent_fn raised %s; scoring as 0.0", exc)
                output = ""
            try:
                value = float(scorer(output, task.expected))
            except Exception as exc:  # pragma: no cover - defensive
                _LOG.warning("scorer raised %s; scoring as 0.0", exc)
                value = 0.0
            task_scores.append(value)
        mean = float(np.mean(task_scores)) if task_scores else 0.0
        scores.append(mean)
    return scores


# ---------------------------------------------------------------------------
# Index computation
# ---------------------------------------------------------------------------

def _split_saltelli_scores(
    scores: np.ndarray, n_base: int, d_eff: int
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Partition a flat Saltelli score array into ``Y_A``, ``Y_B`` and ``Y_AB_i``.

    Parameters
    ----------
    scores:
        Flat array of length ``n_base * (d_eff + 2)``.
    n_base:
        Base sample count ``N``.
    d_eff:
        Effective dimensionality ``d``.

    Returns
    -------
    tuple
        ``(y_a, y_b, y_ab_list)`` where ``y_a`` and ``y_b`` are ``(N,)``
        arrays and ``y_ab_list`` is a length-``d`` list of ``(N,)`` arrays.
    """
    expected = n_base * (d_eff + 2)
    if scores.shape[0] != expected:
        raise ValueError(
            f"score array length {scores.shape[0]} != n_base*(d+2) = {expected}"
        )
    y_a = scores[0:n_base]
    y_b = scores[n_base : 2 * n_base]
    y_ab_list: list[np.ndarray] = []
    for i in range(d_eff):
        start = (2 + i) * n_base
        stop = start + n_base
        y_ab_list.append(scores[start:stop])
    return y_a, y_b, y_ab_list


def compute_sobol_indices(
    samples: list[dict[str, Any]],
    scores: list[float],
    gene_template: dict[str, dict[str, Any]],
    n_base: int,
) -> SobolIndices:
    """Compute first-order and total-order Sobol indices from a Saltelli sample.

    Uses the Jansen (1999) / Saltelli (2010) estimators::

        S_i   = (1/N) sum_j  f(B)_j * (f(A^B_i)_j - f(A)_j) / Var(Y)
        S_Ti  = (1/(2N)) sum_j (f(A)_j - f(A^B_i)_j)^2 / Var(Y)

    For multi-dimensional genes (SET with ``k`` items or NUMERIC_VECTOR
    with ``k`` elements) we aggregate the per-column indices by summing
    across the gene's slice, which is the standard closed-index additive
    decomposition when treating the sub-dimensions as jointly
    describing the same gene.

    Parameters
    ----------
    samples:
        Output of :func:`sobol_sample` (not strictly needed here but
        accepted so callers can pass the full sampling context).
    scores:
        Per-sample mean scores, same order as ``samples``.
    gene_template:
        The gene template used for sampling.
    n_base:
        Base sample count ``N`` used when building the design.

    Returns
    -------
    SobolIndices
        First-order and total-order indices plus an empty ``pairwise``
        mapping (filled later by :func:`compute_pairwise_interactions`).
    """
    layouts = _build_layouts(gene_template)
    d_eff = _effective_dim(layouts)
    gene_names = tuple(l.name for l in layouts)

    first_order: dict[str, float] = {name: 0.0 for name in gene_names}
    total_order: dict[str, float] = {name: 0.0 for name in gene_names}

    if d_eff == 0 or not scores:
        return SobolIndices(
            gene_names=gene_names,
            first_order=first_order,
            total_order=total_order,
            pairwise={},
            total_variance=0.0,
            n_samples=len(scores),
            method="saltelli_jansen",
        )

    scores_arr = np.asarray(scores, dtype=float)
    total_var = float(np.var(scores_arr, ddof=1)) if scores_arr.size > 1 else 0.0

    if total_var <= 0.0 or not math.isfinite(total_var):
        return SobolIndices(
            gene_names=gene_names,
            first_order=first_order,
            total_order=total_order,
            pairwise={},
            total_variance=total_var if math.isfinite(total_var) else 0.0,
            n_samples=len(scores),
            method="saltelli_jansen",
        )

    y_a, y_b, y_ab_list = _split_saltelli_scores(scores_arr, n_base, d_eff)

    # Per-column estimators.
    col_first = np.zeros(d_eff, dtype=float)
    col_total = np.zeros(d_eff, dtype=float)
    for i in range(d_eff):
        y_ab_i = y_ab_list[i]
        col_first[i] = float(np.mean(y_b * (y_ab_i - y_a))) / total_var
        col_total[i] = float(0.5 * np.mean((y_a - y_ab_i) ** 2)) / total_var

    # Aggregate per gene (multi-width genes sum their columns).
    for layout in layouts:
        s_i = float(col_first[layout.start : layout.stop].sum())
        s_t = float(col_total[layout.start : layout.stop].sum())
        first_order[layout.name] = s_i
        total_order[layout.name] = s_t

    return SobolIndices(
        gene_names=gene_names,
        first_order=first_order,
        total_order=total_order,
        pairwise={},
        total_variance=total_var,
        n_samples=int(scores_arr.size),
        method="saltelli_jansen",
    )


# ---------------------------------------------------------------------------
# Pairwise interactions via stratified group means
# ---------------------------------------------------------------------------

def _canonicalise_label(value: Any) -> str:
    """Canonicalise a gene value into a string usable as a stratification key."""
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    return str(value)


def _stratum_key(value: Any, layout: _GeneLayout) -> str:
    """Compute a stratification key for a single gene value.

    * ENUM / TEXT -- raw string value.
    * SET -- sorted tuple of items joined by "|".
    * FLOAT -- ``_FLOAT_BINS``-bin discretisation over the gene range.
    * NUMERIC_VECTOR -- argmax index of the vector.
    """
    gtype = layout.gene_type
    if gtype == GeneType.ENUM or gtype == GeneType.TEXT:
        return str(value)
    if gtype == GeneType.SET:
        items = sorted(value) if isinstance(value, (list, tuple)) else [str(value)]
        return "{" + ",".join(str(v) for v in items) + "}"
    if gtype == GeneType.FLOAT:
        lo, hi = layout.spec["range"]
        lo_f = float(lo)
        hi_f = float(hi)
        span = hi_f - lo_f
        if span <= 0.0:
            return "bin0"
        pos = (float(value) - lo_f) / span
        bin_idx = min(int(math.floor(pos * _FLOAT_BINS)), _FLOAT_BINS - 1)
        bin_idx = max(0, bin_idx)
        return f"bin{bin_idx}"
    if gtype == GeneType.NUMERIC_VECTOR:
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            return "empty"
        return f"argmax{int(np.argmax(arr))}"
    return _canonicalise_label(value)


def _single_gene_closed_index(
    keys: list[str], scores: np.ndarray, mean: float, total_var: float
) -> float:
    """Variance of group means for a single categorical stratification.

    Computes ``Var_stratum(E[Y | stratum])`` via the group-mean estimator
    ``sum_g (n_g / N) * (mean_g - overall_mean) ** 2`` and normalises by
    the total variance.
    """
    if total_var <= 0.0:
        return 0.0
    groups: dict[str, list[int]] = {}
    for idx, key in enumerate(keys):
        groups.setdefault(key, []).append(idx)
    n_total = len(keys)
    if n_total == 0:
        return 0.0
    accum = 0.0
    for idxs in groups.values():
        if not idxs:
            continue
        group_scores = scores[idxs]
        group_mean = float(group_scores.mean())
        accum += (len(idxs) / n_total) * (group_mean - mean) ** 2
    return float(accum / total_var)


def _pair_closed_index(
    keys_i: list[str],
    keys_j: list[str],
    scores: np.ndarray,
    mean: float,
    total_var: float,
) -> float:
    """Closed-index variance of the 2-way conditional expectation.

    Computes ``Var[E[Y | X_i, X_j]] / Var(Y)`` via joint group means.
    """
    if total_var <= 0.0:
        return 0.0
    n_total = len(keys_i)
    if n_total == 0 or n_total != len(keys_j):
        return 0.0
    groups: dict[tuple[str, str], list[int]] = {}
    for idx, (ki, kj) in enumerate(zip(keys_i, keys_j)):
        groups.setdefault((ki, kj), []).append(idx)
    accum = 0.0
    for idxs in groups.values():
        if not idxs:
            continue
        group_scores = scores[idxs]
        group_mean = float(group_scores.mean())
        accum += (len(idxs) / n_total) * (group_mean - mean) ** 2
    return float(accum / total_var)


def compute_pairwise_interactions(
    samples: list[dict[str, Any]],
    scores: list[float],
    gene_template: dict[str, dict[str, Any]],
    indices: SobolIndices,
) -> dict[tuple[str, str], float]:
    """Estimate pairwise Sobol interaction indices :math:`S_{ij}`.

    The pairwise interaction is approximated by subtracting the
    (group-mean-based) first-order contributions of genes *i* and *j*
    from the closed joint index :math:`S_{ij}^{\\text{closed}}`::

        S_ij ~= Var[E[Y | X_i, X_j]] / Var(Y) - Var[E[Y | X_i]] / Var(Y)
                                              - Var[E[Y | X_j]] / Var(Y)

    Note that this differs numerically from the Saltelli-estimator
    first-order indices in :class:`SobolIndices`. Both estimators
    agree in expectation under infinite samples but the closed-form
    group-mean version is what the subtraction requires for the
    identity to hold.

    Parameters
    ----------
    samples:
        Output of :func:`sobol_sample`.
    scores:
        Per-sample mean scores.
    gene_template:
        Original gene template.
    indices:
        Output of :func:`compute_sobol_indices` (used only for the
        list of gene names and the total variance).

    Returns
    -------
    dict[tuple[str, str], float]
        Sorted gene-name tuples mapped to pairwise interaction estimates.
    """
    layouts = _build_layouts(gene_template)
    gene_names = tuple(l.name for l in layouts)
    layout_by_name = {l.name: l for l in layouts}

    pairwise: dict[tuple[str, str], float] = {}

    if not scores or len(scores) != len(samples) or len(gene_names) < 2:
        return pairwise

    scores_arr = np.asarray(scores, dtype=float)
    total_var = float(indices.total_variance)
    if total_var <= 0.0 or not math.isfinite(total_var):
        return pairwise
    mean = float(scores_arr.mean())

    # Pre-compute stratification keys for every gene across every sample.
    keys_by_gene: dict[str, list[str]] = {}
    first_order_closed: dict[str, float] = {}
    for name in gene_names:
        layout = layout_by_name[name]
        keys = [_stratum_key(sample[name], layout) for sample in samples]
        keys_by_gene[name] = keys
        first_order_closed[name] = _single_gene_closed_index(
            keys, scores_arr, mean, total_var
        )

    for i, name_i in enumerate(gene_names):
        for name_j in gene_names[i + 1 :]:
            closed = _pair_closed_index(
                keys_by_gene[name_i],
                keys_by_gene[name_j],
                scores_arr,
                mean,
                total_var,
            )
            interaction = closed - first_order_closed[name_i] - first_order_closed[name_j]
            key = tuple(sorted((name_i, name_j)))
            pairwise[key] = float(interaction)

    return pairwise


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

async def functional_anova_decomposition(
    gene_template: dict[str, dict[str, Any]],
    agent_fn: Callable[[dict[str, Any], str], Awaitable[str]],
    tasks: list[Task],
    scorer: Callable[[str, Any], float],
    n_base: int = 1024,
    seed: int | None = None,
) -> EpistasisReport:
    """Run the full Saltelli/Jansen functional ANOVA pipeline.

    Steps
    -----
    1. Build the Saltelli design from the gene template.
    2. Evaluate every design point on the task set.
    3. Compute first-order and total-order Sobol indices.
    4. Compute pairwise interaction indices.
    5. Rank the top pairwise interactions.
    6. Check the H3 confirmation threshold
       (``sum_pairwise >= 0.10``).
    7. Wrap everything in an :class:`EpistasisReport`.

    Parameters
    ----------
    gene_template:
        Gene template in the same format used by the experiment runner.
    agent_fn:
        Async callable ``(genome_dict, task_str) -> str``.
    tasks:
        Evaluation tasks.
    scorer:
        Per-task scoring function ``(output, expected) -> float``.
    n_base:
        Base sample count ``N`` for the Saltelli design. The total
        number of agent invocations is ``N * (d + 2) * len(tasks)``.
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    EpistasisReport
        Full decomposition report.

    Raises
    ------
    ValueError
        If ``gene_template`` has zero effective dimensions, ``n_base``
        is non-positive, or ``tasks`` is empty.
    """
    layouts = _build_layouts(gene_template)
    d_eff = _effective_dim(layouts)
    if d_eff == 0:
        raise ValueError("gene_template has no effective dimensions")
    if not tasks:
        raise ValueError("tasks must not be empty")

    samples = sobol_sample(gene_template, n_base=n_base, seed=seed)
    scores = await evaluate_sobol_sample(
        samples=samples,
        agent_fn=agent_fn,
        tasks=tasks,
        scorer=scorer,
        gene_template=gene_template,
    )

    indices = compute_sobol_indices(
        samples=samples,
        scores=scores,
        gene_template=gene_template,
        n_base=n_base,
    )

    pairwise = compute_pairwise_interactions(
        samples=samples,
        scores=scores,
        gene_template=gene_template,
        indices=indices,
    )

    enriched = SobolIndices(
        gene_names=indices.gene_names,
        first_order=indices.first_order,
        total_order=indices.total_order,
        pairwise=pairwise,
        total_variance=indices.total_variance,
        n_samples=indices.n_samples,
        method=indices.method,
    )

    top = sorted(
        (
            (a, b, float(v))
            for (a, b), v in pairwise.items()
        ),
        key=lambda triple: triple[2],
        reverse=True,
    )

    note_lines: list[str] = []
    negatives_first = [
        (n, v) for n, v in enriched.first_order.items() if v < 0.0
    ]
    if negatives_first:
        note_lines.append(
            "First-order Sobol indices with negative raw values (clipped to 0 on display): "
            + ", ".join(f"{n}={v:.4f}" for n, v in negatives_first)
        )
    negatives_total = [
        (n, v) for n, v in enriched.total_order.items() if v < 0.0
    ]
    if negatives_total:
        note_lines.append(
            "Total-order Sobol indices with negative raw values (clipped to 0 on display): "
            + ", ".join(f"{n}={v:.4f}" for n, v in negatives_total)
        )
    negatives_pair = [
        (a, b, v) for (a, b), v in pairwise.items() if v < 0.0
    ]
    if negatives_pair:
        note_lines.append(
            f"{len(negatives_pair)} pairwise interaction estimates were negative"
            " (finite-sample noise); clipped to 0 on display."
        )
    if enriched.total_variance <= 0.0:
        note_lines.append(
            "Total output variance is zero -- fitness landscape is constant "
            "and all Sobol indices are reported as zero."
        )
    if not note_lines:
        note_lines.append("No anomalies detected.")

    h3_confirmed = enriched.sum_pairwise >= _H3_THRESHOLD

    return EpistasisReport(
        indices=enriched,
        top_interactions=top,
        h3_confirmed=h3_confirmed,
        notes=" ".join(note_lines),
    )
