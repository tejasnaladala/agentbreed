"""Frozen agent genome schema for the real study.

This module IS the preregistered search space. Any change to the gene list,
legal values, or canonical sweep order after the preregistration lock
(preregistration_real_v1.md) is a protocol deviation and must be recorded
as a dated amendment to the preregistration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Sequence, Tuple, Union
import random


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

class GeneType(str, Enum):
    ENUM = "enum"
    SET = "set"
    FLOAT = "float"
    BOOL = "bool"


class GeneCategory(str, Enum):
    SEMANTIC = "semantic"
    CONTROL_FLOW = "control_flow"
    TOOL_MEMORY = "tool_memory"
    COMPUTE_BUDGET = "compute_budget"


class GeneScope(str, Enum):
    UNIVERSAL = "universal"
    BENCHMARK_SPECIFIC = "benchmark_specific"


# ---------------------------------------------------------------------------
# Benchmark keys — these must match benchmarks/base.py BENCHMARK_IDS
# ---------------------------------------------------------------------------

BENCHMARKS = ("forecastbench", "livecodebench_v6", "gpqa_diamond")


# ---------------------------------------------------------------------------
# Gene spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneSpec:
    name: str
    type: GeneType
    category: GeneCategory
    scope: GeneScope
    # For ENUM/SET: per-benchmark legal values if benchmark_specific, or a single tuple if universal.
    # For FLOAT: (low, high) tuple; discretized_values is the finite set used by discrete-space methods.
    # For BOOL: None.
    legal_values: Union[Tuple[str, ...], Dict[str, Tuple[str, ...]], Tuple[float, float], None]
    discretized_values: Tuple[float, ...] = ()
    default: Union[str, frozenset, float, bool, Dict[str, Union[str, frozenset]]] = ""


# ---------------------------------------------------------------------------
# The 9 genes (v1) — frozen at preregistration lock
# ---------------------------------------------------------------------------

GENES: Tuple[GeneSpec, ...] = (
    # --- Semantic genes ---
    GeneSpec(
        name="system_prompt",
        type=GeneType.ENUM,
        category=GeneCategory.SEMANTIC,
        scope=GeneScope.BENCHMARK_SPECIFIC,
        legal_values={
            "forecastbench": (
                "plain_forecaster",
                "calibrated_base_rates",
                "scenario_planner",
                "devils_advocate",
                "ensemble_of_views",
            ),
            "livecodebench_v6": (
                "plain_coder",
                "test_driven",
                "stepwise_algorithm",
                "edge_case_hunter",
                "explain_then_code",
            ),
            "gpqa_diamond": (
                "plain_scientist",
                "first_principles",
                "expert_in_field",
                "socratic_self_check",
                "compare_and_contrast",
            ),
        },
        default={
            "forecastbench": "plain_forecaster",
            "livecodebench_v6": "plain_coder",
            "gpqa_diamond": "plain_scientist",
        },
    ),
    GeneSpec(
        name="decomposition_style",
        type=GeneType.ENUM,
        category=GeneCategory.SEMANTIC,
        scope=GeneScope.UNIVERSAL,
        legal_values=(
            "none",
            "chain_of_thought",
            "least_to_most",
            "tree_of_thought_depth_2",
            "self_ask",
        ),
        default="none",
    ),
    GeneSpec(
        name="self_critique",
        type=GeneType.ENUM,
        category=GeneCategory.SEMANTIC,
        scope=GeneScope.UNIVERSAL,
        legal_values=(
            "off",
            "single_pass_critique",
            "reflexion_1turn",
            "rebuttal_debate",
        ),
        default="off",
    ),
    # --- Control-flow genes ---
    GeneSpec(
        name="answer_format",
        type=GeneType.ENUM,
        category=GeneCategory.CONTROL_FLOW,
        scope=GeneScope.BENCHMARK_SPECIFIC,
        legal_values={
            "forecastbench": (
                "bare_number",
                "json_probability",
                "markdown_reasoning_then_number",
                "percent_phrase",
            ),
            "livecodebench_v6": (
                "fenced_python",
                "markdown_code_then_explanation",
                "plain_function_only",
                "comment_then_code",
            ),
            "gpqa_diamond": (
                "bare_letter",
                "after_answer_marker",
                "reasoning_then_letter",
                "json_with_letter",
            ),
        },
        default={
            "forecastbench": "markdown_reasoning_then_number",
            "livecodebench_v6": "fenced_python",
            "gpqa_diamond": "after_answer_marker",
        },
    ),
    GeneSpec(
        name="stopping_policy",
        type=GeneType.ENUM,
        category=GeneCategory.CONTROL_FLOW,
        scope=GeneScope.UNIVERSAL,
        legal_values=(
            "default_eos",
            "strict_short",
            "strict_medium",
            "strict_long",
        ),
        default="default_eos",
    ),
    # --- Tool / memory genes ---
    GeneSpec(
        name="memory_structure",
        type=GeneType.ENUM,
        category=GeneCategory.TOOL_MEMORY,
        scope=GeneScope.UNIVERSAL,
        legal_values=(
            "stateless",
            "running_scratchpad_in_prompt",
            "retrieved_similar_past_items",
        ),
        default="stateless",
    ),
    GeneSpec(
        name="tool_policy",
        type=GeneType.SET,
        category=GeneCategory.TOOL_MEMORY,
        scope=GeneScope.BENCHMARK_SPECIFIC,
        legal_values={
            "forecastbench": ("base_rate_lookup", "analogy_finder", "scenario_tree"),
            "livecodebench_v6": (
                "run_example_tests_inline",
                "type_check_inline",
                "edge_case_generator",
            ),
            "gpqa_diamond": ("formula_lookup", "definition_lookup", "unit_converter"),
        },
        default={
            "forecastbench": frozenset(),
            "livecodebench_v6": frozenset(),
            "gpqa_diamond": frozenset(),
        },
    ),
    # --- Compute-budget genes ---
    GeneSpec(
        name="temperature",
        type=GeneType.FLOAT,
        category=GeneCategory.COMPUTE_BUDGET,
        scope=GeneScope.UNIVERSAL,
        legal_values=(0.0, 1.2),
        discretized_values=tuple(round(0.1 * i, 1) for i in range(13)),
        default=0.0,
    ),
    GeneSpec(
        name="prompt_token_budget",
        type=GeneType.ENUM,
        category=GeneCategory.COMPUTE_BUDGET,
        scope=GeneScope.UNIVERSAL,
        legal_values=("tight_512", "medium_1024", "large_2048", "xlarge_4096"),
        default="medium_1024",
    ),
)

GENE_NAMES: Tuple[str, ...] = tuple(g.name for g in GENES)


# ---------------------------------------------------------------------------
# Canonical sweep order for the dimensionality experiment
# ---------------------------------------------------------------------------

CANONICAL_SWEEP_ORDER: Tuple[str, ...] = (
    "system_prompt",
    "decomposition_style",
    "answer_format",
    "self_critique",
    "memory_structure",
    "tool_policy",
    "stopping_policy",
    "temperature",
    "prompt_token_budget",
)

assert set(CANONICAL_SWEEP_ORDER) == set(GENE_NAMES), (
    "CANONICAL_SWEEP_ORDER must be a permutation of the 9 genes"
)


# ---------------------------------------------------------------------------
# Genome
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Genome:
    """Immutable genome for one (benchmark, run) tuple.

    `values` is a tuple of (gene_name, value) pairs in GENE_NAMES order.
    `benchmark` is the benchmark key this genome is legal for.
    """
    benchmark: str
    values: Tuple[Tuple[str, Any], ...]

    def as_dict(self) -> Dict[str, Any]:
        return {name: value for name, value in self.values}

    def get(self, gene_name: str) -> Any:
        for name, value in self.values:
            if name == gene_name:
                return value
        raise KeyError(f"gene {gene_name!r} not in genome")

    def with_override(self, gene_name: str, new_value: Any) -> "Genome":
        new_values = tuple(
            (n, new_value if n == gene_name else v) for n, v in self.values
        )
        return Genome(benchmark=self.benchmark, values=new_values)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gene_by_name(name: str) -> GeneSpec:
    for g in GENES:
        if g.name == name:
            return g
    raise KeyError(f"no gene named {name!r}")


def legal_values_for(gene_name: str, benchmark: str) -> Any:
    """Return the legal value set for a gene on a given benchmark.

    For SET genes, returns the tuple of legal elements (multi-hot semantics).
    For FLOAT genes, returns a tuple (low, high).
    For BOOL, returns (True, False).
    """
    gene = _gene_by_name(gene_name)
    if gene.type == GeneType.BOOL:
        return (True, False)
    if gene.type == GeneType.FLOAT:
        assert isinstance(gene.legal_values, tuple) and len(gene.legal_values) == 2
        return gene.legal_values  # (low, high)

    # ENUM or SET
    if gene.scope == GeneScope.UNIVERSAL:
        assert isinstance(gene.legal_values, tuple)
        return gene.legal_values
    assert isinstance(gene.legal_values, dict)
    if benchmark not in gene.legal_values:
        raise KeyError(f"gene {gene_name!r} has no legal values for benchmark {benchmark!r}")
    return gene.legal_values[benchmark]


def default_genome_for(benchmark: str) -> Genome:
    """Construct the default genome for a benchmark (used for 'gene frozen' ablations)."""
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark!r}; legal: {BENCHMARKS}")
    values: List[Tuple[str, Any]] = []
    for gene in GENES:
        if isinstance(gene.default, dict):
            values.append((gene.name, gene.default[benchmark]))
        else:
            values.append((gene.name, gene.default))
    return Genome(benchmark=benchmark, values=tuple(values))


def random_genome_for(benchmark: str, rng: random.Random) -> Genome:
    """Sample a legal genome uniformly at random.

    FLOAT genes use discretized_values for sampling (aligning with the
    discrete-space search methods). Continuous-space methods can replace
    the temperature value via with_override if needed.
    """
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark!r}")
    values: List[Tuple[str, Any]] = []
    for gene in GENES:
        if gene.type == GeneType.ENUM:
            legal = legal_values_for(gene.name, benchmark)
            values.append((gene.name, rng.choice(legal)))
        elif gene.type == GeneType.SET:
            legal = legal_values_for(gene.name, benchmark)
            subset = frozenset(x for x in legal if rng.random() < 0.5)
            values.append((gene.name, subset))
        elif gene.type == GeneType.FLOAT:
            if gene.discretized_values:
                values.append((gene.name, rng.choice(gene.discretized_values)))
            else:
                low, high = legal_values_for(gene.name, benchmark)
                values.append((gene.name, rng.uniform(low, high)))
        elif gene.type == GeneType.BOOL:
            values.append((gene.name, rng.random() < 0.5))
        else:
            raise ValueError(f"unknown gene type {gene.type!r}")
    return Genome(benchmark=benchmark, values=tuple(values))
