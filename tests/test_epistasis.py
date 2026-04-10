"""Tests for the epistasis / functional ANOVA decomposition module.

Covers
------
* Saltelli design shape (``N * (d + 2)`` rows).
* Decoded samples lie in their gene domains.
* Seed -> deterministic samples.
* Purely additive fitness: :math:`\\sum S_i \\approx 1`, pairwise :math:`\\approx 0`.
* Epistatic fitness (XOR of two ENUMs): the epistatic pair dominates
  the pairwise indices.
* End-to-end pipeline returns a valid :class:`EpistasisReport` with
  non-NaN indices on a small gene template.
* H3 threshold flag is ``True`` iff ``sum_pairwise >= 0.10``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from breed.analysis.epistasis import (
    EpistasisReport,
    SobolIndices,
    _build_layouts,
    _effective_dim,
    compute_pairwise_interactions,
    compute_sobol_indices,
    functional_anova_decomposition,
    sobol_sample,
)
from breed.arenas.base import Task


# ---------------------------------------------------------------------------
# Shared gene templates
# ---------------------------------------------------------------------------

def _additive_template() -> dict[str, dict[str, Any]]:
    """Small mixed template: 2 ENUMs + 1 FLOAT + 1 SET (2 items)."""
    return {
        "prompt_style": {
            "type": "enum",
            "options": ["a", "b", "c", "d"],
            "mutation_rate": 0.1,
        },
        "decision_rule": {
            "type": "enum",
            "options": ["x", "y", "z"],
            "mutation_rate": 0.1,
        },
        "temperature": {
            "type": "float",
            "range": (0.0, 1.0),
            "mutation_rate": 0.1,
        },
        "tool_policy": {
            "type": "set",
            "available": ["search", "calc"],
            "mutation_rate": 0.1,
        },
    }


def _pure_enum_template() -> dict[str, dict[str, Any]]:
    """Template with two ENUM genes used for the XOR epistasis test."""
    return {
        "gene_i": {
            "type": "enum",
            "options": ["A", "B"],
            "mutation_rate": 0.1,
        },
        "gene_j": {
            "type": "enum",
            "options": ["A", "B"],
            "mutation_rate": 0.1,
        },
        "gene_k": {
            "type": "enum",
            "options": ["A", "B", "C"],
            "mutation_rate": 0.1,
        },
    }


# ---------------------------------------------------------------------------
# Shape / domain / determinism
# ---------------------------------------------------------------------------

def test_sobol_sample_shape() -> None:
    template = _additive_template()
    layouts = _build_layouts(template)
    d_eff = _effective_dim(layouts)  # 1 + 1 + 1 + 2 == 5
    assert d_eff == 5
    n_base = 32
    samples = sobol_sample(template, n_base=n_base, seed=7)
    assert len(samples) == n_base * (d_eff + 2)


def test_sobol_sample_in_domain() -> None:
    template = _additive_template()
    samples = sobol_sample(template, n_base=16, seed=3)
    for sample in samples:
        assert sample["prompt_style"] in template["prompt_style"]["options"]
        assert sample["decision_rule"] in template["decision_rule"]["options"]
        t = sample["temperature"]
        lo, hi = template["temperature"]["range"]
        assert lo <= t <= hi
        tools = sample["tool_policy"]
        available = set(template["tool_policy"]["available"])
        assert set(tools).issubset(available)


def test_sobol_sample_deterministic() -> None:
    template = _additive_template()
    samples_a = sobol_sample(template, n_base=16, seed=42)
    samples_b = sobol_sample(template, n_base=16, seed=42)
    assert len(samples_a) == len(samples_b)
    for a, b in zip(samples_a, samples_b):
        assert a["prompt_style"] == b["prompt_style"]
        assert a["decision_rule"] == b["decision_rule"]
        assert math.isclose(a["temperature"], b["temperature"])
        assert a["tool_policy"] == b["tool_policy"]


# ---------------------------------------------------------------------------
# Additive landscape
# ---------------------------------------------------------------------------

def _additive_score(sample: dict[str, Any]) -> float:
    """Purely additive quality function over the shared template."""
    prompt_w = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}[sample["prompt_style"]]
    decision_w = {"x": 0.05, "y": 0.15, "z": 0.25}[sample["decision_rule"]]
    temp_w = 0.3 * float(sample["temperature"])
    tools = sample["tool_policy"]
    tool_w = 0.1 * len(tools)
    return prompt_w + decision_w + temp_w + tool_w


def test_compute_sobol_indices_additive() -> None:
    template = _additive_template()
    n_base = 512
    samples = sobol_sample(template, n_base=n_base, seed=11)
    scores = [_additive_score(s) for s in samples]
    indices = compute_sobol_indices(samples, scores, template, n_base)

    assert indices.total_variance > 0.0
    # Additive function: sum of first-order indices should be close to 1.
    assert abs(indices.sum_first_order - 1.0) < 0.1

    pairwise = compute_pairwise_interactions(samples, scores, template, indices)
    # Pairwise interactions should be negligible on a purely additive surface.
    max_abs_pair = max(abs(v) for v in pairwise.values()) if pairwise else 0.0
    assert max_abs_pair < 0.05


# ---------------------------------------------------------------------------
# Epistatic landscape (XOR of two ENUMs)
# ---------------------------------------------------------------------------

def _xor_score(sample: dict[str, Any]) -> float:
    """Score driven by an XOR between ``gene_i`` and ``gene_j``.

    The third gene ``gene_k`` has a small independent linear effect so
    the design retains some additive structure and the test is not
    pathological.
    """
    xor = 1.0 if sample["gene_i"] != sample["gene_j"] else 0.0
    k_bonus = {"A": 0.0, "B": 0.05, "C": 0.1}[sample["gene_k"]]
    return xor + k_bonus


def test_compute_sobol_indices_epistatic() -> None:
    template = _pure_enum_template()
    n_base = 1024
    samples = sobol_sample(template, n_base=n_base, seed=17)
    scores = [_xor_score(s) for s in samples]
    indices = compute_sobol_indices(samples, scores, template, n_base)
    assert indices.total_variance > 0.0

    pairwise = compute_pairwise_interactions(samples, scores, template, indices)
    assert pairwise, "Expected non-empty pairwise interactions"

    # The XOR pair (gene_i, gene_j) should be the dominant interaction.
    key_xor = tuple(sorted(("gene_i", "gene_j")))
    max_pair = max(pairwise, key=lambda k: pairwise[k])
    assert max_pair == key_xor, f"Dominant pair was {max_pair}, expected {key_xor}"
    assert pairwise[key_xor] > 0.3

    # The first-order indices of gene_i and gene_j should be near zero
    # because changing only one of them while averaging over the other
    # flips XOR uniformly.
    s_i = indices.first_order["gene_i"]
    s_j = indices.first_order["gene_j"]
    assert abs(s_i) < 0.15
    assert abs(s_j) < 0.15
    # Total-order indices for the pair should be substantially larger.
    assert indices.total_order["gene_i"] > 0.2
    assert indices.total_order["gene_j"] > 0.2


# ---------------------------------------------------------------------------
# End-to-end pipeline smoke test
# ---------------------------------------------------------------------------

def _sync_noise(seed_str: str) -> float:
    """Deterministic [0, 1] noise derived from a string seed."""
    return (hash(seed_str) & 0xFFFF) / float(0xFFFF)


async def _toy_agent(genome_dict: dict[str, Any], task: str) -> str:
    """Toy agent that returns a probability based on a few genes."""
    genes = genome_dict.get("genes", {})
    prompt = genes.get("prompt_style", {}).get("value", "a")
    decision = genes.get("decision_rule", {}).get("value", "x")
    temp = float(genes.get("temperature", {}).get("value", 0.5))
    interaction = 0.25 if (prompt == "a" and decision == "z") else 0.0
    base = 0.2 + 0.1 * ord(prompt[0]) / 128.0 + 0.3 * temp + interaction
    nudge = 0.05 * _sync_noise(prompt + decision + task)
    return f"{max(0.0, min(1.0, base + nudge)):.4f}"


def _brier_scorer(output: str, expected: Any) -> float:
    try:
        p = float(output)
    except (TypeError, ValueError):
        p = 0.5
    p = max(0.0, min(1.0, p))
    y = float(expected)
    return 1.0 - (p - y) ** 2


async def test_functional_anova_smoke() -> None:
    template = _additive_template()
    tasks = [
        Task(task_id="t0", prompt="q0", expected=1.0),
        Task(task_id="t1", prompt="q1", expected=0.0),
        Task(task_id="t2", prompt="q2", expected=1.0),
    ]
    report = await functional_anova_decomposition(
        gene_template=template,
        agent_fn=_toy_agent,
        tasks=tasks,
        scorer=_brier_scorer,
        n_base=32,
        seed=5,
    )
    assert isinstance(report, EpistasisReport)
    assert isinstance(report.indices, SobolIndices)

    for v in report.indices.first_order.values():
        assert math.isfinite(v)
    for v in report.indices.total_order.values():
        assert math.isfinite(v)
    for v in report.indices.pairwise.values():
        assert math.isfinite(v)

    # Round-trip serialization should not crash.
    assert isinstance(report.to_dict(), dict)
    assert isinstance(report.to_markdown(), str)


# ---------------------------------------------------------------------------
# H3 threshold
# ---------------------------------------------------------------------------

def _make_report(pairwise_values: dict[tuple[str, str], float]) -> EpistasisReport:
    """Build a minimal :class:`EpistasisReport` from synthetic pairwise values."""
    gene_names = tuple(sorted({g for pair in pairwise_values for g in pair}))
    indices = SobolIndices(
        gene_names=gene_names,
        first_order={g: 0.0 for g in gene_names},
        total_order={g: 0.0 for g in gene_names},
        pairwise={tuple(sorted(k)): v for k, v in pairwise_values.items()},
        total_variance=1.0,
        n_samples=128,
        method="synthetic",
    )
    sum_pair = indices.sum_pairwise
    return EpistasisReport(
        indices=indices,
        top_interactions=sorted(
            ((a, b, v) for (a, b), v in indices.pairwise.items()),
            key=lambda triple: triple[2],
            reverse=True,
        ),
        h3_confirmed=sum_pair >= 0.10,
        notes="synthetic",
    )


def test_h3_threshold_confirmed() -> None:
    report = _make_report({("g1", "g2"): 0.07, ("g1", "g3"): 0.05})
    assert report.indices.sum_pairwise >= 0.10
    assert report.h3_confirmed is True


def test_h3_threshold_not_confirmed() -> None:
    report = _make_report({("g1", "g2"): 0.04, ("g1", "g3"): 0.02})
    assert report.indices.sum_pairwise < 0.10
    assert report.h3_confirmed is False


def test_h3_threshold_edge_case_exactly_ten_percent() -> None:
    report = _make_report({("g1", "g2"): 0.10})
    assert report.indices.sum_pairwise == pytest.approx(0.10)
    assert report.h3_confirmed is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_constant_score_produces_zero_indices() -> None:
    template = _additive_template()
    samples = sobol_sample(template, n_base=32, seed=9)
    scores = [0.5] * len(samples)
    indices = compute_sobol_indices(samples, scores, template, n_base=32)
    assert indices.total_variance == 0.0
    assert all(v == 0.0 for v in indices.first_order.values())
    assert all(v == 0.0 for v in indices.total_order.values())

    pairwise = compute_pairwise_interactions(samples, scores, template, indices)
    # Pairwise mapping may be empty or all zero -- either is acceptable
    # for a constant landscape.
    assert all(v == 0.0 for v in pairwise.values())


def test_empty_gene_template_raises() -> None:
    with pytest.raises(ValueError):
        sobol_sample({}, n_base=8, seed=1)


def test_sobol_sample_rejects_non_positive_n() -> None:
    template = _additive_template()
    with pytest.raises(ValueError):
        sobol_sample(template, n_base=0, seed=1)
