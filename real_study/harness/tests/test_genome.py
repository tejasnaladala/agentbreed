"""Unit tests for the frozen 9-gene schema.

These tests protect against accidental drift in the schema. If any test here
fails, it means either (a) the schema has been edited post-lock, in which case
the preregistration has been violated, or (b) a benchmark id or gene name has
changed, in which case the change must be recorded as a dated amendment to the
preregistration and the tests updated.
"""

from __future__ import annotations

import random

import pytest

from real_study.harness.genome import (
    BENCHMARKS,
    CANONICAL_SWEEP_ORDER,
    GENE_NAMES,
    GENES,
    GeneCategory,
    GeneType,
    Genome,
    default_genome_for,
    genome_content_hash,
    legal_values_for,
    random_genome_for,
    validate_genome,
)
from real_study.harness.genome.schema import BENCHMARKS as BENCHMARK_IDS


# ---------------------------------------------------------------------------
# Schema locked-down tests
# ---------------------------------------------------------------------------

def test_exactly_nine_genes():
    assert len(GENES) == 9


def test_gene_names_are_unique():
    assert len(set(GENE_NAMES)) == 9


def test_canonical_sweep_order_is_a_permutation():
    assert set(CANONICAL_SWEEP_ORDER) == set(GENE_NAMES)
    assert len(CANONICAL_SWEEP_ORDER) == 9


def test_three_benchmarks_exactly():
    assert BENCHMARK_IDS == ("forecastbench", "livecodebench_v6", "gpqa_diamond")


def test_gene_categories_are_balanced():
    # 3 semantic / 2 control_flow / 2 tool_memory / 2 compute_budget
    counts = {cat: 0 for cat in GeneCategory}
    for g in GENES:
        counts[g.category] += 1
    assert counts[GeneCategory.SEMANTIC] == 3
    assert counts[GeneCategory.CONTROL_FLOW] == 2
    assert counts[GeneCategory.TOOL_MEMORY] == 2
    assert counts[GeneCategory.COMPUTE_BUDGET] == 2


def test_enum_legal_values_counts():
    """Spot-check a few legal-value set sizes from docs/04_genome_schema.md."""
    for b in BENCHMARKS:
        sp = legal_values_for("system_prompt", b)
        assert len(sp) == 5, f"system_prompt for {b} should have 5 values"
        af = legal_values_for("answer_format", b)
        assert len(af) == 4, f"answer_format for {b} should have 4 values"
    # Universal genes
    assert len(legal_values_for("decomposition_style", "forecastbench")) == 5
    assert len(legal_values_for("self_critique", "forecastbench")) == 4
    assert len(legal_values_for("stopping_policy", "forecastbench")) == 4
    assert len(legal_values_for("memory_structure", "forecastbench")) == 3
    assert len(legal_values_for("prompt_token_budget", "forecastbench")) == 4


def test_temperature_range_is_zero_to_one_point_two():
    low, high = legal_values_for("temperature", "forecastbench")
    assert low == 0.0
    assert high == 1.2


# ---------------------------------------------------------------------------
# Default genome tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("benchmark", list(BENCHMARKS))
def test_default_genome_validates(benchmark):
    g = default_genome_for(benchmark)
    validate_genome(g)  # should not raise
    assert g.benchmark == benchmark
    assert tuple(n for n, _ in g.values) == GENE_NAMES


@pytest.mark.parametrize("benchmark", list(BENCHMARKS))
def test_default_genome_has_sensible_temperature(benchmark):
    g = default_genome_for(benchmark)
    t = g.get("temperature")
    assert 0.0 <= float(t) <= 1.2


# ---------------------------------------------------------------------------
# Random genome tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("benchmark", list(BENCHMARKS))
@pytest.mark.parametrize("seed", [0, 1, 42, 2024])
def test_random_genome_validates(benchmark, seed):
    rng = random.Random(seed)
    g = random_genome_for(benchmark, rng)
    validate_genome(g)
    assert g.benchmark == benchmark


@pytest.mark.parametrize("benchmark", list(BENCHMARKS))
def test_random_genome_covers_space_across_many_seeds(benchmark):
    """Sample 200 random genomes and verify we see multiple values for at least one gene."""
    seen_values = {name: set() for name in GENE_NAMES}
    for seed in range(200):
        rng = random.Random(seed)
        g = random_genome_for(benchmark, rng)
        for name, val in g.values:
            # Convert unhashables
            if isinstance(val, frozenset):
                seen_values[name].add(tuple(sorted(val)))
            else:
                seen_values[name].add(val)
    # Every ENUM gene should have at least 2 distinct values after 200 samples
    for gene in GENES:
        if gene.type == GeneType.ENUM:
            assert len(seen_values[gene.name]) >= 2, (
                f"gene {gene.name} did not explore multiple values over 200 samples"
            )


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

def test_validator_rejects_wrong_benchmark():
    from real_study.harness.genome.validator import GenomeValidationError
    g = default_genome_for("forecastbench")
    bad = Genome(benchmark="nonsense", values=g.values)
    with pytest.raises(GenomeValidationError):
        validate_genome(bad)


def test_validator_rejects_illegal_enum_value():
    from real_study.harness.genome.validator import GenomeValidationError
    g = default_genome_for("forecastbench")
    # Override system_prompt to something illegal
    bad = g.with_override("system_prompt", "not_a_real_template")
    with pytest.raises(GenomeValidationError):
        validate_genome(bad)


def test_validator_rejects_out_of_range_temperature():
    from real_study.harness.genome.validator import GenomeValidationError
    g = default_genome_for("forecastbench")
    bad = g.with_override("temperature", 99.9)
    with pytest.raises(GenomeValidationError):
        validate_genome(bad)


# ---------------------------------------------------------------------------
# Content hash tests
# ---------------------------------------------------------------------------

def test_content_hash_is_stable_across_construction():
    """Two genomes with the same values produce the same hash."""
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    g1 = random_genome_for("forecastbench", rng1)
    g2 = random_genome_for("forecastbench", rng2)
    assert genome_content_hash(g1) == genome_content_hash(g2)


def test_content_hash_differs_across_benchmarks():
    g_fcb = default_genome_for("forecastbench")
    g_gpqa = default_genome_for("gpqa_diamond")
    assert genome_content_hash(g_fcb) != genome_content_hash(g_gpqa)


def test_content_hash_differs_on_override():
    g = default_genome_for("forecastbench")
    g2 = g.with_override("decomposition_style", "chain_of_thought")
    assert genome_content_hash(g) != genome_content_hash(g2)


def test_content_hash_is_hex_sha256():
    g = default_genome_for("forecastbench")
    h = genome_content_hash(g)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
