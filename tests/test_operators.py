"""Tests for breed.operators -- crossover and mutation operators."""

from __future__ import annotations

import pytest

from breed.genome import GeneType, Genome, create_random_genome
from breed.operators.crossover import (
    component_swap,
    uniform_crossover,
    weighted_crossover,
)
from breed.operators.mutation import (
    calibration_adjustment,
    parameter_jitter,
    prompt_perturb_simple,
    strategy_mutation,
    tool_swap,
    vector_jitter,
)


# ── Crossover helpers ────────────────────────────────────────────────────

def _make_parents(seed_a: int = 1, seed_b: int = 2) -> tuple[Genome, Genome]:
    return create_random_genome(seed=seed_a), create_random_genome(seed=seed_b)


# ── uniform_crossover ───────────────────────────────────────────────────

class TestUniformCrossover:
    """Verify uniform_crossover produces valid offspring."""

    def test_child_has_correct_parents(self) -> None:
        a, b = _make_parents()
        child = uniform_crossover(a, b, seed=42)
        assert child.parents == [a.genome_id, b.genome_id]

    def test_child_generation_incremented(self) -> None:
        a, b = _make_parents()
        child = uniform_crossover(a, b, seed=42)
        assert child.generation == max(a.generation, b.generation) + 1

    def test_child_has_new_id(self) -> None:
        a, b = _make_parents()
        child = uniform_crossover(a, b, seed=42)
        assert child.genome_id != a.genome_id
        assert child.genome_id != b.genome_id

    def test_child_genes_come_from_parents(self) -> None:
        a, b = _make_parents()
        child = uniform_crossover(a, b, seed=42)
        for gene_name, child_gene in child.genes.items():
            parent_values = set()
            if gene_name in a.genes:
                parent_values.add(repr(a.genes[gene_name].value))
            if gene_name in b.genes:
                parent_values.add(repr(b.genes[gene_name].value))
            assert repr(child_gene.value) in parent_values

    def test_child_has_all_genes(self) -> None:
        a, b = _make_parents()
        child = uniform_crossover(a, b, seed=42)
        assert set(child.genes.keys()) == set(a.genes.keys()) | set(b.genes.keys())

    def test_deterministic_with_seed(self) -> None:
        a, b = _make_parents()
        c1 = uniform_crossover(a, b, seed=100)
        c2 = uniform_crossover(a, b, seed=100)
        for name in c1.genes:
            assert c1.genes[name].value == c2.genes[name].value


# ── component_swap ───────────────────────────────────────────────────────

class TestComponentSwap:
    """Verify component_swap produces valid offspring."""

    def test_child_has_correct_parents(self) -> None:
        a, b = _make_parents()
        child = component_swap(a, b, seed=42)
        assert child.parents == [a.genome_id, b.genome_id]

    def test_child_generation_incremented(self) -> None:
        a, b = _make_parents()
        child = component_swap(a, b, seed=42)
        assert child.generation == 1

    def test_child_has_all_genes(self) -> None:
        a, b = _make_parents()
        child = component_swap(a, b, seed=42)
        assert set(child.genes.keys()) == set(a.genes.keys())

    def test_child_has_new_id(self) -> None:
        a, b = _make_parents()
        child = component_swap(a, b, seed=42)
        assert child.genome_id != a.genome_id
        assert child.genome_id != b.genome_id


# ── weighted_crossover ───────────────────────────────────────────────────

class TestWeightedCrossover:
    """Verify weighted_crossover produces valid offspring biased toward fitter parent."""

    def test_child_has_correct_parents(self) -> None:
        a, b = _make_parents()
        child = weighted_crossover(a, b, 0.9, 0.1, seed=42)
        assert child.parents == [a.genome_id, b.genome_id]

    def test_child_generation_incremented(self) -> None:
        a, b = _make_parents()
        child = weighted_crossover(a, b, 0.5, 0.5, seed=42)
        assert child.generation == 1

    def test_child_has_new_id(self) -> None:
        a, b = _make_parents()
        child = weighted_crossover(a, b, 0.5, 0.5, seed=42)
        assert child.genome_id != a.genome_id

    def test_biases_toward_fitter_parent(self) -> None:
        """Over 100 trials, discrete genes should come from the fitter parent >60% of the time."""
        a, b = _make_parents(seed_a=10, seed_b=20)
        # Make them have different enum values to detect bias
        from_a_count = 0
        total_discrete = 0

        for trial in range(100):
            child = weighted_crossover(a, b, 0.9, 0.1, seed=trial)
            for gene_name in child.genes:
                gene_a = a.genes.get(gene_name)
                gene_b = b.genes.get(gene_name)
                if gene_a is None or gene_b is None:
                    continue
                # Only check discrete genes where parents differ
                if gene_a.type in (GeneType.TEXT, GeneType.ENUM, GeneType.SET):
                    if repr(gene_a.value) != repr(gene_b.value):
                        total_discrete += 1
                        if repr(child.genes[gene_name].value) == repr(gene_a.value):
                            from_a_count += 1

        if total_discrete > 0:
            ratio = from_a_count / total_discrete
            assert ratio > 0.60, (
                f"Expected fitter parent (A) to contribute >60%% of discrete genes, "
                f"got {ratio:.2%} ({from_a_count}/{total_discrete})"
            )

    def test_numeric_genes_blended(self) -> None:
        """Float genes should be blended between parents, not just copied."""
        a, b = _make_parents(seed_a=10, seed_b=20)
        child = weighted_crossover(a, b, 0.7, 0.3, seed=42)
        # temperature is FLOAT
        temp_a = a.genes["temperature"].value
        temp_b = b.genes["temperature"].value
        temp_child = child.genes["temperature"].value
        if temp_a != temp_b:
            # Child should be between parents (or very close to weighted blend)
            lo = min(temp_a, temp_b)
            hi = max(temp_a, temp_b)
            assert lo <= temp_child <= hi


# ── parameter_jitter ─────────────────────────────────────────────────────

class TestParameterJitter:
    """Verify parameter_jitter on FLOAT genes."""

    def test_changes_value(self) -> None:
        genome = create_random_genome(seed=42)
        mutated = parameter_jitter(genome, "temperature", strength=0.5, seed=1)
        assert mutated.genes["temperature"].value != genome.genes["temperature"].value

    def test_stays_in_range(self) -> None:
        genome = create_random_genome(seed=42)
        for seed in range(50):
            mutated = parameter_jitter(genome, "temperature", strength=1.0, seed=seed)
            val = mutated.genes["temperature"].value
            lo, hi = mutated.genes["temperature"].range
            assert lo <= val <= hi

    def test_returns_new_genome(self) -> None:
        genome = create_random_genome(seed=42)
        mutated = parameter_jitter(genome, "temperature", seed=1)
        assert mutated.genome_id != genome.genome_id

    def test_preserves_other_genes(self) -> None:
        genome = create_random_genome(seed=42)
        mutated = parameter_jitter(genome, "temperature", seed=1)
        for name in genome.genes:
            if name != "temperature":
                assert mutated.genes[name].value == genome.genes[name].value

    def test_rejects_non_float_gene(self) -> None:
        genome = create_random_genome(seed=42)
        with pytest.raises(ValueError, match="expected FLOAT"):
            parameter_jitter(genome, "memory_structure", seed=1)


# ── tool_swap ────────────────────────────────────────────────────────────

class TestToolSwap:
    """Verify tool_swap on the tool_policy SET gene."""

    def test_changes_tool_set(self) -> None:
        genome = create_random_genome(seed=42)
        original_tools = set(genome.genes["tool_policy"].value)
        # Try multiple seeds to find one that actually changes the set
        changed = False
        for seed in range(20):
            mutated = tool_swap(genome, seed=seed)
            if set(mutated.genes["tool_policy"].value) != original_tools:
                changed = True
                break
        assert changed, "tool_swap should change the tool set for at least one seed"

    def test_result_tools_are_valid(self) -> None:
        genome = create_random_genome(seed=42)
        available = set(genome.genes["tool_policy"].available)
        for seed in range(20):
            mutated = tool_swap(genome, seed=seed)
            assert set(mutated.genes["tool_policy"].value).issubset(available)

    def test_returns_new_genome(self) -> None:
        genome = create_random_genome(seed=42)
        mutated = tool_swap(genome, seed=1)
        assert mutated.genome_id != genome.genome_id

    def test_preserves_other_genes(self) -> None:
        genome = create_random_genome(seed=42)
        mutated = tool_swap(genome, seed=1)
        for name in genome.genes:
            if name != "tool_policy":
                assert mutated.genes[name].value == genome.genes[name].value


# ── strategy_mutation ────────────────────────────────────────────────────

class TestStrategyMutation:
    """Verify strategy_mutation on ENUM genes."""

    def test_changes_enum_value(self) -> None:
        genome = create_random_genome(seed=42)
        original = genome.genes["memory_structure"].value
        mutated = strategy_mutation(genome, "memory_structure", seed=1)
        assert mutated.genes["memory_structure"].value != original

    def test_new_value_in_options(self) -> None:
        genome = create_random_genome(seed=42)
        mutated = strategy_mutation(genome, "memory_structure", seed=1)
        assert mutated.genes["memory_structure"].value in genome.genes["memory_structure"].options

    def test_returns_new_genome(self) -> None:
        genome = create_random_genome(seed=42)
        mutated = strategy_mutation(genome, "memory_structure", seed=1)
        assert mutated.genome_id != genome.genome_id

    def test_rejects_non_enum_gene(self) -> None:
        genome = create_random_genome(seed=42)
        with pytest.raises(ValueError, match="expected ENUM"):
            strategy_mutation(genome, "temperature", seed=1)


# ── prompt_perturb_simple ────────────────────────────────────────────────

class TestPromptPerturbSimple:
    """Verify prompt_perturb_simple on the prompt_template TEXT gene."""

    def test_changes_prompt(self) -> None:
        genome = create_random_genome(seed=42)
        original = genome.genes["prompt_template"].value
        mutated = prompt_perturb_simple(genome, seed=1)
        assert mutated.genes["prompt_template"].value != original

    def test_returns_new_genome(self) -> None:
        genome = create_random_genome(seed=42)
        mutated = prompt_perturb_simple(genome, seed=1)
        assert mutated.genome_id != genome.genome_id

    def test_preserves_other_genes(self) -> None:
        genome = create_random_genome(seed=42)
        mutated = prompt_perturb_simple(genome, seed=1)
        for name in genome.genes:
            if name != "prompt_template":
                assert mutated.genes[name].value == genome.genes[name].value


# ── vector_jitter ────────────────────────────────────────────────────────

class TestVectorJitter:
    """Verify vector_jitter on NUMERIC_VECTOR genes."""

    def test_changes_vector(self) -> None:
        genome = create_random_genome(seed=42)
        mutated = vector_jitter(genome, "evidence_weighting", strength=0.5, seed=1)
        assert mutated.genes["evidence_weighting"].value != genome.genes["evidence_weighting"].value

    def test_vector_sums_to_one(self) -> None:
        genome = create_random_genome(seed=42)
        for seed in range(20):
            mutated = vector_jitter(genome, "evidence_weighting", strength=0.3, seed=seed)
            total = sum(mutated.genes["evidence_weighting"].value)
            assert abs(total - 1.0) < 0.01

    def test_vector_elements_non_negative(self) -> None:
        genome = create_random_genome(seed=42)
        for seed in range(20):
            mutated = vector_jitter(genome, "evidence_weighting", strength=0.5, seed=seed)
            for v in mutated.genes["evidence_weighting"].value:
                assert v >= 0.0

    def test_returns_new_genome(self) -> None:
        genome = create_random_genome(seed=42)
        mutated = vector_jitter(genome, "evidence_weighting", seed=1)
        assert mutated.genome_id != genome.genome_id

    def test_rejects_non_vector_gene(self) -> None:
        genome = create_random_genome(seed=42)
        with pytest.raises(ValueError, match="expected NUMERIC_VECTOR"):
            vector_jitter(genome, "temperature", seed=1)


# ── calibration_adjustment ───────────────────────────────────────────────

class TestCalibrationAdjustment:
    """Verify calibration_adjustment on the calibration_rule TEXT gene."""

    def test_changes_calibration(self) -> None:
        genome = create_random_genome(seed=42)
        original = genome.genes["calibration_rule"].value
        mutated = calibration_adjustment(genome, seed=1)
        assert mutated.genes["calibration_rule"].value != original

    def test_returns_new_genome(self) -> None:
        genome = create_random_genome(seed=42)
        mutated = calibration_adjustment(genome, seed=1)
        assert mutated.genome_id != genome.genome_id

    def test_preserves_other_genes(self) -> None:
        genome = create_random_genome(seed=42)
        mutated = calibration_adjustment(genome, seed=1)
        for name in genome.genes:
            if name != "calibration_rule":
                assert mutated.genes[name].value == genome.genes[name].value


# ── Mutation immutability ────────────────────────────────────────────────

class TestMutationImmutability:
    """Verify that no mutation operator modifies the original genome."""

    def test_parameter_jitter_does_not_mutate_original(self) -> None:
        genome = create_random_genome(seed=42)
        original_val = genome.genes["temperature"].value
        _ = parameter_jitter(genome, "temperature", seed=1)
        assert genome.genes["temperature"].value == original_val

    def test_tool_swap_does_not_mutate_original(self) -> None:
        genome = create_random_genome(seed=42)
        original_tools = list(genome.genes["tool_policy"].value)
        _ = tool_swap(genome, seed=1)
        assert genome.genes["tool_policy"].value == original_tools

    def test_strategy_mutation_does_not_mutate_original(self) -> None:
        genome = create_random_genome(seed=42)
        original_val = genome.genes["memory_structure"].value
        _ = strategy_mutation(genome, "memory_structure", seed=1)
        assert genome.genes["memory_structure"].value == original_val

    def test_prompt_perturb_does_not_mutate_original(self) -> None:
        genome = create_random_genome(seed=42)
        original_val = genome.genes["prompt_template"].value
        _ = prompt_perturb_simple(genome, seed=1)
        assert genome.genes["prompt_template"].value == original_val
