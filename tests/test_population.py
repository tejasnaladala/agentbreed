"""Tests for breed.population.Population.

Uses the real Genome, crossover, mutation, and selection implementations
since they exist in the codebase.
"""

from __future__ import annotations

import pytest

from breed.population import Population


# ── spawn ────────────────────────────────────────────────────────────────────


class TestSpawn:
    def test_creates_correct_size(self) -> None:
        pop = Population.spawn(10, seed=42)
        assert pop.size == 10

    def test_unique_ids(self) -> None:
        pop = Population.spawn(20, seed=7)
        ids = [g.genome_id for g in pop.members]
        assert len(set(ids)) == 20

    def test_generation_is_zero(self) -> None:
        pop = Population.spawn(5, seed=1)
        assert pop.generation == 0

    def test_members_have_genes(self) -> None:
        pop = Population.spawn(3, seed=99)
        for genome in pop.members:
            assert len(genome.genes) > 0

    def test_invalid_size_raises(self) -> None:
        with pytest.raises(ValueError):
            Population.spawn(0)


# ── select ───────────────────────────────────────────────────────────────────


class TestSelect:
    def test_reduces_size_tournament(self) -> None:
        pop = Population.spawn(10, seed=42)
        scores = [float(i) for i in range(10)]
        pop.select(scores, top_k=4, method="tournament", seed=1)
        assert pop.size == 4

    def test_reduces_size_truncation(self) -> None:
        pop = Population.spawn(10, seed=42)
        scores = [float(i) for i in range(10)]
        pop.select(scores, top_k=3, method="truncation")
        assert pop.size == 3

    def test_truncation_keeps_fittest(self) -> None:
        pop = Population.spawn(5, seed=42)
        # Assign ascending scores so last member is fittest
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        best_id = pop.members[4].genome_id
        pop.select(scores, top_k=1, method="truncation")
        assert pop.size == 1
        assert pop.members[0].genome_id == best_id

    def test_mismatched_scores_raises(self) -> None:
        pop = Population.spawn(5, seed=1)
        with pytest.raises(ValueError, match="length"):
            pop.select([1.0, 2.0], top_k=2)

    def test_invalid_top_k_raises(self) -> None:
        pop = Population.spawn(5, seed=1)
        with pytest.raises(ValueError):
            pop.select([1.0] * 5, top_k=0)

    def test_unknown_method_raises(self) -> None:
        pop = Population.spawn(5, seed=1)
        with pytest.raises(ValueError, match="Unknown"):
            pop.select([1.0] * 5, top_k=2, method="bogus")  # type: ignore[arg-type]


# ── breed ────────────────────────────────────────────────────────────────────


class TestBreed:
    def test_produces_correct_size(self) -> None:
        pop = Population.spawn(4, seed=42)
        pop.breed(target_size=10, seed=99)
        assert pop.size == 10

    def test_elites_survive(self) -> None:
        pop = Population.spawn(3, seed=42)
        original_ids = {g.genome_id for g in pop.members}
        pop.breed(target_size=8, seed=1)
        # First 3 members should be the unchanged elites
        surviving_ids = {g.genome_id for g in pop.members[:3]}
        assert original_ids == surviving_ids

    def test_children_have_parents(self) -> None:
        pop = Population.spawn(3, seed=42)
        original_ids = {g.genome_id for g in pop.members}
        pop.breed(target_size=6, seed=1)
        children = pop.members[3:]
        for child in children:
            assert len(child.parents) == 2
            for p in child.parents:
                assert p in original_ids

    def test_generation_increments(self) -> None:
        pop = Population.spawn(4, seed=42)
        assert pop.generation == 0
        pop.breed(target_size=8, seed=1)
        assert pop.generation == 1
        # Select down and breed again
        scores = [float(i) for i in range(pop.size)]
        pop.select(scores, top_k=4, method="truncation")
        pop.breed(target_size=8, seed=2)
        assert pop.generation == 2

    def test_breed_with_fitness_scores(self) -> None:
        """weighted_crossover path should not error when scores provided."""
        pop = Population.spawn(4, seed=42)
        scores = [0.5, 0.6, 0.7, 0.8]
        pop.breed(target_size=10, seed=1, fitness_scores=scores)
        assert pop.size == 10

    def test_target_smaller_than_current_raises(self) -> None:
        pop = Population.spawn(5, seed=1)
        with pytest.raises(ValueError, match="target_size"):
            pop.breed(target_size=3, seed=1)


# ── mutate ───────────────────────────────────────────────────────────────────


class TestMutate:
    def test_preserves_population_size(self) -> None:
        pop = Population.spawn(10, seed=42)
        pop.mutate(rate=0.5, seed=1)
        assert pop.size == 10

    def test_rate_zero_preserves_all(self) -> None:
        pop = Population.spawn(10, seed=42)
        original_ids = [g.genome_id for g in pop.members]
        pop.mutate(rate=0.0, seed=1)
        after_ids = [g.genome_id for g in pop.members]
        assert original_ids == after_ids

    def test_rate_one_mutates_all(self) -> None:
        pop = Population.spawn(10, seed=42)
        original_ids = set(g.genome_id for g in pop.members)
        pop.mutate(rate=1.0, seed=1)
        after_ids = set(g.genome_id for g in pop.members)
        # Mutation clones with a fresh UUID, so ids should change
        assert original_ids != after_ids

    def test_partial_rate_changes_some(self) -> None:
        pop = Population.spawn(50, seed=42)
        original_ids = set(g.genome_id for g in pop.members)
        pop.mutate(rate=0.5, seed=1)
        after_ids = set(g.genome_id for g in pop.members)
        # Some should be different, some the same
        assert after_ids != original_ids
        # At least one should survive unchanged (probabilistic but with 50
        # members and 50% rate, extremely unlikely all mutate)
        assert len(original_ids & after_ids) > 0

    def test_invalid_rate_raises(self) -> None:
        pop = Population.spawn(5, seed=1)
        with pytest.raises(ValueError):
            pop.mutate(rate=1.5)
        with pytest.raises(ValueError):
            pop.mutate(rate=-0.1)


# ── add_immigrants ───────────────────────────────────────────────────────────


class TestAddImmigrants:
    def test_increases_size(self) -> None:
        pop = Population.spawn(5, seed=42)
        pop.add_immigrants(3, seed=1)
        assert pop.size == 8

    def test_zero_immigrants(self) -> None:
        pop = Population.spawn(5, seed=42)
        pop.add_immigrants(0, seed=1)
        assert pop.size == 5

    def test_original_members_preserved(self) -> None:
        pop = Population.spawn(5, seed=42)
        original_ids = [g.genome_id for g in pop.members]
        pop.add_immigrants(3, seed=1)
        surviving_ids = [g.genome_id for g in pop.members[:5]]
        assert surviving_ids == original_ids

    def test_negative_count_raises(self) -> None:
        pop = Population.spawn(5, seed=1)
        with pytest.raises(ValueError):
            pop.add_immigrants(-1)


# ── size property ────────────────────────────────────────────────────────────


class TestSizeProperty:
    def test_matches_members_length(self) -> None:
        pop = Population.spawn(7, seed=42)
        assert pop.size == len(pop.members)
