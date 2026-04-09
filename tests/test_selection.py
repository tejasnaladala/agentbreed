"""Tests for breed.operators.selection and breed.diversity."""

from __future__ import annotations

import pytest

from breed.diversity import inject_immigrants, novelty_score
from breed.genome import Genome, create_random_genome
from breed.operators.selection import tournament_select, truncation_select


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_population(n: int = 10) -> list[Genome]:
    return [create_random_genome(seed=i) for i in range(n)]


def _make_fitness(population: list[Genome]) -> dict[str, float]:
    """Assign linearly increasing fitness: genome at index i gets fitness i."""
    return {g.genome_id: float(i) for i, g in enumerate(population)}


# ── Tournament selection ─────────────────────────────────────────────────

class TestTournamentSelect:
    """Verify tournament_select returns correct count and favours fitter genomes."""

    def test_returns_correct_count(self) -> None:
        pop = _make_population(10)
        fitness = _make_fitness(pop)
        selected = tournament_select(pop, fitness, k=5, seed=42)
        assert len(selected) == 5

    def test_returns_genomes(self) -> None:
        pop = _make_population(10)
        fitness = _make_fitness(pop)
        selected = tournament_select(pop, fitness, k=3, seed=42)
        for g in selected:
            assert isinstance(g, Genome)

    def test_favors_fitter_genomes(self) -> None:
        """Over many selections, higher-fitness genomes should appear more often."""
        pop = _make_population(10)
        fitness = _make_fitness(pop)
        # The fittest genome has id = pop[9].genome_id, fitness = 9.0
        fittest_id = pop[9].genome_id

        count_fittest = 0
        total_selections = 200
        selected = tournament_select(pop, fitness, k=total_selections, tournament_size=3, seed=42)
        for g in selected:
            if g.genome_id == fittest_id:
                count_fittest += 1

        # With tournament_size=3 and 10 pop, the fittest should be selected
        # more often than uniform (1/10 = 20 expected). We expect significantly more.
        assert count_fittest > 30, (
            f"Expected fittest genome to be selected often, got {count_fittest}/{total_selections}"
        )

    def test_rejects_empty_population(self) -> None:
        with pytest.raises(ValueError, match="empty population"):
            tournament_select([], {}, k=1)

    def test_rejects_k_zero(self) -> None:
        pop = _make_population(5)
        with pytest.raises(ValueError, match="k must be >= 1"):
            tournament_select(pop, {}, k=0)


# ── Truncation selection ─────────────────────────────────────────────────

class TestTruncationSelect:
    """Verify truncation_select returns the top-k by fitness."""

    def test_returns_correct_count(self) -> None:
        pop = _make_population(10)
        fitness = _make_fitness(pop)
        selected = truncation_select(pop, fitness, k=3)
        assert len(selected) == 3

    def test_returns_top_k(self) -> None:
        pop = _make_population(10)
        fitness = _make_fitness(pop)
        selected = truncation_select(pop, fitness, k=3)
        selected_ids = {g.genome_id for g in selected}
        # Top-3 by fitness are pop[9], pop[8], pop[7]
        expected_ids = {pop[i].genome_id for i in [9, 8, 7]}
        assert selected_ids == expected_ids

    def test_sorted_by_fitness_descending(self) -> None:
        pop = _make_population(10)
        fitness = _make_fitness(pop)
        selected = truncation_select(pop, fitness, k=5)
        scores = [fitness[g.genome_id] for g in selected]
        assert scores == sorted(scores, reverse=True)

    def test_rejects_empty_population(self) -> None:
        with pytest.raises(ValueError, match="empty population"):
            truncation_select([], {}, k=1)

    def test_k_larger_than_population(self) -> None:
        pop = _make_population(3)
        fitness = _make_fitness(pop)
        selected = truncation_select(pop, fitness, k=10)
        assert len(selected) == 3  # Returns all available


# ── Novelty score ────────────────────────────────────────────────────────

class TestNoveltyScore:
    """Verify novelty_score produces valid, non-negative scores."""

    def test_scores_non_negative(self) -> None:
        pop = _make_population(5)
        scores = novelty_score(pop)
        for gid, score in scores.items():
            assert score >= 0.0, f"Genome {gid} has negative novelty {score}"

    def test_scores_in_valid_range(self) -> None:
        pop = _make_population(5)
        scores = novelty_score(pop)
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_returns_score_for_each_genome(self) -> None:
        pop = _make_population(5)
        scores = novelty_score(pop)
        assert set(scores.keys()) == {g.genome_id for g in pop}

    def test_single_genome_score_zero(self) -> None:
        pop = _make_population(1)
        scores = novelty_score(pop)
        assert scores[pop[0].genome_id] == 0.0

    def test_identical_genomes_have_zero_novelty(self) -> None:
        genome = create_random_genome(seed=42)
        # Clone the same genome 5 times (same genes, different IDs)
        pop = []
        for _ in range(5):
            clone = Genome.from_dict(genome.to_dict())
            pop.append(clone)
        scores = novelty_score(pop)
        for score in scores.values():
            assert score == pytest.approx(0.0)


# ── Inject immigrants ────────────────────────────────────────────────────

class TestInjectImmigrants:
    """Verify inject_immigrants adds the correct count of genomes."""

    def test_adds_correct_count(self) -> None:
        pop = _make_population(5)
        expanded = inject_immigrants(pop, count=3, seed=42)
        assert len(expanded) == 8

    def test_original_population_preserved(self) -> None:
        pop = _make_population(5)
        original_ids = {g.genome_id for g in pop}
        expanded = inject_immigrants(pop, count=3, seed=42)
        expanded_ids = {g.genome_id for g in expanded}
        assert original_ids.issubset(expanded_ids)

    def test_does_not_mutate_original_list(self) -> None:
        pop = _make_population(5)
        original_len = len(pop)
        _ = inject_immigrants(pop, count=3, seed=42)
        assert len(pop) == original_len

    def test_zero_immigrants(self) -> None:
        pop = _make_population(5)
        expanded = inject_immigrants(pop, count=0, seed=42)
        assert len(expanded) == 5

    def test_immigrants_are_valid_genomes(self) -> None:
        pop = _make_population(3)
        expanded = inject_immigrants(pop, count=5, seed=42)
        for genome in expanded:
            assert isinstance(genome, Genome)
            assert len(genome.genes) > 0

    def test_deterministic_with_seed(self) -> None:
        pop = _make_population(3)
        e1 = inject_immigrants(pop, count=2, seed=100)
        e2 = inject_immigrants(pop, count=2, seed=100)
        # Immigrants should have same gene values
        for g1, g2 in zip(e1[3:], e2[3:]):
            for name in g1.genes:
                assert g1.genes[name].value == g2.genes[name].value
