"""Tests for breed.lineage.LineageTracker.

Uses the real Genome from breed.genome since it exists in the codebase.
"""

from __future__ import annotations

import json
from pathlib import Path


from breed.genome import Genome, create_random_genome
from breed.lineage import LineageTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_genome(
    genome_id: str = "",
    generation: int = 0,
    parents: list[str] | None = None,
) -> Genome:
    """Create a minimal Genome for testing lineage tracking."""
    g = create_random_genome(seed=hash(genome_id or "default") % (2**31))
    # Override fields for test control
    g = g.model_copy(update={
        "genome_id": genome_id or g.genome_id,
        "generation": generation,
        "parents": parents or [],
    })
    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLogWritesJsonl:
    def test_single_record(self, tmp_path: Path) -> None:
        tracker = LineageTracker(tmp_path / "lineage.jsonl")
        genome = _make_genome("agent-001", generation=0)

        record = tracker.log(
            genome,
            fitness_score=0.85,
            fitness_breakdown={"accuracy": 0.9, "calibration": 0.8},
            survived=True,
        )

        assert record.genome_id == "agent-001"
        assert record.fitness_score == 0.85

        # Verify the file contains valid JSON
        lines = (tmp_path / "lineage.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["genome_id"] == "agent-001"
        assert data["fitness_score"] == 0.85
        assert data["survived"] is True

    def test_multiple_records_appended(self, tmp_path: Path) -> None:
        tracker = LineageTracker(tmp_path / "lineage.jsonl")

        for i in range(5):
            genome = _make_genome(f"agent-{i:03d}", generation=0)
            tracker.log(genome, fitness_score=i * 0.2)

        lines = (tmp_path / "lineage.jsonl").read_text().strip().split("\n")
        assert len(lines) == 5

    def test_record_contains_genome_snapshot(self, tmp_path: Path) -> None:
        tracker = LineageTracker(tmp_path / "lineage.jsonl")
        genome = _make_genome("snap-001", generation=1)
        tracker.log(genome, fitness_score=0.5)

        lines = (tmp_path / "lineage.jsonl").read_text().strip().split("\n")
        data = json.loads(lines[0])
        assert "genome_snapshot" in data
        assert data["genome_snapshot"]["genome_id"] == "snap-001"

    def test_record_has_timestamp(self, tmp_path: Path) -> None:
        tracker = LineageTracker(tmp_path / "lineage.jsonl")
        genome = _make_genome("ts-001")
        record = tracker.log(genome, fitness_score=0.5)

        assert record.timestamp  # non-empty
        assert "T" in record.timestamp  # ISO format

    def test_mutations_and_crossover_logged(self, tmp_path: Path) -> None:
        tracker = LineageTracker(tmp_path / "lineage.jsonl")
        genome = _make_genome("mut-001", generation=1, parents=["p1", "p2"])
        record = tracker.log(
            genome,
            fitness_score=0.7,
            crossover_op="uniform_crossover",
            mutations_applied=["parameter_jitter", "tool_swap"],
        )

        assert record.crossover_operator == "uniform_crossover"
        assert record.mutations_applied == ["parameter_jitter", "tool_swap"]


class TestGetChampion:
    def test_returns_best_in_generation(self, tmp_path: Path) -> None:
        tracker = LineageTracker(tmp_path / "lineage.jsonl")

        for i, score in enumerate([0.3, 0.9, 0.6, 0.1]):
            genome = _make_genome(f"gen0-{i}", generation=0)
            tracker.log(genome, fitness_score=score)

        champion = tracker.get_champion(0)
        assert champion is not None
        assert champion.genome_id == "gen0-1"
        assert champion.fitness_score == 0.9

    def test_returns_none_for_missing_generation(self, tmp_path: Path) -> None:
        tracker = LineageTracker(tmp_path / "lineage.jsonl")
        genome = _make_genome("only-gen0", generation=0)
        tracker.log(genome, fitness_score=0.5)

        assert tracker.get_champion(99) is None

    def test_multiple_generations(self, tmp_path: Path) -> None:
        tracker = LineageTracker(tmp_path / "lineage.jsonl")

        # Gen 0
        tracker.log(_make_genome("g0-a", generation=0), fitness_score=0.4)
        tracker.log(_make_genome("g0-b", generation=0), fitness_score=0.8)

        # Gen 1
        tracker.log(_make_genome("g1-a", generation=1), fitness_score=0.95)
        tracker.log(_make_genome("g1-b", generation=1), fitness_score=0.6)

        champ_0 = tracker.get_champion(0)
        champ_1 = tracker.get_champion(1)
        assert champ_0 is not None and champ_0.genome_id == "g0-b"
        assert champ_1 is not None and champ_1.genome_id == "g1-a"


class TestGetLineage:
    def test_traces_ancestry_to_gen0(self, tmp_path: Path) -> None:
        tracker = LineageTracker(tmp_path / "lineage.jsonl")

        # Gen 0 founders
        tracker.log(_make_genome("root-a", generation=0), fitness_score=0.5)
        tracker.log(_make_genome("root-b", generation=0), fitness_score=0.6)

        # Gen 1 child of root-a and root-b
        child = _make_genome("child-1", generation=1, parents=["root-a", "root-b"])
        tracker.log(child, fitness_score=0.7)

        # Gen 2 grandchild
        grandchild = _make_genome(
            "grandchild-1", generation=2, parents=["child-1", "root-a"]
        )
        tracker.log(grandchild, fitness_score=0.8)

        lineage = tracker.get_lineage("grandchild-1")
        lineage_ids = {r.genome_id for r in lineage}
        assert "grandchild-1" in lineage_ids
        assert "child-1" in lineage_ids
        assert "root-a" in lineage_ids
        assert "root-b" in lineage_ids

    def test_unknown_genome_returns_empty(self, tmp_path: Path) -> None:
        tracker = LineageTracker(tmp_path / "lineage.jsonl")
        tracker.log(_make_genome("exists", generation=0), fitness_score=0.5)

        lineage = tracker.get_lineage("does-not-exist")
        assert lineage == []

    def test_gen0_genome_has_self_in_lineage(self, tmp_path: Path) -> None:
        tracker = LineageTracker(tmp_path / "lineage.jsonl")
        tracker.log(_make_genome("founder", generation=0), fitness_score=0.5)

        lineage = tracker.get_lineage("founder")
        assert len(lineage) == 1
        assert lineage[0].genome_id == "founder"


class TestLoad:
    def test_loads_all_records(self, tmp_path: Path) -> None:
        tracker = LineageTracker(tmp_path / "lineage.jsonl")
        for i in range(10):
            tracker.log(_make_genome(f"a-{i}", generation=0), fitness_score=i / 10)

        records = tracker.load()
        assert len(records) == 10
        assert records[0].genome_id == "a-0"
        assert records[9].genome_id == "a-9"

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        filepath = tmp_path / "empty.jsonl"
        filepath.touch()
        tracker = LineageTracker(filepath)
        assert tracker.load() == []

    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        tracker = LineageTracker(tmp_path / "nonexistent.jsonl")
        assert tracker.load() == []


class TestGetTree:
    def test_builds_parent_child_links(self, tmp_path: Path) -> None:
        tracker = LineageTracker(tmp_path / "lineage.jsonl")

        tracker.log(_make_genome("p1", generation=0), fitness_score=0.5)
        tracker.log(_make_genome("p2", generation=0), fitness_score=0.6)
        tracker.log(
            _make_genome("c1", generation=1, parents=["p1", "p2"]),
            fitness_score=0.7,
        )

        tree = tracker.get_tree()
        assert "p1" in tree
        assert "p2" in tree
        assert "c1" in tree
        assert "c1" in tree["p1"].children
        assert "c1" in tree["p2"].children
        assert tree["c1"].parents == ["p1", "p2"]


class TestMultiGeneration:
    def test_full_evolution_cycle(self, tmp_path: Path) -> None:
        """Simulate 3 generations and verify records are consistent."""
        tracker = LineageTracker(tmp_path / "lineage.jsonl")

        # Gen 0
        gen0_ids = [f"g0-{i}" for i in range(4)]
        for gid in gen0_ids:
            tracker.log(
                _make_genome(gid, generation=0),
                fitness_score=0.5,
                survived=True,
            )

        # Gen 1
        gen1_ids = [f"g1-{i}" for i in range(4)]
        for gid in gen1_ids:
            parents = [gen0_ids[0], gen0_ids[1]]
            tracker.log(
                _make_genome(gid, generation=1, parents=parents),
                fitness_score=0.7,
                survived=True,
                crossover_op="uniform_crossover",
            )

        # Gen 2
        gen2_ids = [f"g2-{i}" for i in range(4)]
        for gid in gen2_ids:
            parents = [gen1_ids[0], gen1_ids[1]]
            tracker.log(
                _make_genome(gid, generation=2, parents=parents),
                fitness_score=0.85,
                survived=True,
                crossover_op="component_swap",
                mutations_applied=["parameter_jitter"],
            )

        records = tracker.load()
        assert len(records) == 12  # 4 per generation

        # Champions improve over generations
        champ_0 = tracker.get_champion(0)
        champ_1 = tracker.get_champion(1)
        champ_2 = tracker.get_champion(2)
        assert champ_0 is not None
        assert champ_1 is not None
        assert champ_2 is not None
        assert champ_0.fitness_score <= champ_1.fitness_score <= champ_2.fitness_score

        # Lineage traces back through generations
        lineage = tracker.get_lineage("g2-0")
        lineage_gens = {r.generation for r in lineage}
        assert 0 in lineage_gens
        assert 1 in lineage_gens
        assert 2 in lineage_gens

        # Tree has correct structure
        tree = tracker.get_tree()
        assert len(tree) == 12
