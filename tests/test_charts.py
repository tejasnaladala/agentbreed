"""Tests for breed.charts -- chart generation module.

Tests verify that chart functions produce PNG files on disk.
Visual correctness is not tested; only file creation and error
handling are validated.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from breed.genome import Genome, create_random_genome
from breed.lineage import LineageTracker


# ---------------------------------------------------------------------------
# Helpers -- synthetic lineage data
# ---------------------------------------------------------------------------


def _make_genome(
    genome_id: str,
    generation: int,
    parents: list[str] | None = None,
    seed: int | None = None,
) -> Genome:
    """Create a minimal Genome for chart tests."""
    g = create_random_genome(seed=seed or (hash(genome_id) % (2**31)))
    return g.model_copy(update={
        "genome_id": genome_id,
        "generation": generation,
        "parents": parents or [],
    })


def _populate_tracker(tracker: LineageTracker, *, generations: int = 5, pop_size: int = 10) -> None:
    """Write synthetic evolution data to *tracker*.

    Creates *generations* generations of *pop_size* genomes each, with
    increasing fitness over time to simulate realistic evolution.
    """
    import random

    rng = random.Random(42)
    prev_ids: list[str] = []

    for gen in range(generations):
        current_ids: list[str] = []
        for idx in range(pop_size):
            gid = f"g{gen}-{idx}"
            current_ids.append(gid)

            parents: list[str] = []
            if prev_ids:
                parents = rng.sample(prev_ids, min(2, len(prev_ids)))

            genome = _make_genome(gid, generation=gen, parents=parents, seed=gen * 100 + idx)
            base_fitness = 0.3 + gen * 0.1
            noise = rng.gauss(0, 0.05)
            fitness = max(0.0, min(1.0, base_fitness + noise))

            tracker.log(
                genome,
                fitness_score=round(fitness, 4),
                fitness_breakdown={"accuracy": round(fitness * 0.9, 4)},
                survived=fitness > base_fitness - 0.05,
                crossover_op="uniform_crossover" if parents else None,
                mutations_applied=["parameter_jitter"] if rng.random() > 0.5 else [],
            )

        prev_ids = current_ids


@pytest.fixture()
def populated_tracker(tmp_path: Path) -> LineageTracker:
    """Return a tracker pre-loaded with 5 generations of 10 genomes."""
    tracker = LineageTracker(tmp_path / "lineage.jsonl")
    _populate_tracker(tracker, generations=5, pop_size=10)
    return tracker


# ---------------------------------------------------------------------------
# Tests -- fitness_curve
# ---------------------------------------------------------------------------


class TestFitnessCurve:
    def test_creates_png_file(self, populated_tracker: LineageTracker, tmp_path: Path) -> None:
        from breed.charts import fitness_curve

        out = str(tmp_path / "charts" / "fitness.png")
        result = fitness_curve(populated_tracker, out)

        assert result == out
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    def test_returns_output_path(self, populated_tracker: LineageTracker, tmp_path: Path) -> None:
        from breed.charts import fitness_curve

        out = str(tmp_path / "fc.png")
        assert fitness_curve(populated_tracker, out) == out

    def test_raises_on_empty_tracker(self, tmp_path: Path) -> None:
        from breed.charts import fitness_curve

        empty_tracker = LineageTracker(tmp_path / "empty.jsonl")
        with pytest.raises(ValueError, match="No records"):
            fitness_curve(empty_tracker, str(tmp_path / "nope.png"))


# ---------------------------------------------------------------------------
# Tests -- population_distribution
# ---------------------------------------------------------------------------


class TestPopulationDistribution:
    def test_creates_png_file(self, populated_tracker: LineageTracker, tmp_path: Path) -> None:
        from breed.charts import population_distribution

        out = str(tmp_path / "pop_dist.png")
        result = population_distribution(populated_tracker, 0, out)

        assert result == out
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    def test_raises_on_missing_generation(self, populated_tracker: LineageTracker, tmp_path: Path) -> None:
        from breed.charts import population_distribution

        with pytest.raises(ValueError, match="No records for generation"):
            population_distribution(populated_tracker, 999, str(tmp_path / "nope.png"))


# ---------------------------------------------------------------------------
# Tests -- gene_persistence
# ---------------------------------------------------------------------------


class TestGenePersistence:
    def test_creates_png_file(self, populated_tracker: LineageTracker, tmp_path: Path) -> None:
        from breed.charts import gene_persistence

        out = str(tmp_path / "gene_persist.png")
        result = gene_persistence(populated_tracker, out)

        assert result == out
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0


# ---------------------------------------------------------------------------
# Tests -- lineage_tree_static
# ---------------------------------------------------------------------------


class TestLineageTreeStatic:
    def test_creates_png_file(self, populated_tracker: LineageTracker, tmp_path: Path) -> None:
        from breed.charts import lineage_tree_static

        out = str(tmp_path / "tree.png")
        result = lineage_tree_static(populated_tracker, out)

        assert result == out
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    def test_raises_on_empty_tracker(self, tmp_path: Path) -> None:
        from breed.charts import lineage_tree_static

        empty_tracker = LineageTracker(tmp_path / "empty.jsonl")
        with pytest.raises(ValueError, match="No records"):
            lineage_tree_static(empty_tracker, str(tmp_path / "nope.png"))


# ---------------------------------------------------------------------------
# Tests -- generate_all_charts
# ---------------------------------------------------------------------------


class TestGenerateAllCharts:
    def test_creates_multiple_files(self, populated_tracker: LineageTracker, tmp_path: Path) -> None:
        from breed.charts import generate_all_charts

        out_dir = str(tmp_path / "all_charts")
        paths = generate_all_charts(populated_tracker, out_dir)

        assert len(paths) >= 3  # fitness_curve + at least 1 pop dist + tree
        for p in paths:
            assert Path(p).exists()
            assert Path(p).stat().st_size > 0

    def test_creates_output_directory(self, populated_tracker: LineageTracker, tmp_path: Path) -> None:
        from breed.charts import generate_all_charts

        out_dir = str(tmp_path / "nested" / "deep" / "charts")
        paths = generate_all_charts(populated_tracker, out_dir)

        assert Path(out_dir).is_dir()
        assert len(paths) > 0


# ---------------------------------------------------------------------------
# Tests -- ImportError guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_fitness_curve_raises_import_error(self, tmp_path: Path) -> None:
        """Verify graceful ImportError when matplotlib is not installed."""
        import breed.charts as charts_mod

        # Temporarily make matplotlib un-importable
        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _mock_import(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("mocked missing matplotlib")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            tracker = LineageTracker(tmp_path / "lineage.jsonl")
            with pytest.raises(ImportError, match="agentbreed\\[charts\\]"):
                charts_mod.fitness_curve(tracker, str(tmp_path / "nope.png"))

    def test_population_distribution_raises_import_error(self, tmp_path: Path) -> None:
        import breed.charts as charts_mod

        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _mock_import(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("mocked missing matplotlib")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            tracker = LineageTracker(tmp_path / "lineage.jsonl")
            with pytest.raises(ImportError, match="agentbreed\\[charts\\]"):
                charts_mod.population_distribution(tracker, 0, str(tmp_path / "nope.png"))
