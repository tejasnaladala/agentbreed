"""Tests for paper-quality figure generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from breed.paper_figures import (
    MethodCurve,
    MethodScores,
    aggregate_curves_across_seeds,
    aggregate_scores_across_seeds,
    fig_ablation_bars,
    fig_cost_performance,
    fig_fitness_curves,
    fig_gene_persistence,
    fig_lineage_tree,
    fig_reliability_diagram,
)


@pytest.fixture
def sample_curves() -> list[MethodCurve]:
    """Three methods with different fitness trajectories."""
    rng = np.random.default_rng(42)
    n_gens = 10
    n_seeds = 5

    curves = []
    for method, base in [
        ("full_evolution", 0.5),
        ("mutation_only", 0.45),
        ("random_search", 0.4),
    ]:
        per_seed = []
        for s in range(n_seeds):
            # Increasing fitness with noise
            curve = base + np.linspace(0, 0.3, n_gens) + rng.normal(0, 0.02, n_gens)
            per_seed.append(curve.tolist())
        curves.append(aggregate_curves_across_seeds(
            method=method, per_seed_curves=per_seed, n_bootstrap=200, seed=s,
        ))
    return curves


@pytest.fixture
def sample_scores() -> list[MethodScores]:
    rng = np.random.default_rng(42)
    return [
        aggregate_scores_across_seeds(
            "full_evolution",
            per_seed_scores=rng.normal(0.78, 0.03, 5).tolist(),
            n_bootstrap=200, seed=1,
        ),
        aggregate_scores_across_seeds(
            "mutation_only",
            per_seed_scores=rng.normal(0.74, 0.03, 5).tolist(),
            n_bootstrap=200, seed=2,
        ),
        aggregate_scores_across_seeds(
            "random_search",
            per_seed_scores=rng.normal(0.70, 0.03, 5).tolist(),
            n_bootstrap=200, seed=3,
        ),
    ]


class TestFitnessCurves:
    def test_creates_pdf_and_png(self, sample_curves: list[MethodCurve], tmp_path: Path) -> None:
        pdf_path = tmp_path / "fig1.pdf"
        png_path = tmp_path / "fig1.png"
        result_pdf = fig_fitness_curves(sample_curves, pdf_path)
        result_png = fig_fitness_curves(sample_curves, png_path)
        assert Path(result_pdf).exists()
        assert Path(result_png).exists()
        assert pdf_path.stat().st_size > 0
        assert png_path.stat().st_size > 0

    def test_with_title(self, sample_curves: list[MethodCurve], tmp_path: Path) -> None:
        path = tmp_path / "fig1.pdf"
        fig_fitness_curves(sample_curves, path, title="Test title")
        assert path.exists()


class TestReliabilityDiagram:
    def test_creates_file(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        preds = rng.uniform(0, 1, 100).tolist()
        outs = [1.0 if rng.random() < p else 0.0 for p in preds]
        path = tmp_path / "reliability.pdf"
        fig_reliability_diagram(
            methods_data={
                "full_evolution": (preds, outs),
                "mutation_only": (preds, outs),
            },
            output_path=path,
        )
        assert path.exists()


class TestAblationBars:
    def test_creates_file(self, sample_scores: list[MethodScores], tmp_path: Path) -> None:
        path = tmp_path / "ablation.pdf"
        fig_ablation_bars(sample_scores, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_sort_by_mean(self, sample_scores: list[MethodScores], tmp_path: Path) -> None:
        path = tmp_path / "ablation.pdf"
        fig_ablation_bars(sample_scores, path, sort_by_mean=True)
        assert path.exists()


class TestCostPerformance:
    def test_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "cost.pdf"
        fig_cost_performance(
            method_scores={
                "full_evolution": (1000.0, 0.78, 50.0, 0.02),
                "mutation_only": (1000.0, 0.74, 50.0, 0.03),
                "random_search": (1000.0, 0.70, 50.0, 0.04),
            },
            output_path=path,
        )
        assert path.exists()


class TestGenePersistence:
    def test_creates_file(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        # 5 genes x 10 generations of modal-value frequencies
        matrix = rng.uniform(0.2, 1.0, (5, 10))
        names = ["prompt", "calibration", "tools", "temperature", "strategy"]
        path = tmp_path / "gene_persistence.pdf"
        fig_gene_persistence(matrix, names, path)
        assert path.exists()


class TestLineageTree:
    def test_creates_file(self, tmp_path: Path) -> None:
        # Build a small synthetic lineage
        records = [
            {"genome_id": "g0", "generation": 0, "parents": [], "fitness_score": 0.5},
            {"genome_id": "g1", "generation": 0, "parents": [], "fitness_score": 0.6},
            {"genome_id": "g2", "generation": 1, "parents": ["g0", "g1"], "fitness_score": 0.7},
            {"genome_id": "g3", "generation": 2, "parents": ["g2"], "fitness_score": 0.85},
        ]
        path = tmp_path / "lineage.pdf"
        fig_lineage_tree(records, champion_id="g3", output_path=path)
        assert path.exists()


class TestAggregators:
    def test_aggregate_curves(self) -> None:
        per_seed = [
            [0.1, 0.2, 0.3],
            [0.15, 0.25, 0.35],
            [0.05, 0.15, 0.25],
        ]
        curve = aggregate_curves_across_seeds("test", per_seed, n_bootstrap=100, seed=1)
        assert curve.method == "test"
        assert curve.n_seeds == 3
        assert len(curve.generations) == 3
        assert len(curve.mean) == 3
        assert curve.mean[0] == pytest.approx(0.1, abs=0.01)
        assert curve.mean[1] == pytest.approx(0.2, abs=0.01)
        assert curve.mean[2] == pytest.approx(0.3, abs=0.01)

    def test_aggregate_scores(self) -> None:
        per_seed = [0.7, 0.75, 0.8, 0.78, 0.72]
        scores = aggregate_scores_across_seeds(
            "test", per_seed, n_bootstrap=200, seed=1,
        )
        assert scores.method == "test"
        assert scores.mean == pytest.approx(0.75, abs=0.005)
        assert scores.ci_low < scores.mean < scores.ci_high
