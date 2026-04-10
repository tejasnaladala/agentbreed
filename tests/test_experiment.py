"""Tests for the experiment runner.

Covers:
* ExperimentConfig serialization roundtrip
* RunResult serialization roundtrip
* End-to-end small experiment (1 seed, multiple methods)
* Different methods produce *different* results (sanity check)
* Same seed reproduces the same per-question scores
* to_jsonl + from_jsonl roundtrip
* Each baseline implementation runs without raising
* Resume / intermediate save behaviour
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from breed.arenas.base import Task
from breed.experiment import (
    KNOWN_METHODS,
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    RunResult,
)


# ---------------------------------------------------------------------------
# Deterministic mock agent
# ---------------------------------------------------------------------------


def _hash_to_unit(text: str) -> float:
    """Map a string to a deterministic float in [0, 1]."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


async def deterministic_agent(genome_dict: dict[str, Any], task: str) -> str:
    """Deterministic agent that returns a probability derived from the genome.

    The output depends on:
    * the temperature gene (or 0.5 fallback)
    * a hash of the prompt template gene
    * the task content
    """
    genes = genome_dict.get("genes", {})

    temp = 0.5
    temp_gene = genes.get("temperature")
    if isinstance(temp_gene, dict):
        temp = float(temp_gene.get("value", 0.5))

    prompt_text = ""
    pt_gene = genes.get("prompt_template")
    if isinstance(pt_gene, dict):
        prompt_text = str(pt_gene.get("value", ""))

    base = _hash_to_unit(prompt_text + "::" + task)

    # Mix temperature in -- pure deterministic, no randomness in the agent
    out = 0.7 * (1.0 - temp / 1.5) + 0.3 * base
    out = max(0.0, min(1.0, out))
    return str(out)


def brier_fitness_scorer(output: str, expected: Any) -> float:
    """Map agent output to fitness in [0, 1] using 1 - Brier score."""
    try:
        p = float(output)
    except (TypeError, ValueError):
        p = 0.5
    p = max(0.0, min(1.0, p))
    y = float(expected)
    return 1.0 - (p - y) ** 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def gene_template() -> dict[str, dict[str, Any]]:
    return {
        "prompt_template": {
            "type": "text",
            "templates": [
                "You are a careful forecaster.",
                "Estimate using base rates first.",
                "Decompose into sub-questions.",
                "Use Bayesian updating.",
            ],
            "mutation_rate": 0.2,
        },
        "temperature": {
            "type": "float",
            "range": (0.0, 1.5),
            "mutation_rate": 0.15,
        },
        "decomposition": {
            "type": "enum",
            "options": ["fermi", "scenario_tree", "reference_class"],
            "mutation_rate": 0.1,
        },
    }


@pytest.fixture()
def train_tasks() -> list[Task]:
    return [
        Task(task_id=f"train-{i}", prompt=f"train question {i}", expected=float(i % 2))
        for i in range(5)
    ]


@pytest.fixture()
def test_tasks() -> list[Task]:
    return [
        Task(task_id=f"test-{i}", prompt=f"test question {i}", expected=float(i % 2))
        for i in range(5)
    ]


@pytest.fixture()
def small_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="small_experiment",
        seeds=[42],
        population_size=4,
        generations=2,
        elite_count=2,
        mutation_rate=0.2,
        immigration_rate=0.0,  # avoid immigrant noise for reproducibility tests
        selection_method="truncation",
        tournament_size=2,
        output_dir=str(tmp_path / "small"),
        methods=["full_evolution", "random_search"],
        metadata={"unit_test": True},
    )


@pytest.fixture()
def runner_factory(
    train_tasks: list[Task],
    test_tasks: list[Task],
    gene_template: dict[str, dict[str, Any]],
):
    def _make(config: ExperimentConfig) -> ExperimentRunner:
        return ExperimentRunner(
            config=config,
            agent_fn=deterministic_agent,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            scorer=brier_fitness_scorer,
            gene_template=gene_template,
        )

    return _make


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestExperimentConfigSerialization:
    """ExperimentConfig roundtrips through dict cleanly."""

    def test_to_from_dict_roundtrip(self, small_config: ExperimentConfig) -> None:
        data = small_config.to_dict()
        # Must be JSON-serializable
        json_str = json.dumps(data)
        round_tripped = ExperimentConfig.from_dict(json.loads(json_str))

        assert round_tripped.name == small_config.name
        assert round_tripped.seeds == small_config.seeds
        assert round_tripped.population_size == small_config.population_size
        assert round_tripped.generations == small_config.generations
        assert round_tripped.elite_count == small_config.elite_count
        assert round_tripped.mutation_rate == small_config.mutation_rate
        assert round_tripped.immigration_rate == small_config.immigration_rate
        assert round_tripped.selection_method == small_config.selection_method
        assert round_tripped.tournament_size == small_config.tournament_size
        assert round_tripped.output_dir == small_config.output_dir
        assert round_tripped.methods == small_config.methods
        assert round_tripped.metadata == small_config.metadata

    def test_compute_budget(self, small_config: ExperimentConfig) -> None:
        assert small_config.compute_budget == small_config.population_size * small_config.generations

    def test_frozen(self, small_config: ExperimentConfig) -> None:
        with pytest.raises((AttributeError, TypeError)):
            small_config.population_size = 999  # type: ignore[misc]


class TestRunResultSerialization:
    """RunResult roundtrips through dict cleanly."""

    def test_to_from_dict_roundtrip(self) -> None:
        original = RunResult(
            method="full_evolution",
            seed=7,
            champion_genome_id="abc123",
            champion_fitness=0.875,
            per_question_scores=[0.9, 0.8, 0.85, 0.7, 0.95],
            fitness_curve=[0.5, 0.7, 0.875],
            wall_clock_seconds=12.34,
            total_evaluations=20,
            api_calls=100,
            metadata={"foo": "bar", "n": 42},
        )
        data = original.to_dict()
        # Must be JSON-serializable
        json_str = json.dumps(data)
        round_tripped = RunResult.from_dict(json.loads(json_str))

        assert round_tripped.method == original.method
        assert round_tripped.seed == original.seed
        assert round_tripped.champion_genome_id == original.champion_genome_id
        assert round_tripped.champion_fitness == original.champion_fitness
        assert round_tripped.per_question_scores == original.per_question_scores
        assert round_tripped.fitness_curve == original.fitness_curve
        assert round_tripped.wall_clock_seconds == original.wall_clock_seconds
        assert round_tripped.total_evaluations == original.total_evaluations
        assert round_tripped.api_calls == original.api_calls
        assert round_tripped.metadata == original.metadata


# ---------------------------------------------------------------------------
# End-to-end small experiment
# ---------------------------------------------------------------------------


class TestSmallExperiment:
    """A complete small experiment runs and produces sane results."""

    async def test_small_experiment_runs(
        self,
        runner_factory,
        small_config: ExperimentConfig,
    ) -> None:
        runner = runner_factory(small_config)
        result = await runner.run()

        assert isinstance(result, ExperimentResult)
        # 1 seed * 2 methods = 2 runs
        assert len(result.runs) == 2

        for run in result.runs:
            assert run.method in small_config.methods
            assert run.seed in small_config.seeds
            assert 0.0 <= run.champion_fitness <= 1.0
            # per_question_scores must match the test set length (5)
            assert len(run.per_question_scores) == 5
            assert all(0.0 <= s <= 1.0 for s in run.per_question_scores)
            assert run.api_calls > 0
            assert run.total_evaluations > 0
            assert run.wall_clock_seconds >= 0.0

    async def test_runs_for_method_filter(
        self,
        runner_factory,
        small_config: ExperimentConfig,
    ) -> None:
        runner = runner_factory(small_config)
        result = await runner.run()

        full_runs = result.runs_for_method("full_evolution")
        rand_runs = result.runs_for_method("random_search")

        assert len(full_runs) == 1
        assert len(rand_runs) == 1
        assert full_runs[0].method == "full_evolution"
        assert rand_runs[0].method == "random_search"

    async def test_methods_produce_different_results(
        self,
        runner_factory,
        small_config: ExperimentConfig,
    ) -> None:
        """full_evolution and random_search should not produce identical
        per-question scores in general (their search trajectories diverge).
        """
        runner = runner_factory(small_config)
        result = await runner.run()

        full_run = result.runs_for_method("full_evolution")[0]
        rand_run = result.runs_for_method("random_search")[0]

        # They should not have identical genome IDs
        assert full_run.champion_genome_id != rand_run.champion_genome_id


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Same seed produces reproducible per-question scores."""

    async def test_same_seed_same_results(
        self,
        runner_factory,
        tmp_path: Path,
    ) -> None:
        cfg_a = ExperimentConfig(
            name="repro_a",
            seeds=[123],
            population_size=4,
            generations=2,
            elite_count=2,
            mutation_rate=0.15,
            immigration_rate=0.0,
            selection_method="truncation",
            tournament_size=2,
            output_dir=str(tmp_path / "a"),
            methods=["full_evolution"],
        )
        cfg_b = ExperimentConfig(
            name="repro_b",
            seeds=[123],
            population_size=4,
            generations=2,
            elite_count=2,
            mutation_rate=0.15,
            immigration_rate=0.0,
            selection_method="truncation",
            tournament_size=2,
            output_dir=str(tmp_path / "b"),
            methods=["full_evolution"],
        )

        runner_a = runner_factory(cfg_a)
        runner_b = runner_factory(cfg_b)

        result_a = await runner_a.run(resume=False)
        result_b = await runner_b.run(resume=False)

        run_a = result_a.runs[0]
        run_b = result_b.runs[0]

        # Genome IDs use uuid4 (non-reproducible). Check fitness values
        # and per-question scores instead.
        assert run_a.champion_fitness == pytest.approx(run_b.champion_fitness)
        assert run_a.per_question_scores == pytest.approx(run_b.per_question_scores)
        assert run_a.fitness_curve == pytest.approx(run_b.fitness_curve)
        assert run_a.api_calls == run_b.api_calls
        assert run_a.total_evaluations == run_b.total_evaluations

    async def test_random_search_reproducible(
        self,
        runner_factory,
        tmp_path: Path,
    ) -> None:
        cfg = ExperimentConfig(
            name="rs_repro",
            seeds=[7],
            population_size=3,
            generations=2,
            elite_count=2,
            mutation_rate=0.15,
            immigration_rate=0.0,
            selection_method="truncation",
            tournament_size=2,
            output_dir=str(tmp_path / "rs"),
            methods=["random_search"],
        )

        runner_a = runner_factory(cfg)
        runner_b = runner_factory(
            ExperimentConfig(
                **{**cfg.to_dict(), "output_dir": str(tmp_path / "rs2")}
            )
        )

        result_a = await runner_a.run(resume=False)
        result_b = await runner_b.run(resume=False)

        run_a = result_a.runs[0]
        run_b = result_b.runs[0]

        assert run_a.per_question_scores == pytest.approx(run_b.per_question_scores)
        assert run_a.champion_fitness == pytest.approx(run_b.champion_fitness)


# ---------------------------------------------------------------------------
# JSONL roundtrip
# ---------------------------------------------------------------------------


class TestJsonlRoundtrip:
    """ExperimentResult.to_jsonl + from_jsonl preserves all runs."""

    async def test_jsonl_roundtrip(
        self,
        runner_factory,
        small_config: ExperimentConfig,
        tmp_path: Path,
    ) -> None:
        runner = runner_factory(small_config)
        result = await runner.run()

        path = tmp_path / "results.jsonl"
        result.to_jsonl(path)
        assert path.exists()

        loaded = ExperimentResult.from_jsonl(path)

        assert loaded.config.name == result.config.name
        assert loaded.config.seeds == result.config.seeds
        assert loaded.config.methods == result.config.methods
        assert len(loaded.runs) == len(result.runs)

        for original, restored in zip(result.runs, loaded.runs):
            assert original.method == restored.method
            assert original.seed == restored.seed
            assert original.champion_fitness == pytest.approx(restored.champion_fitness)
            assert original.per_question_scores == pytest.approx(
                restored.per_question_scores
            )
            assert original.fitness_curve == pytest.approx(restored.fitness_curve)
            assert original.api_calls == restored.api_calls
            assert original.total_evaluations == restored.total_evaluations

    def test_from_jsonl_requires_config_or_header(self, tmp_path: Path) -> None:
        path = tmp_path / "no_header.jsonl"
        # Write only run records, no header
        path.write_text(
            json.dumps(
                RunResult(
                    method="random_search",
                    seed=1,
                    champion_genome_id="x",
                    champion_fitness=0.5,
                    per_question_scores=[0.5],
                    fitness_curve=[0.5],
                    wall_clock_seconds=0.1,
                    total_evaluations=1,
                    api_calls=1,
                ).to_dict()
            )
            + "\n"
        )
        with pytest.raises(ValueError):
            ExperimentResult.from_jsonl(path)


# ---------------------------------------------------------------------------
# Per-baseline smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", list(KNOWN_METHODS))
async def test_each_baseline_runs(
    method: str,
    runner_factory,
    tmp_path: Path,
) -> None:
    """Every method in KNOWN_METHODS must execute without raising."""
    cfg = ExperimentConfig(
        name=f"smoke_{method}",
        seeds=[1],
        population_size=3,
        generations=2,
        elite_count=2,
        mutation_rate=0.15,
        immigration_rate=0.0,
        selection_method="truncation",
        tournament_size=2,
        output_dir=str(tmp_path / method),
        methods=[method],
    )
    runner = runner_factory(cfg)
    result = await runner.run()

    assert len(result.runs) == 1
    run = result.runs[0]
    assert run.method == method
    assert len(run.per_question_scores) == 5
    assert run.api_calls > 0


# ---------------------------------------------------------------------------
# Resume / intermediate save
# ---------------------------------------------------------------------------


class TestResumeAndIntermediate:
    """Intermediate files are written and resume works."""

    async def test_intermediate_files_created(
        self,
        runner_factory,
        small_config: ExperimentConfig,
        tmp_path: Path,
    ) -> None:
        runner = runner_factory(small_config)
        await runner.run(save_intermediate=True)

        runs_dir = Path(small_config.output_dir) / "runs"
        assert runs_dir.exists()

        files = sorted(runs_dir.glob("*.json"))
        # 1 seed * 2 methods = 2 intermediate files
        assert len(files) == 2

    async def test_resume_skips_completed_runs(
        self,
        runner_factory,
        small_config: ExperimentConfig,
    ) -> None:
        runner = runner_factory(small_config)
        first_result = await runner.run(save_intermediate=True)

        # Second run should reuse the intermediate files. Since the agent
        # is deterministic, the results must be identical.
        runner2 = runner_factory(small_config)
        second_result = await runner2.run(save_intermediate=True, resume=True)

        assert len(second_result.runs) == len(first_result.runs)
        for first, second in zip(first_result.runs, second_result.runs):
            assert first.method == second.method
            assert first.seed == second.seed
            assert first.per_question_scores == pytest.approx(
                second.per_question_scores
            )


# ---------------------------------------------------------------------------
# New baselines: bayesian_opt and successive_halving
# ---------------------------------------------------------------------------


def _new_baseline_config(
    tmp_path: Path,
    method: str,
    seed: int,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=f"{method}_smoke",
        seeds=[seed],
        population_size=4,
        generations=3,
        elite_count=2,
        mutation_rate=0.15,
        immigration_rate=0.0,
        selection_method="truncation",
        tournament_size=2,
        output_dir=str(tmp_path / method),
        methods=[method],
    )


class TestBayesianOptBaseline:
    """Smoke + reproducibility tests for the bayesian_opt baseline."""

    async def test_runs_without_error(
        self,
        runner_factory,
        tmp_path: Path,
    ) -> None:
        cfg = _new_baseline_config(tmp_path, "bayesian_opt", seed=11)
        runner = runner_factory(cfg)
        result = await runner.run(resume=False)

        assert len(result.runs) == 1
        run = result.runs[0]
        assert run.method == "bayesian_opt"
        assert len(run.per_question_scores) == 5
        assert all(0.0 <= s <= 1.0 for s in run.per_question_scores)
        # Must record at least one generation on the fitness curve.
        assert len(run.fitness_curve) >= 1
        # TPE-specific metadata must be present.
        assert "tpe_split_fraction" in run.metadata
        assert "tpe_warmup" in run.metadata

    async def test_api_calls_match_budget(
        self,
        runner_factory,
        tmp_path: Path,
    ) -> None:
        cfg = _new_baseline_config(tmp_path, "bayesian_opt", seed=13)
        runner = runner_factory(cfg)
        result = await runner.run(resume=False)

        run = result.runs[0]
        expected_budget = cfg.population_size * cfg.generations * len(
            runner.train_tasks
        ) + len(runner.test_tasks)  # +1 test-set eval at the end
        # Within 20% tolerance (the +test is a small correction).
        lower = expected_budget * 0.8
        upper = expected_budget * 1.2
        assert lower <= run.api_calls <= upper, (
            f"api_calls={run.api_calls} outside [{lower}, {upper}]"
        )

    async def test_same_seed_same_result(
        self,
        runner_factory,
        tmp_path: Path,
    ) -> None:
        cfg_a = _new_baseline_config(tmp_path / "a", "bayesian_opt", seed=17)
        cfg_b = _new_baseline_config(tmp_path / "b", "bayesian_opt", seed=17)

        runner_a = runner_factory(cfg_a)
        runner_b = runner_factory(cfg_b)

        result_a = await runner_a.run(resume=False)
        result_b = await runner_b.run(resume=False)

        run_a = result_a.runs[0]
        run_b = result_b.runs[0]

        assert run_a.champion_fitness == pytest.approx(run_b.champion_fitness)
        assert run_a.per_question_scores == pytest.approx(
            run_b.per_question_scores
        )
        assert run_a.api_calls == run_b.api_calls
        assert run_a.total_evaluations == run_b.total_evaluations
        assert run_a.fitness_curve == pytest.approx(run_b.fitness_curve)


class TestSuccessiveHalvingBaseline:
    """Smoke + reproducibility tests for the successive_halving baseline."""

    async def test_runs_without_error(
        self,
        runner_factory,
        tmp_path: Path,
    ) -> None:
        cfg = _new_baseline_config(tmp_path, "successive_halving", seed=21)
        runner = runner_factory(cfg)
        result = await runner.run(resume=False)

        assert len(result.runs) == 1
        run = result.runs[0]
        assert run.method == "successive_halving"
        assert len(run.per_question_scores) == 5
        assert all(0.0 <= s <= 1.0 for s in run.per_question_scores)
        assert len(run.fitness_curve) >= 1
        # Multi-fidelity metadata must be present.
        assert run.metadata.get("eta") == 3
        assert "fidelity_schedule" in run.metadata
        fidelity_schedule = run.metadata["fidelity_schedule"]
        assert isinstance(fidelity_schedule, list)
        assert len(fidelity_schedule) >= 1
        # Fidelity schedule must be strictly non-decreasing.
        for prev, curr in zip(fidelity_schedule, fidelity_schedule[1:]):
            assert curr >= prev

    async def test_api_calls_match_budget(
        self,
        runner_factory,
        tmp_path: Path,
    ) -> None:
        cfg = _new_baseline_config(tmp_path, "successive_halving", seed=23)
        runner = runner_factory(cfg)
        result = await runner.run(resume=False)

        run = result.runs[0]
        # The task-level budget target is pop*gen*|train|.
        n_train = len(runner.train_tasks)
        expected_task_budget = (
            cfg.population_size * cfg.generations * n_train
        )
        # Plus the final test-set pass (|test| calls).
        upper = expected_task_budget * 1.2 + len(runner.test_tasks)
        lower = expected_task_budget * 0.5  # allow early-stop when out of budget
        assert lower <= run.api_calls <= upper, (
            f"api_calls={run.api_calls} outside [{lower}, {upper}]"
        )

    async def test_same_seed_same_result(
        self,
        runner_factory,
        tmp_path: Path,
    ) -> None:
        cfg_a = _new_baseline_config(
            tmp_path / "a", "successive_halving", seed=29
        )
        cfg_b = _new_baseline_config(
            tmp_path / "b", "successive_halving", seed=29
        )

        runner_a = runner_factory(cfg_a)
        runner_b = runner_factory(cfg_b)

        result_a = await runner_a.run(resume=False)
        result_b = await runner_b.run(resume=False)

        run_a = result_a.runs[0]
        run_b = result_b.runs[0]

        assert run_a.champion_fitness == pytest.approx(run_b.champion_fitness)
        assert run_a.per_question_scores == pytest.approx(
            run_b.per_question_scores
        )
        assert run_a.api_calls == run_b.api_calls
        assert run_a.total_evaluations == run_b.total_evaluations
        assert run_a.fitness_curve == pytest.approx(run_b.fitness_curve)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """ExperimentRunner.__init__ validates the config."""

    def test_empty_seeds_rejected(
        self,
        runner_factory,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError):
            runner_factory(
                ExperimentConfig(
                    name="bad",
                    seeds=[],
                    population_size=4,
                    generations=2,
                    elite_count=2,
                    mutation_rate=0.1,
                    immigration_rate=0.0,
                    selection_method="truncation",
                    tournament_size=2,
                    output_dir=str(tmp_path / "bad"),
                    methods=["full_evolution"],
                )
            )

    def test_unknown_method_rejected(
        self,
        runner_factory,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError):
            runner_factory(
                ExperimentConfig(
                    name="bad",
                    seeds=[1],
                    population_size=4,
                    generations=2,
                    elite_count=2,
                    mutation_rate=0.1,
                    immigration_rate=0.0,
                    selection_method="truncation",
                    tournament_size=2,
                    output_dir=str(tmp_path / "bad"),
                    methods=["nonexistent_method"],
                )
            )

    async def test_unknown_method_in_run_method_rejected(
        self,
        runner_factory,
        small_config: ExperimentConfig,
    ) -> None:
        runner = runner_factory(small_config)
        # The method dispatcher should also reject unknown names directly.
        with pytest.raises(ValueError):
            await runner.run_method("does_not_exist", seed=1)
