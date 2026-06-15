"""Tests for the breeding engine orchestration loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from breed.adapters.base import Adapter, AgentResult
from breed.arenas.base import Arena, EvalResult, Task
from breed.engine import BreedingConfig, BreedingEngine
from breed.events import (
    AgentEvaluated,
    BreedingComplete,
    BreedingStarted,
    ChampionChanged,
    EventBus,
    GenerationComplete,
    ImmigrantsInjected,
)
from breed.genome import Genome


# ---------------------------------------------------------------------------
# Deterministic mock arena
# ---------------------------------------------------------------------------


class _MockArena(Arena):
    """Arena that scores genomes based on their temperature gene.

    Lower temperature => higher fitness (simulates a known landscape
    where selection pressure can be verified).
    """

    async def generate_tasks(
        self, count: int, seed: int | None = None
    ) -> list[Task]:
        return [
            Task(task_id=f"task-{i}", prompt=f"Question {i}", expected=1.0)
            for i in range(count)
        ]

    async def evaluate(
        self, genome: Genome, adapter: Adapter, tasks: list[Task]
    ) -> EvalResult:
        # Use the adapter so the mock gets called (verifiable in tests)
        result = await adapter.run(genome, tasks[0].prompt)

        # Fitness is inversely proportional to temperature gene value.
        # Temperature range is [0.0, 1.5], so normalise to [0, 1].
        temp_gene = genome.genes.get("temperature")
        if temp_gene is not None:
            raw = float(temp_gene.value)
            # Lower temp -> higher fitness: fitness = 1 - (temp / 1.5)
            fitness = max(0.0, min(1.0, 1.0 - raw / 1.5))
        else:
            fitness = 0.5

        return EvalResult(
            genome_id=genome.genome_id,
            fitness_score=fitness,
            fitness_breakdown={"temperature_fitness": fitness},
            task_results=[result],
            cost_tokens=result.cost_tokens,
        )


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------


def _make_mock_adapter() -> Adapter:
    """Create an adapter that returns a fixed output without real API calls."""
    adapter = AsyncMock(spec=Adapter)
    adapter.run.return_value = AgentResult(
        output="0.75",
        cost_tokens=5,
    )
    return adapter


# ---------------------------------------------------------------------------
# Event collector
# ---------------------------------------------------------------------------


class _EventCollector:
    """Observer that records every event for later assertion."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def on_event(self, event: Any) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [e.event_type for e in self.events]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_results(tmp_path: Path) -> Path:
    """Return a temporary directory for engine results."""
    results = tmp_path / "results"
    results.mkdir()
    return results


@pytest.fixture()
def small_config(tmp_results: Path) -> BreedingConfig:
    """A small-scale config for fast test runs."""
    return BreedingConfig(
        population_size=8,
        elite_count=3,
        generations=3,
        crossover_rate=0.7,
        mutation_rate=0.15,
        immigration_rate=0.1,
        selection_method="truncation",
        tournament_size=3,
        num_tasks=3,
        seed=42,
        lineage_file="lineage.jsonl",
        results_dir=str(tmp_results),
    )


@pytest.fixture()
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def collector() -> _EventCollector:
    return _EventCollector()


# ---------------------------------------------------------------------------
# Tests: full run
# ---------------------------------------------------------------------------


class TestEngineFullRun:
    """The engine runs the full loop and returns a champion."""

    async def test_returns_champion_genome(
        self, small_config: BreedingConfig
    ) -> None:
        engine = BreedingEngine(
            config=small_config,
            arena=_MockArena(),
            adapter=_make_mock_adapter(),
        )
        champion = await engine.run()

        assert isinstance(champion, Genome)
        assert champion.genome_id is not None

    async def test_champion_fitness_is_positive(
        self, small_config: BreedingConfig
    ) -> None:
        engine = BreedingEngine(
            config=small_config,
            arena=_MockArena(),
            adapter=_make_mock_adapter(),
        )
        await engine.run()

        assert engine.champion_fitness > 0.0

    async def test_correct_number_of_generations(
        self,
        small_config: BreedingConfig,
        event_bus: EventBus,
        collector: _EventCollector,
    ) -> None:
        await event_bus.add_observer(collector)
        engine = BreedingEngine(
            config=small_config,
            arena=_MockArena(),
            adapter=_make_mock_adapter(),
            event_bus=event_bus,
        )
        await engine.run()

        gen_complete_events = [
            e for e in collector.events if isinstance(e, GenerationComplete)
        ]
        assert len(gen_complete_events) == small_config.generations


# ---------------------------------------------------------------------------
# Tests: fitness improvement
# ---------------------------------------------------------------------------


class TestFitnessImprovement:
    """Champion fitness should not degrade over generations."""

    async def test_champion_does_not_degrade(
        self,
        small_config: BreedingConfig,
        event_bus: EventBus,
        collector: _EventCollector,
    ) -> None:
        await event_bus.add_observer(collector)
        engine = BreedingEngine(
            config=small_config,
            arena=_MockArena(),
            adapter=_make_mock_adapter(),
            event_bus=event_bus,
        )
        await engine.run()

        # ChampionChanged events should have monotonically non-decreasing fitness
        champion_events = [
            e for e in collector.events if isinstance(e, ChampionChanged)
        ]
        fitnesses = [e.fitness_score for e in champion_events]

        for i in range(1, len(fitnesses)):
            assert fitnesses[i] >= fitnesses[i - 1], (
                f"Champion fitness decreased from {fitnesses[i - 1]} to {fitnesses[i]}"
            )


# ---------------------------------------------------------------------------
# Tests: event ordering
# ---------------------------------------------------------------------------


class TestEventOrdering:
    """Events are emitted in the correct lifecycle order."""

    async def test_breeding_started_is_first(
        self,
        small_config: BreedingConfig,
        event_bus: EventBus,
        collector: _EventCollector,
    ) -> None:
        await event_bus.add_observer(collector)
        engine = BreedingEngine(
            config=small_config,
            arena=_MockArena(),
            adapter=_make_mock_adapter(),
            event_bus=event_bus,
        )
        await engine.run()

        assert len(collector.events) > 0
        assert isinstance(collector.events[0], BreedingStarted)

    async def test_breeding_complete_is_last(
        self,
        small_config: BreedingConfig,
        event_bus: EventBus,
        collector: _EventCollector,
    ) -> None:
        await event_bus.add_observer(collector)
        engine = BreedingEngine(
            config=small_config,
            arena=_MockArena(),
            adapter=_make_mock_adapter(),
            event_bus=event_bus,
        )
        await engine.run()

        assert isinstance(collector.events[-1], BreedingComplete)

    async def test_agent_evaluated_events_emitted(
        self,
        small_config: BreedingConfig,
        event_bus: EventBus,
        collector: _EventCollector,
    ) -> None:
        await event_bus.add_observer(collector)
        engine = BreedingEngine(
            config=small_config,
            arena=_MockArena(),
            adapter=_make_mock_adapter(),
            event_bus=event_bus,
        )
        await engine.run()

        eval_events = [
            e for e in collector.events if isinstance(e, AgentEvaluated)
        ]
        # First generation evaluates population_size genomes. Subsequent
        # generations evaluate population_size + immigrants from the
        # previous generation. So total >= population_size * generations.
        assert len(eval_events) >= small_config.population_size

    async def test_generation_complete_follows_evaluations(
        self,
        small_config: BreedingConfig,
        event_bus: EventBus,
        collector: _EventCollector,
    ) -> None:
        """Each GenerationComplete should appear after all AgentEvaluated for that gen."""
        await event_bus.add_observer(collector)
        engine = BreedingEngine(
            config=small_config,
            arena=_MockArena(),
            adapter=_make_mock_adapter(),
            event_bus=event_bus,
        )
        await engine.run()

        types = collector.types()
        gen_complete_indices = [
            i for i, t in enumerate(types) if t == "generation_complete"
        ]
        # The first generation_complete must come after at least pop_size evaluations
        if gen_complete_indices:
            first_gc = gen_complete_indices[0]
            eval_before = sum(
                1 for t in types[:first_gc] if t == "agent_evaluated"
            )
            assert eval_before >= small_config.population_size


# ---------------------------------------------------------------------------
# Tests: lineage file
# ---------------------------------------------------------------------------


class TestLineageTracking:
    """Lineage file is created and contains records."""

    async def test_lineage_file_created(
        self, small_config: BreedingConfig
    ) -> None:
        engine = BreedingEngine(
            config=small_config,
            arena=_MockArena(),
            adapter=_make_mock_adapter(),
        )
        await engine.run()

        lineage_path = Path(small_config.results_dir) / small_config.lineage_file
        assert lineage_path.exists()

    async def test_lineage_contains_records(
        self, small_config: BreedingConfig
    ) -> None:
        engine = BreedingEngine(
            config=small_config,
            arena=_MockArena(),
            adapter=_make_mock_adapter(),
        )
        await engine.run()

        records = engine.lineage_tracker.load()
        assert len(records) > 0

    async def test_lineage_record_count_matches_evaluations(
        self, small_config: BreedingConfig
    ) -> None:
        engine = BreedingEngine(
            config=small_config,
            arena=_MockArena(),
            adapter=_make_mock_adapter(),
        )
        await engine.run()

        records = engine.lineage_tracker.load()
        # Each generation evaluates every member; population grows with immigrants
        # The total records should equal total_evaluations
        status = engine.get_status()
        assert len(records) == status["total_evaluations"]


# ---------------------------------------------------------------------------
# Tests: reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Engine respects seed for deterministic results."""

    async def test_same_seed_produces_same_champion(
        self, tmp_path: Path
    ) -> None:
        results_a = tmp_path / "run_a"
        results_b = tmp_path / "run_b"

        config_a = BreedingConfig(
            population_size=8,
            elite_count=3,
            generations=3,
            seed=999,
            num_tasks=3,
            selection_method="truncation",
            results_dir=str(results_a),
        )
        config_b = BreedingConfig(
            population_size=8,
            elite_count=3,
            generations=3,
            seed=999,
            num_tasks=3,
            selection_method="truncation",
            results_dir=str(results_b),
        )

        engine_a = BreedingEngine(
            config=config_a, arena=_MockArena(), adapter=_make_mock_adapter()
        )
        engine_b = BreedingEngine(
            config=config_b, arena=_MockArena(), adapter=_make_mock_adapter()
        )

        champ_a = await engine_a.run()
        champ_b = await engine_b.run()

        # Genome IDs contain uuid4 randomness, but the gene values and
        # fitness produced by two identically-seeded runs must match.
        assert engine_a.champion_fitness == engine_b.champion_fitness

        # Same gene values for the champion
        for gene_name in champ_a.genes:
            assert gene_name in champ_b.genes
            assert champ_a.genes[gene_name].value == champ_b.genes[gene_name].value


# ---------------------------------------------------------------------------
# Tests: get_status
# ---------------------------------------------------------------------------


class TestGetStatus:
    """get_status returns expected fields."""

    async def test_status_before_run(
        self, small_config: BreedingConfig
    ) -> None:
        engine = BreedingEngine(
            config=small_config,
            arena=_MockArena(),
            adapter=_make_mock_adapter(),
        )
        status = engine.get_status()

        expected_keys = {
            "current_generation",
            "total_generations",
            "champion_id",
            "champion_fitness",
            "population_size",
            "running",
            "paused",
            "total_evaluations",
        }
        assert set(status.keys()) == expected_keys
        assert status["running"] is False
        assert status["champion_id"] is None
        assert status["population_size"] == 0

    async def test_status_after_run(
        self, small_config: BreedingConfig
    ) -> None:
        engine = BreedingEngine(
            config=small_config,
            arena=_MockArena(),
            adapter=_make_mock_adapter(),
        )
        await engine.run()
        status = engine.get_status()

        assert status["running"] is False
        assert status["champion_id"] is not None
        assert status["champion_fitness"] > 0.0
        assert status["total_evaluations"] > 0
        assert status["total_generations"] == small_config.generations


# ---------------------------------------------------------------------------
# Tests: immigrants
# ---------------------------------------------------------------------------


class TestImmigration:
    """Immigration events are emitted when immigration_rate > 0."""

    async def test_immigrants_injected_events(
        self,
        small_config: BreedingConfig,
        event_bus: EventBus,
        collector: _EventCollector,
    ) -> None:
        await event_bus.add_observer(collector)
        engine = BreedingEngine(
            config=small_config,
            arena=_MockArena(),
            adapter=_make_mock_adapter(),
            event_bus=event_bus,
        )
        await engine.run()

        immigrant_events = [
            e for e in collector.events if isinstance(e, ImmigrantsInjected)
        ]
        # One per generation
        assert len(immigrant_events) == small_config.generations

    async def test_no_immigrants_when_rate_zero(
        self, tmp_path: Path
    ) -> None:
        results = tmp_path / "no_imm"
        config = BreedingConfig(
            population_size=8,
            elite_count=3,
            generations=2,
            immigration_rate=0.0,
            seed=42,
            num_tasks=3,
            selection_method="truncation",
            results_dir=str(results),
        )
        bus = EventBus()
        coll = _EventCollector()
        await bus.add_observer(coll)

        engine = BreedingEngine(
            config=config,
            arena=_MockArena(),
            adapter=_make_mock_adapter(),
            event_bus=bus,
        )
        await engine.run()

        immigrant_events = [
            e for e in coll.events if isinstance(e, ImmigrantsInjected)
        ]
        assert len(immigrant_events) == 0


# ---------------------------------------------------------------------------
# Tests: pause / resume
# ---------------------------------------------------------------------------


class TestPauseResume:
    """Engine can be paused and resumed."""

    async def test_pause_flag(self, small_config: BreedingConfig) -> None:
        engine = BreedingEngine(
            config=small_config,
            arena=_MockArena(),
            adapter=_make_mock_adapter(),
        )
        assert engine.get_status()["paused"] is False

        engine.pause()
        assert engine.get_status()["paused"] is True

        engine.resume()
        assert engine.get_status()["paused"] is False


# ---------------------------------------------------------------------------
# Tests: run_generation (step-by-step)
# ---------------------------------------------------------------------------


class TestRunGeneration:
    """run_generation allows step-by-step manual control."""

    async def test_single_generation_step(
        self,
        small_config: BreedingConfig,
        event_bus: EventBus,
        collector: _EventCollector,
    ) -> None:
        await event_bus.add_observer(collector)
        engine = BreedingEngine(
            config=small_config,
            arena=_MockArena(),
            adapter=_make_mock_adapter(),
            event_bus=event_bus,
        )

        await engine.run_generation(0)

        gen_events = [
            e for e in collector.events if isinstance(e, GenerationComplete)
        ]
        assert len(gen_events) == 1
        assert gen_events[0].generation == 0

    async def test_population_auto_spawned(
        self, small_config: BreedingConfig
    ) -> None:
        engine = BreedingEngine(
            config=small_config,
            arena=_MockArena(),
            adapter=_make_mock_adapter(),
        )
        assert engine.population is None

        await engine.run_generation(0)

        assert engine.population is not None
        assert engine.population.size > 0


# ---------------------------------------------------------------------------
# Tests: config defaults
# ---------------------------------------------------------------------------


class TestBreedingConfig:
    """BreedingConfig has sensible defaults."""

    def test_defaults(self) -> None:
        config = BreedingConfig()
        assert config.population_size == 20
        assert config.elite_count == 4
        assert config.generations == 25
        assert config.crossover_rate == 0.7
        assert config.mutation_rate == 0.15
        assert config.immigration_rate == 0.1
        assert config.selection_method == "tournament"
        assert config.tournament_size == 3
        assert config.num_tasks == 50
        assert config.seed is None
        assert config.lineage_file == "lineage.jsonl"
        assert config.results_dir == "results"

    def test_frozen(self) -> None:
        config = BreedingConfig()
        with pytest.raises(AttributeError):
            config.population_size = 100  # type: ignore[misc]
