"""Tests for breed.events -- event types, observers, and EventBus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from breed.events import (
    AgentBorn,
    AgentDied,
    AgentEvaluated,
    BreedingComplete,
    BreedingStarted,
    CallbackObserver,
    ChampionChanged,
    CompositeObserver,
    CrossoverEvent,
    EventBus,
    GenerationComplete,
    ImmigrantsInjected,
    JSONObserver,
    MutationEvent,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_agent_born(**overrides: Any) -> AgentBorn:
    defaults: dict[str, Any] = {
        "genome_id": "abc123",
        "generation": 0,
        "parents": [],
    }
    return AgentBorn(**{**defaults, **overrides})


# ── Event creation & serialisation ──────────────────────────────────────────


class TestEventCreation:
    def test_agent_born_fields(self) -> None:
        event = AgentBorn(genome_id="g1", generation=1, parents=["p1", "p2"])
        assert event.genome_id == "g1"
        assert event.generation == 1
        assert event.parents == ["p1", "p2"]
        assert event.timestamp  # non-empty

    def test_agent_born_is_frozen(self) -> None:
        event = _make_agent_born()
        with pytest.raises(AttributeError):
            event.genome_id = "new"  # type: ignore[misc]

    def test_event_type_snake_case(self) -> None:
        assert AgentBorn(genome_id="g", generation=0, parents=[]).event_type == "agent_born"
        assert AgentDied(genome_id="g", generation=0, fitness_score=0.5).event_type == "agent_died"
        assert CrossoverEvent(
            parent_a_id="a", parent_b_id="b", child_id="c", operator="uniform"
        ).event_type == "crossover_event"
        assert MutationEvent(
            genome_id="g", operator="gaussian", gene_name="temperature"
        ).event_type == "mutation_event"
        assert GenerationComplete(
            generation=1, best_fitness=0.9, avg_fitness=0.5,
            worst_fitness=0.1, population_size=20, champion_id="ch",
        ).event_type == "generation_complete"
        assert ChampionChanged(
            old_champion_id=None, new_champion_id="ch", generation=1, fitness_score=0.95
        ).event_type == "champion_changed"
        assert BreedingStarted(
            total_generations=100, population_size=50, arena_type="forecasting"
        ).event_type == "breeding_started"
        assert BreedingComplete(
            final_generation=100, champion_id="ch",
            champion_fitness=0.95, total_evaluations=5000,
        ).event_type == "breeding_complete"
        assert ImmigrantsInjected(count=5, generation=10).event_type == "immigrants_injected"

    def test_agent_evaluated_event_type(self) -> None:
        event = AgentEvaluated(
            genome_id="g1", generation=2, fitness_score=0.85,
            fitness_breakdown={"accuracy": 0.9, "calibration": 0.8},
        )
        assert event.event_type == "agent_evaluated"

    def test_to_dict_roundtrip(self) -> None:
        event = AgentBorn(genome_id="g1", generation=3, parents=["p1"])
        d = event.to_dict()
        assert isinstance(d, dict)
        assert d["genome_id"] == "g1"
        assert d["generation"] == 3
        assert d["parents"] == ["p1"]
        assert d["event_type"] == "agent_born"
        assert "timestamp" in d

    def test_to_dict_json_serialisable(self) -> None:
        event = AgentEvaluated(
            genome_id="g1", generation=1, fitness_score=0.75,
            fitness_breakdown={"brier": 0.8, "pass_rate": 0.7},
        )
        raw = json.dumps(event.to_dict())
        parsed = json.loads(raw)
        assert parsed["fitness_breakdown"] == {"brier": 0.8, "pass_rate": 0.7}

    def test_all_event_types_have_timestamp(self) -> None:
        events = [
            AgentBorn(genome_id="g", generation=0, parents=[]),
            AgentDied(genome_id="g", generation=0, fitness_score=0.1),
            AgentEvaluated(genome_id="g", generation=0, fitness_score=0.5, fitness_breakdown={}),
            CrossoverEvent(parent_a_id="a", parent_b_id="b", child_id="c", operator="uniform"),
            MutationEvent(genome_id="g", operator="gaussian", gene_name=None),
            GenerationComplete(
                generation=0, best_fitness=0.9, avg_fitness=0.5,
                worst_fitness=0.1, population_size=10, champion_id="c",
            ),
            ChampionChanged(
                old_champion_id=None, new_champion_id="c", generation=0, fitness_score=0.9,
            ),
            BreedingStarted(total_generations=10, population_size=20, arena_type="test"),
            BreedingComplete(
                final_generation=10, champion_id="c",
                champion_fitness=0.95, total_evaluations=200,
            ),
            ImmigrantsInjected(count=3, generation=5),
        ]
        for event in events:
            d = event.to_dict()
            assert "timestamp" in d, f"{type(event).__name__} missing timestamp"
            assert len(d["timestamp"]) > 0


# ── JSONObserver ────────────────────────────────────────────────────────────


class TestJSONObserver:
    @pytest.mark.asyncio
    async def test_writes_json_lines(self, tmp_path: Path) -> None:
        filepath = tmp_path / "events.jsonl"
        observer = JSONObserver(filepath)

        born = AgentBorn(genome_id="g1", generation=0, parents=[])
        died = AgentDied(genome_id="g1", generation=5, fitness_score=0.3)

        await observer.on_event(born)
        await observer.on_event(died)

        lines = filepath.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["event_type"] == "agent_born"
        assert first["genome_id"] == "g1"

        second = json.loads(lines[1])
        assert second["event_type"] == "agent_died"
        assert second["fitness_score"] == 0.3

    @pytest.mark.asyncio
    async def test_appends_to_existing_file(self, tmp_path: Path) -> None:
        filepath = tmp_path / "events.jsonl"
        filepath.write_text('{"existing": true}\n', encoding="utf-8")

        observer = JSONObserver(filepath)
        await observer.on_event(_make_agent_born())

        lines = filepath.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"existing": True}


# ── CallbackObserver ────────────────────────────────────────────────────────


class TestCallbackObserver:
    @pytest.mark.asyncio
    async def test_dispatches_to_correct_handler(self) -> None:
        received: list[str] = []

        def on_born(event: AgentBorn) -> None:
            received.append(f"born:{event.genome_id}")

        def on_died(event: AgentDied) -> None:
            received.append(f"died:{event.genome_id}")

        observer = CallbackObserver({
            "agent_born": [on_born],
            "agent_died": [on_died],
        })

        await observer.on_event(AgentBorn(genome_id="g1", generation=0, parents=[]))
        await observer.on_event(AgentDied(genome_id="g2", generation=1, fitness_score=0.5))

        assert received == ["born:g1", "died:g2"]

    @pytest.mark.asyncio
    async def test_wildcard_receives_all_events(self) -> None:
        received: list[str] = []

        def on_any(event: Any) -> None:
            received.append(event.event_type)

        observer = CallbackObserver({"*": [on_any]})

        await observer.on_event(AgentBorn(genome_id="g1", generation=0, parents=[]))
        await observer.on_event(AgentDied(genome_id="g1", generation=1, fitness_score=0.5))
        await observer.on_event(ImmigrantsInjected(count=2, generation=3))

        assert received == ["agent_born", "agent_died", "immigrants_injected"]

    @pytest.mark.asyncio
    async def test_async_callback(self) -> None:
        received: list[str] = []

        async def on_born(event: AgentBorn) -> None:
            received.append(event.genome_id)

        observer = CallbackObserver({"agent_born": [on_born]})
        await observer.on_event(AgentBorn(genome_id="async_g", generation=0, parents=[]))

        assert received == ["async_g"]

    @pytest.mark.asyncio
    async def test_no_matching_handler_is_silent(self) -> None:
        observer = CallbackObserver({"agent_born": [lambda e: None]})
        # Should not raise -- no handler for agent_died
        await observer.on_event(AgentDied(genome_id="g1", generation=0, fitness_score=0.1))

    @pytest.mark.asyncio
    async def test_register_adds_handler(self) -> None:
        received: list[str] = []
        observer = CallbackObserver()
        observer.register("agent_born", lambda e: received.append(e.genome_id))

        await observer.on_event(AgentBorn(genome_id="reg1", generation=0, parents=[]))
        assert received == ["reg1"]


# ── CompositeObserver ───────────────────────────────────────────────────────


class TestCompositeObserver:
    @pytest.mark.asyncio
    async def test_fans_out_to_multiple_observers(self, tmp_path: Path) -> None:
        file_a = tmp_path / "a.jsonl"
        file_b = tmp_path / "b.jsonl"

        observer = CompositeObserver([
            JSONObserver(file_a),
            JSONObserver(file_b),
        ])

        event = AgentBorn(genome_id="g1", generation=0, parents=[])
        await observer.on_event(event)

        lines_a = file_a.read_text(encoding="utf-8").strip().splitlines()
        lines_b = file_b.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines_a) == 1
        assert len(lines_b) == 1

    @pytest.mark.asyncio
    async def test_add_observer(self, tmp_path: Path) -> None:
        filepath = tmp_path / "late.jsonl"
        composite = CompositeObserver()
        composite.add(JSONObserver(filepath))

        await composite.on_event(_make_agent_born())
        assert filepath.exists()

    @pytest.mark.asyncio
    async def test_empty_composite_is_noop(self) -> None:
        composite = CompositeObserver()
        # Should not raise
        await composite.on_event(_make_agent_born())


# ── EventBus ────────────────────────────────────────────────────────────────


class TestEventBus:
    @pytest.mark.asyncio
    async def test_emit_sends_to_all_observers(self, tmp_path: Path) -> None:
        bus = EventBus()
        file_a = tmp_path / "a.jsonl"
        file_b = tmp_path / "b.jsonl"

        await bus.add_observer(JSONObserver(file_a))
        await bus.add_observer(JSONObserver(file_b))

        await bus.emit(AgentBorn(genome_id="g1", generation=0, parents=[]))

        assert len(file_a.read_text(encoding="utf-8").strip().splitlines()) == 1
        assert len(file_b.read_text(encoding="utf-8").strip().splitlines()) == 1

    @pytest.mark.asyncio
    async def test_remove_observer(self, tmp_path: Path) -> None:
        bus = EventBus()
        filepath = tmp_path / "removable.jsonl"
        observer = JSONObserver(filepath)

        await bus.add_observer(observer)
        await bus.emit(_make_agent_born())
        assert len(filepath.read_text(encoding="utf-8").strip().splitlines()) == 1

        await bus.remove_observer(observer)
        await bus.emit(_make_agent_born())
        # Still 1 line because observer was removed before second emit
        assert len(filepath.read_text(encoding="utf-8").strip().splitlines()) == 1

    @pytest.mark.asyncio
    async def test_remove_nonexistent_observer_is_silent(self) -> None:
        bus = EventBus()
        observer = JSONObserver(Path("/tmp/nonexistent.jsonl"))
        # Should not raise
        await bus.remove_observer(observer)

    @pytest.mark.asyncio
    async def test_emit_with_no_observers(self) -> None:
        bus = EventBus()
        # Should not raise
        await bus.emit(_make_agent_born())

    @pytest.mark.asyncio
    async def test_callback_observer_via_bus(self) -> None:
        received: list[str] = []
        cb_observer = CallbackObserver({"*": [lambda e: received.append(e.event_type)]})

        bus = EventBus()
        await bus.add_observer(cb_observer)

        await bus.emit(AgentBorn(genome_id="g1", generation=0, parents=[]))
        await bus.emit(BreedingComplete(
            final_generation=10, champion_id="c",
            champion_fitness=0.9, total_evaluations=100,
        ))

        assert received == ["agent_born", "breeding_complete"]
