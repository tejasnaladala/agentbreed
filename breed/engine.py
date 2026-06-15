"""Breeding engine -- core orchestration loop for agent evolution.

Ties together population management, evaluation arenas, adapters, event
emission, and lineage tracking into a single configurable async loop.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from breed.adapters.base import Adapter
from breed.arenas.base import Arena, EvalResult
from breed.events import (
    AgentDied,
    AgentEvaluated,
    BreedingComplete,
    BreedingStarted,
    ChampionChanged,
    EventBus,
    GenerationComplete,
    ImmigrantsInjected,
)
from breed.genome import Genome
from breed.lineage import LineageTracker
from breed.population import Population


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BreedingConfig:
    """Immutable configuration for a breeding run.

    Attributes:
        population_size: Number of genomes per generation.
        elite_count: Survivors kept each generation (must be < population_size).
        generations: Total number of generations to evolve.
        crossover_rate: Probability of crossover (currently used as a signal;
            breeding always produces offspring to fill population).
        mutation_rate: Per-genome probability of mutation.
        immigration_rate: Fraction of population replaced by random immigrants.
        selection_method: ``"tournament"`` or ``"truncation"``.
        tournament_size: K for tournament selection.
        num_tasks: Number of evaluation tasks per generation.
        seed: Optional RNG seed for full reproducibility.
        lineage_file: Path to the JSONL lineage log.
        results_dir: Directory for persisted artefacts.
    """

    population_size: int = 20
    elite_count: int = 4
    generations: int = 25
    crossover_rate: float = 0.7
    mutation_rate: float = 0.15
    immigration_rate: float = 0.1
    selection_method: str = "tournament"
    tournament_size: int = 3
    num_tasks: int = 50
    seed: int | None = None
    lineage_file: str = "lineage.jsonl"
    results_dir: str = "results"


# ---------------------------------------------------------------------------
# Engine state
# ---------------------------------------------------------------------------


@dataclass
class _EngineState:
    """Mutable internal state for the breeding engine."""

    current_generation: int = 0
    champion_genome: Genome | None = None
    champion_fitness: float = -1.0
    population: Population | None = None
    running: bool = False
    paused: bool = False
    total_evaluations: int = 0


# ---------------------------------------------------------------------------
# Breeding engine
# ---------------------------------------------------------------------------


class BreedingEngine:
    """Core orchestration loop for evolutionary agent breeding.

    Parameters
    ----------
    config:
        Breeding hyper-parameters.
    arena:
        Evaluation domain that generates tasks and scores genomes.
    adapter:
        Translates genomes into runnable agents.
    event_bus:
        Optional event bus for broadcasting lifecycle events.
    """

    def __init__(
        self,
        config: BreedingConfig,
        arena: Arena,
        adapter: Adapter,
        event_bus: EventBus | None = None,
        gene_template: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._config = config
        self._arena = arena
        self._adapter = adapter
        self._event_bus = event_bus or EventBus()
        self._gene_template = gene_template
        self._state = _EngineState()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # not paused initially

        results_dir = Path(config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        lineage_path = results_dir / config.lineage_file
        self._lineage = LineageTracker(lineage_path)

        self._rng = random.Random(config.seed)

    # -- public properties ---------------------------------------------------

    @property
    def config(self) -> BreedingConfig:
        return self._config

    @property
    def population(self) -> Population | None:
        return self._state.population

    @property
    def current_generation(self) -> int:
        return self._state.current_generation

    @property
    def champion_genome(self) -> Genome | None:
        return self._state.champion_genome

    @property
    def champion_fitness(self) -> float:
        return self._state.champion_fitness

    @property
    def running(self) -> bool:
        return self._state.running

    @property
    def lineage_tracker(self) -> LineageTracker:
        return self._lineage

    # -- control -------------------------------------------------------------

    def pause(self) -> None:
        """Pause the breeding loop after the current generation completes."""
        self._state.paused = True
        self._pause_event.clear()

    def resume(self) -> None:
        """Resume a paused breeding loop."""
        self._state.paused = False
        self._pause_event.set()

    # -- status --------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return a snapshot of the current engine state."""
        return {
            "current_generation": self._state.current_generation,
            "total_generations": self._config.generations,
            "champion_id": (
                self._state.champion_genome.genome_id
                if self._state.champion_genome is not None
                else None
            ),
            "champion_fitness": self._state.champion_fitness,
            "population_size": (
                self._state.population.size
                if self._state.population is not None
                else 0
            ),
            "running": self._state.running,
            "paused": self._state.paused,
            "total_evaluations": self._state.total_evaluations,
        }

    # -- main loop -----------------------------------------------------------

    async def run(self) -> Genome:
        """Execute the full breeding loop and return the champion genome.

        Raises
        ------
        RuntimeError
            If no genomes remain after evolution (should never happen with
            valid configuration).
        """
        self._state.running = True

        await self._emit(
            BreedingStarted(
                total_generations=self._config.generations,
                population_size=self._config.population_size,
                arena_type=type(self._arena).__name__,
            )
        )

        # Spawn initial population
        spawn_seed = self._next_seed()
        self._state.population = Population.spawn(
            self._config.population_size,
            seed=spawn_seed,
            gene_template=self._gene_template,
        )

        # Generate tasks once (reused across generations for consistency)
        tasks = await self._arena.generate_tasks(
            self._config.num_tasks, seed=self._next_seed()
        )

        for gen in range(self._config.generations):
            # Honour pause requests
            await self._pause_event.wait()

            if not self._state.running:
                break

            self._state.current_generation = gen
            await self._run_generation_inner(gen, tasks)

        # Final bookkeeping
        self._state.running = False

        champion = self._state.champion_genome
        if champion is None:
            raise RuntimeError("No champion genome produced -- population was empty")

        await self._emit(
            BreedingComplete(
                final_generation=self._state.current_generation,
                champion_id=champion.genome_id,
                champion_fitness=self._state.champion_fitness,
                total_evaluations=self._state.total_evaluations,
            )
        )

        return champion

    async def run_generation(self, generation: int) -> None:
        """Execute a single generation step for manual / interactive control.

        Callers are responsible for spawning the population and generating
        tasks before invoking this method.  If the population has not been
        initialised, it will be spawned automatically.
        """
        if self._state.population is None:
            spawn_seed = self._next_seed()
            self._state.population = Population.spawn(
                self._config.population_size,
                seed=spawn_seed,
                gene_template=self._gene_template,
            )

        tasks = await self._arena.generate_tasks(
            self._config.num_tasks, seed=self._next_seed()
        )

        self._state.current_generation = generation
        await self._run_generation_inner(generation, tasks)

    # -- internal generation logic -------------------------------------------

    async def _run_generation_inner(self, gen: int, tasks: list) -> None:
        """Run evaluation, selection, breeding, mutation, immigration."""
        population = self._state.population
        assert population is not None

        # (a-c) Evaluate every genome
        eval_results: list[EvalResult] = []
        for genome in population.members:
            result = await self._arena.evaluate(genome, self._adapter, tasks)
            eval_results.append(result)
            self._state.total_evaluations += 1

            await self._emit(
                AgentEvaluated(
                    genome_id=genome.genome_id,
                    generation=gen,
                    fitness_score=result.fitness_score,
                    fitness_breakdown=result.fitness_breakdown,
                )
            )

        # (d) Extract fitness scores (parallel to population.members order)
        fitness_scores = [r.fitness_score for r in eval_results]

        # (e) Track champion changes
        gen_best_idx = _argmax(fitness_scores)
        gen_best_genome = population.members[gen_best_idx]
        gen_best_fitness = fitness_scores[gen_best_idx]

        if gen_best_fitness > self._state.champion_fitness:
            old_id = (
                self._state.champion_genome.genome_id
                if self._state.champion_genome is not None
                else None
            )
            self._state.champion_genome = gen_best_genome
            self._state.champion_fitness = gen_best_fitness

            await self._emit(
                ChampionChanged(
                    old_champion_id=old_id,
                    new_champion_id=gen_best_genome.genome_id,
                    generation=gen,
                    fitness_score=gen_best_fitness,
                )
            )

        # (f) Log all genomes to lineage tracker
        survivor_ids = self._compute_survivor_ids(population.members, fitness_scores)
        for genome, result in zip(population.members, eval_results):
            self._lineage.log(
                genome=genome,
                fitness_score=result.fitness_score,
                fitness_breakdown=result.fitness_breakdown,
                survived=genome.genome_id in survivor_ids,
            )

        # (g) Select survivors -- emit AgentDied for culled genomes
        pre_select_ids = {g.genome_id for g in population.members}
        population.select(
            fitness_scores=fitness_scores,
            top_k=self._config.elite_count,
            method=self._config.selection_method,  # type: ignore[arg-type]
            tournament_size=self._config.tournament_size,
            seed=self._next_seed(),
        )
        post_select_ids = {g.genome_id for g in population.members}
        culled_ids = pre_select_ids - post_select_ids

        # Build a lookup of genome_id -> fitness from eval results
        id_to_fitness: dict[str, float] = {
            er.genome_id: er.fitness_score for er in eval_results
        }

        for gid in culled_ids:
            await self._emit(
                AgentDied(
                    genome_id=gid,
                    generation=gen,
                    fitness_score=id_to_fitness.get(gid, 0.0),
                )
            )

        # (h) Breed offspring
        elite_fitness = [
            id_to_fitness.get(g.genome_id, 0.0) for g in population.members
        ]
        population.breed(
            target_size=self._config.population_size,
            seed=self._next_seed(),
            fitness_scores=elite_fitness,
        )

        # (i) Mutate
        population.mutate(
            rate=self._config.mutation_rate,
            seed=self._next_seed(),
        )

        # (j) Immigration
        immigrant_count = max(
            1,
            int(self._config.population_size * self._config.immigration_rate),
        ) if self._config.immigration_rate > 0 else 0

        if immigrant_count > 0:
            population.add_immigrants(immigrant_count, seed=self._next_seed())

            await self._emit(
                ImmigrantsInjected(count=immigrant_count, generation=gen)
            )

        # (k) Emit GenerationComplete
        avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0
        await self._emit(
            GenerationComplete(
                generation=gen,
                best_fitness=gen_best_fitness,
                avg_fitness=avg_fitness,
                worst_fitness=min(fitness_scores) if fitness_scores else 0.0,
                population_size=population.size,
                champion_id=gen_best_genome.genome_id,
            )
        )

    # -- helpers -------------------------------------------------------------

    def _compute_survivor_ids(
        self, members: list[Genome], scores: list[float]
    ) -> set[str]:
        """Return genome IDs of the top *elite_count* members by fitness."""
        indexed = sorted(
            enumerate(scores), key=lambda t: t[1], reverse=True
        )
        top_indices = [i for i, _ in indexed[: self._config.elite_count]]
        return {members[i].genome_id for i in top_indices}

    def _next_seed(self) -> int:
        """Draw a deterministic child seed from the engine's RNG."""
        return self._rng.randint(0, 2**63)

    async def _emit(self, event: Any) -> None:
        """Emit an event via the event bus."""
        await self._event_bus.emit(event)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _argmax(values: list[float]) -> int:
    """Return index of the maximum value."""
    if not values:
        raise ValueError("Cannot find argmax of empty list")
    best_idx = 0
    best_val = values[0]
    for i, v in enumerate(values):
        if v > best_val:
            best_val = v
            best_idx = i
    return best_idx


