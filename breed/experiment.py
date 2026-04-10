"""Experiment runner for multi-seed, multi-baseline agentbreed research.

This module is the experimental backbone of the agentbreed research paper.
It runs a primary evolutionary method against multiple baselines under
matched compute budgets, across multiple seeds, with rigorous logging of
per-question scores, fitness curves, wall-clock time, and API call counts.

Design goals
------------
* **Reproducible** -- given a seed, runs are deterministic.
* **Fair** -- every baseline respects the same compute budget
  (``population_size * generations`` agent invocations on the train set).
* **Auditable** -- every per-question score is recorded so downstream
  statistical analysis (paired bootstraps, sign tests, etc.) is possible.
* **Crash-safe** -- intermediate results are written to disk after each
  ``(method, seed)`` run completes; the runner can resume after a crash.

Usage
-----
::

    from breed.experiment import (
        ExperimentConfig, ExperimentRunner,
    )
    from breed.arenas.base import Task

    async def my_agent(genome_dict, task):
        # ... do something with genome_dict["genes"] and the task ...
        return "0.6"

    def my_scorer(output, expected):
        # higher is better, in [0, 1]
        try:
            p = float(output)
        except (TypeError, ValueError):
            p = 0.5
        # Brier-derived "fitness": 1 - (p - y)^2
        return 1.0 - (p - float(expected)) ** 2

    train = [Task(task_id=str(i), prompt=f"q{i}", expected=1.0) for i in range(10)]
    test  = [Task(task_id=f"t{i}", prompt=f"t{i}", expected=0.0) for i in range(5)]

    cfg = ExperimentConfig(
        name="forecasting_v1",
        seeds=[1, 2, 3],
        population_size=8,
        generations=4,
        elite_count=3,
        mutation_rate=0.15,
        immigration_rate=0.1,
        selection_method="truncation",
        tournament_size=3,
        output_dir="experiments/forecasting_v1",
        methods=["full_evolution", "random_search", "static_best"],
    )

    runner = ExperimentRunner(
        config=cfg,
        agent_fn=my_agent,
        train_tasks=train,
        test_tasks=test,
        scorer=my_scorer,
        gene_template={
            "temperature": {"type": "float", "range": (0.0, 1.5), "mutation_rate": 0.1},
            "system_prompt": {
                "type": "text",
                "templates": ["You are helpful.", "You are precise."],
                "mutation_rate": 0.2,
            },
        },
    )
    result = await runner.run()
    result.to_jsonl("experiments/forecasting_v1/results.jsonl")
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from breed.adapters.callable_adapter import CallableAdapter
from breed.arenas.base import Task
from breed.arenas.custom import CustomArena
from breed.engine import BreedingConfig, BreedingEngine
from breed.events import CallbackObserver, EventBus, GenerationComplete
from breed.genome import Gene, GeneType, Genome
from breed.population import Population


# ---------------------------------------------------------------------------
# Configuration & result dataclasses
# ---------------------------------------------------------------------------


# Methods supported out-of-the-box.
KNOWN_METHODS: tuple[str, ...] = (
    "full_evolution",
    "mutation_only",
    "crossover_only",
    "random_search",
    "static_best",
    "static_ensemble",
    "prompt_only_evolution",
    "bayesian_opt",
    "successive_halving",
    "coordinate_descent",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level config for a multi-seed, multi-baseline experiment.

    Attributes
    ----------
    name:
        Human-readable experiment identifier (used in result filenames).
    seeds:
        List of seeds. Each seed is run for every method.
    population_size:
        Number of genomes per generation (also the budget unit -- baselines
        evaluate ``population_size * generations`` candidates total).
    generations:
        Number of evolutionary generations.
    elite_count:
        Number of survivors per generation in evolutionary methods.
    mutation_rate:
        Per-genome mutation probability.
    immigration_rate:
        Fraction of population replaced by random immigrants per generation.
    selection_method:
        ``"tournament"`` or ``"truncation"``.
    tournament_size:
        Tournament size for tournament selection.
    output_dir:
        Directory to write per-run intermediate JSONL and the aggregate file.
    methods:
        Methods to run. Each must be in :data:`KNOWN_METHODS`.
    metadata:
        Free-form metadata stored alongside the experiment results
        (e.g. paper section, dataset version).
    """

    name: str
    seeds: list[int]
    population_size: int
    generations: int
    elite_count: int
    mutation_rate: float
    immigration_rate: float
    selection_method: str
    tournament_size: int
    output_dir: str
    methods: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "name": self.name,
            "seeds": list(self.seeds),
            "population_size": self.population_size,
            "generations": self.generations,
            "elite_count": self.elite_count,
            "mutation_rate": self.mutation_rate,
            "immigration_rate": self.immigration_rate,
            "selection_method": self.selection_method,
            "tournament_size": self.tournament_size,
            "output_dir": self.output_dir,
            "methods": list(self.methods),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Deserialize from a plain dict."""
        return cls(
            name=data["name"],
            seeds=list(data["seeds"]),
            population_size=int(data["population_size"]),
            generations=int(data["generations"]),
            elite_count=int(data["elite_count"]),
            mutation_rate=float(data["mutation_rate"]),
            immigration_rate=float(data["immigration_rate"]),
            selection_method=str(data["selection_method"]),
            tournament_size=int(data["tournament_size"]),
            output_dir=str(data["output_dir"]),
            methods=list(data["methods"]),
            metadata=dict(data.get("metadata", {})),
        )

    @property
    def compute_budget(self) -> int:
        """Total agent evaluations on train tasks per (method, seed) run."""
        return self.population_size * self.generations


@dataclass
class RunResult:
    """Result of a single ``(method, seed)`` run.

    Attributes
    ----------
    method:
        Name of the method that produced this result.
    seed:
        Seed used for this run.
    champion_genome_id:
        ID of the chosen "champion" genome (the one evaluated on test tasks).
    champion_fitness:
        Mean fitness of the champion on the **train** tasks (used for
        method comparison on the in-distribution objective).
    per_question_scores:
        Per-question scores produced by the champion on the **test** tasks.
        These are the raw values that downstream statistical tests consume.
    fitness_curve:
        Best-of-generation fitness on **train** tasks. For non-evolutionary
        baselines this is monotonically non-decreasing (running maximum).
    wall_clock_seconds:
        Total time spent inside ``run_method``.
    total_evaluations:
        Total number of arena evaluations performed (one per genome per
        evaluation pass on either the train or test set).
    api_calls:
        Total number of agent invocations (one per ``agent_fn`` call).
    metadata:
        Free-form metadata (genome details, intermediate stats, etc.).
    """

    method: str
    seed: int
    champion_genome_id: str
    champion_fitness: float
    per_question_scores: list[float]
    fitness_curve: list[float]
    wall_clock_seconds: float
    total_evaluations: int
    api_calls: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunResult":
        return cls(
            method=str(data["method"]),
            seed=int(data["seed"]),
            champion_genome_id=str(data["champion_genome_id"]),
            champion_fitness=float(data["champion_fitness"]),
            per_question_scores=[float(x) for x in data["per_question_scores"]],
            fitness_curve=[float(x) for x in data["fitness_curve"]],
            wall_clock_seconds=float(data["wall_clock_seconds"]),
            total_evaluations=int(data["total_evaluations"]),
            api_calls=int(data["api_calls"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ExperimentResult:
    """Aggregated result of an experiment across all methods and seeds."""

    config: ExperimentConfig
    runs: list[RunResult]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # -- queries -------------------------------------------------------------

    def runs_for_method(self, method: str) -> list[RunResult]:
        """Return every run produced by *method* (across all seeds)."""
        return [r for r in self.runs if r.method == method]

    def runs_for_seed(self, seed: int) -> list[RunResult]:
        """Return every run produced for *seed* (across all methods)."""
        return [r for r in self.runs if r.seed == seed]

    # -- (de)serialization ---------------------------------------------------

    def to_jsonl(self, path: str | Path) -> None:
        """Write all runs to a JSONL file (one run per line).

        The first line contains a header record with the config and timestamp,
        prefixed with ``"__header__": true`` so it can be skipped on read.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            header = {
                "__header__": True,
                "config": self.config.to_dict(),
                "timestamp": self.timestamp,
            }
            f.write(json.dumps(header) + "\n")
            for run in self.runs:
                f.write(json.dumps(run.to_dict()) + "\n")

    @classmethod
    def from_jsonl(
        cls, path: str | Path, config: ExperimentConfig | None = None
    ) -> "ExperimentResult":
        """Read an experiment result back from a JSONL file.

        If *config* is None, the embedded header config is used. If both
        the header and *config* are provided, the explicit *config* takes
        precedence (useful when migrating result files).
        """
        path = Path(path)
        runs: list[RunResult] = []
        loaded_config: ExperimentConfig | None = None
        timestamp: str | None = None

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("__header__"):
                    loaded_config = ExperimentConfig.from_dict(obj["config"])
                    timestamp = obj.get("timestamp")
                    continue
                runs.append(RunResult.from_dict(obj))

        final_config = config or loaded_config
        if final_config is None:
            raise ValueError(
                "from_jsonl: no config provided and the file has no header"
            )

        return cls(
            config=final_config,
            runs=runs,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_seed_chain(master_seed: int) -> random.Random:
    """Build an RNG seeded deterministically from *master_seed*."""
    return random.Random(master_seed)


def _argmax(values: list[float]) -> int:
    if not values:
        raise ValueError("Cannot argmax over empty list")
    best_idx = 0
    best_val = values[0]
    for i, v in enumerate(values):
        if v > best_val:
            best_val = v
            best_idx = i
    return best_idx


def _running_max(values: list[float]) -> list[float]:
    """Return the running maximum of *values* (cumulative best)."""
    out: list[float] = []
    best = float("-inf")
    for v in values:
        if v > best:
            best = v
        out.append(best)
    return out


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


class ExperimentRunner:
    """Run a complete experiment with multiple methods and seeds.

    Parameters
    ----------
    config:
        :class:`ExperimentConfig` describing the experiment.
    agent_fn:
        Async callable ``(genome_dict, task) -> str``. Same signature as
        :func:`breed.api.evolve`'s ``agent`` argument.
    train_tasks:
        Tasks used for **training** -- evolutionary methods see these
        repeatedly during evolution; baselines also evaluate on these to
        choose their champion.
    test_tasks:
        Tasks used for **evaluation** -- the chosen champion of every
        method is evaluated on these once. Per-question scores are
        recorded so downstream statistical tests can be performed.
    scorer:
        Callable ``(agent_output, expected) -> float`` returning a value
        in [0, 1] where higher is better.
    gene_template:
        Dict describing the evolvable genes (same format as
        :func:`breed.api.evolve`'s ``genes`` argument).
    """

    def __init__(
        self,
        config: ExperimentConfig,
        agent_fn: Callable[[dict[str, Any], str], Awaitable[str]],
        train_tasks: list[Task],
        test_tasks: list[Task],
        scorer: Callable[[str, Any], float],
        gene_template: dict[str, dict[str, Any]],
    ) -> None:
        self._validate_config(config)
        self.config = config
        self._raw_agent_fn = agent_fn
        self.train_tasks = list(train_tasks)
        self.test_tasks = list(test_tasks)
        self.scorer = scorer
        self.gene_template = gene_template

        # Counters reset per (method, seed) run.
        self._call_count = 0
        self._eval_count = 0

        # Output dir setup
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # -- validation ----------------------------------------------------------

    @staticmethod
    def _validate_config(config: ExperimentConfig) -> None:
        if not config.seeds:
            raise ValueError("ExperimentConfig.seeds must be non-empty")
        if not config.methods:
            raise ValueError("ExperimentConfig.methods must be non-empty")
        if config.population_size < 1:
            raise ValueError("population_size must be >= 1")
        if config.generations < 1:
            raise ValueError("generations must be >= 1")
        if config.elite_count < 1 or config.elite_count > config.population_size:
            raise ValueError(
                "elite_count must be in [1, population_size]"
            )
        unknown = [m for m in config.methods if m not in KNOWN_METHODS]
        if unknown:
            raise ValueError(
                f"Unknown methods: {unknown}. Valid methods: {list(KNOWN_METHODS)}"
            )

    # -- counter wrapper -----------------------------------------------------

    def _wrap_agent(self) -> Callable[[dict[str, Any], str], Awaitable[str]]:
        """Return a counter-wrapped agent_fn that increments self._call_count."""
        raw_fn = self._raw_agent_fn

        async def _wrapped(genome_dict: dict[str, Any], task: str) -> str:
            self._call_count += 1
            return await raw_fn(genome_dict, task)

        return _wrapped

    def _build_arena(self, tasks: list[Task]) -> CustomArena:
        return CustomArena(tasks=tasks, scorer=self.scorer)

    def _build_adapter(self) -> CallableAdapter:
        return CallableAdapter(self._wrap_agent())

    def _reset_counters(self) -> None:
        self._call_count = 0
        self._eval_count = 0

    # -- public dispatcher ---------------------------------------------------

    async def run_method(self, method: str, seed: int) -> RunResult:
        """Run a single ``(method, seed)`` combination."""
        if method == "full_evolution":
            return await self._run_full_evolution(seed)
        if method == "mutation_only":
            return await self._run_mutation_only(seed)
        if method == "crossover_only":
            return await self._run_crossover_only(seed)
        if method == "random_search":
            return await self._run_random_search(seed)
        if method == "static_best":
            return await self._run_static_best(seed)
        if method == "static_ensemble":
            return await self._run_static_ensemble(seed)
        if method == "prompt_only_evolution":
            return await self._run_prompt_only(seed)
        if method == "bayesian_opt":
            return await self._run_bayesian_opt(seed)
        if method == "successive_halving":
            return await self._run_successive_halving(seed)
        if method == "coordinate_descent":
            return await self._run_coordinate_descent(seed)
        raise ValueError(
            f"Unknown method: {method!r}. Valid methods: {list(KNOWN_METHODS)}"
        )

    # -- top-level orchestrator ---------------------------------------------

    async def run(
        self,
        save_intermediate: bool = True,
        resume: bool = True,
    ) -> ExperimentResult:
        """Run all ``(method, seed)`` combinations.

        Parameters
        ----------
        save_intermediate:
            If True (default) write each completed :class:`RunResult` to a
            per-run JSONL file under ``output_dir`` immediately after it
            finishes. This means a long run can be resumed if it crashes.
        resume:
            If True, skip ``(method, seed)`` combinations whose intermediate
            file already exists, and load the existing result instead.
        """
        runs: list[RunResult] = []

        for method in self.config.methods:
            for seed in self.config.seeds:
                inter_path = self._intermediate_path(method, seed)

                if resume and inter_path.exists():
                    try:
                        with inter_path.open("r", encoding="utf-8") as f:
                            data = json.loads(f.read())
                        existing = RunResult.from_dict(data)
                        runs.append(existing)
                        continue
                    except (OSError, json.JSONDecodeError, KeyError, ValueError):
                        # Corrupt intermediate -- re-run.
                        pass

                run_result = await self.run_method(method, seed)
                runs.append(run_result)

                if save_intermediate:
                    self._write_intermediate(inter_path, run_result)

        result = ExperimentResult(config=self.config, runs=runs)

        # Final aggregate file
        aggregate_path = self._output_dir / f"{self.config.name}.jsonl"
        result.to_jsonl(aggregate_path)

        return result

    # -- intermediate I/O ----------------------------------------------------

    def _intermediate_path(self, method: str, seed: int) -> Path:
        return self._output_dir / "runs" / f"{method}__seed{seed}.json"

    def _write_intermediate(self, path: Path, run: RunResult) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(run.to_dict(), f)
        # Atomic-ish replace; on Windows this still works for non-locked files.
        tmp.replace(path)

    # -- shared evaluation helpers ------------------------------------------

    async def _evaluate_on_train(
        self, genome: Genome, adapter: CallableAdapter
    ) -> float:
        """Evaluate a genome on **train** tasks. Returns mean score in [0, 1]."""
        arena = self._build_arena(self.train_tasks)
        result = await arena.evaluate(genome, adapter, self.train_tasks)
        self._eval_count += 1
        return float(result.fitness_score)

    async def _evaluate_on_test(self, genome: Genome) -> list[float]:
        """Run *genome* on the test set, return per-question scores.

        Always uses a fresh adapter so the call counter still ticks.
        """
        adapter = self._build_adapter()
        scores: list[float] = []
        for task in self.test_tasks:
            agent_result = await adapter.run(genome, task.prompt)
            if agent_result.success:
                raw = self.scorer(agent_result.output, task.expected)
                # clamp to [0, 1] to be safe
                clamped = max(0.0, min(1.0, float(raw)))
            else:
                clamped = 0.0
            scores.append(clamped)
        self._eval_count += 1
        return scores

    # -- per-method runners --------------------------------------------------

    async def _run_full_evolution(self, seed: int) -> RunResult:
        """Full breeding loop: spawn -> evaluate -> select -> breed -> mutate -> immigrate."""
        return await self._run_evolution_variant(
            seed=seed,
            method_name="full_evolution",
            allow_mutation=True,
            allow_crossover=True,
            mutation_rate_override=None,
            prompt_only=False,
        )

    async def _run_mutation_only(self, seed: int) -> RunResult:
        """Same as full evolution but skip the crossover step.

        Children are produced as direct copies of randomly selected elites
        (via ``Population.breed`` with no crossover) and then mutated at
        full force.
        """
        return await self._run_evolution_variant(
            seed=seed,
            method_name="mutation_only",
            allow_mutation=True,
            allow_crossover=False,
            mutation_rate_override=None,
            prompt_only=False,
        )

    async def _run_crossover_only(self, seed: int) -> RunResult:
        """Crossover but no mutation."""
        return await self._run_evolution_variant(
            seed=seed,
            method_name="crossover_only",
            allow_mutation=False,
            allow_crossover=True,
            mutation_rate_override=0.0,
            prompt_only=False,
        )

    async def _run_prompt_only(self, seed: int) -> RunResult:
        """Evolve only the ``prompt_template`` gene; freeze all others.

        Implementation: extract only the prompt_template gene from the
        user's template (regardless of type -- TEXT, ENUM, etc.) and run
        a full evolution loop over that reduced search space. If the
        template does not contain a ``prompt_template`` gene, fall back
        to full evolution and mark the result accordingly.
        """
        if "prompt_template" not in self.gene_template:
            # No prompt_template gene -> nothing to isolate. Fall back.
            result = await self._run_full_evolution(seed)
            # Relabel so the caller sees this as prompt_only_evolution.
            return RunResult(
                method="prompt_only_evolution",
                seed=result.seed,
                champion_genome_id=result.champion_genome_id,
                champion_fitness=result.champion_fitness,
                per_question_scores=result.per_question_scores,
                fitness_curve=result.fitness_curve,
                wall_clock_seconds=result.wall_clock_seconds,
                total_evaluations=result.total_evaluations,
                api_calls=result.api_calls,
                metadata={
                    **result.metadata,
                    "prompt_only_fallback": True,
                    "reason": "no prompt_template gene in template",
                },
            )

        prompt_only_template = {
            "prompt_template": self.gene_template["prompt_template"]
        }

        return await self._run_evolution_variant(
            seed=seed,
            method_name="prompt_only_evolution",
            allow_mutation=True,
            allow_crossover=True,
            mutation_rate_override=None,
            prompt_only=True,
            override_template=prompt_only_template,
        )

    async def _run_evolution_variant(
        self,
        seed: int,
        method_name: str,
        allow_mutation: bool,
        allow_crossover: bool,
        mutation_rate_override: float | None,
        prompt_only: bool,
        override_template: dict[str, dict[str, Any]] | None = None,
    ) -> RunResult:
        """Shared evolutionary loop with knobs for the variant baselines.

        This re-implements the core breeding loop directly (rather than
        using :class:`BreedingEngine`) so the variants can disable mutation
        or crossover cleanly without forking the engine.
        """
        self._reset_counters()
        start = time.perf_counter()
        rng = _make_seed_chain(seed)

        template = override_template or self.gene_template

        # Spawn initial population
        population = Population.spawn(
            self.config.population_size,
            seed=rng.randint(0, 2**31 - 1),
            gene_template=template,
        )

        adapter = self._build_adapter()
        train_arena = self._build_arena(self.train_tasks)

        fitness_curve: list[float] = []
        champion_genome: Genome | None = None
        champion_fitness = float("-inf")

        for gen in range(self.config.generations):
            # Evaluate every member on train tasks
            scores: list[float] = []
            for genome in population.members:
                eval_result = await train_arena.evaluate(
                    genome, adapter, self.train_tasks
                )
                self._eval_count += 1
                scores.append(float(eval_result.fitness_score))

            # Track champion
            best_idx = _argmax(scores)
            best_fitness = scores[best_idx]
            best_genome = population.members[best_idx]

            if best_fitness > champion_fitness:
                champion_fitness = best_fitness
                champion_genome = best_genome

            fitness_curve.append(champion_fitness)

            # Stop early if this is the last generation -- no need to breed
            if gen == self.config.generations - 1:
                break

            # Selection -- preserve a fitness lookup so we can pass scores
            # for the surviving elites to ``Population.breed``.
            id_to_fitness: dict[str, float] = {
                g.genome_id: s for g, s in zip(population.members, scores)
            }
            population.select(
                fitness_scores=scores,
                top_k=self.config.elite_count,
                method=self.config.selection_method,  # type: ignore[arg-type]
                tournament_size=self.config.tournament_size,
                seed=rng.randint(0, 2**31 - 1),
            )

            # Breed
            if allow_crossover:
                elite_fitness = [
                    id_to_fitness.get(g.genome_id, 0.0)
                    for g in population.members
                ]
                population.breed(
                    target_size=self.config.population_size,
                    seed=rng.randint(0, 2**31 - 1),
                    fitness_scores=elite_fitness,
                )
            else:
                # Mutation-only: refill the population by cloning random
                # elites (no crossover). Population.breed currently always
                # crosses over, so we do this manually using model_copy.
                self._refill_with_clones(
                    population, target_size=self.config.population_size, rng=rng
                )

            # Mutate
            effective_mut_rate = (
                mutation_rate_override
                if mutation_rate_override is not None
                else self.config.mutation_rate
            )
            if allow_mutation and effective_mut_rate > 0:
                population.mutate(
                    rate=effective_mut_rate,
                    seed=rng.randint(0, 2**31 - 1),
                )

            # Immigration
            if self.config.immigration_rate > 0:
                immigrant_count = max(
                    1,
                    int(self.config.population_size * self.config.immigration_rate),
                )
                population.add_immigrants(
                    immigrant_count, seed=rng.randint(0, 2**31 - 1)
                )

        if champion_genome is None:
            raise RuntimeError(
                f"Method {method_name!r} produced no champion -- empty population"
            )

        # Final per-question evaluation on the test set
        per_q_scores = await self._evaluate_on_test(champion_genome)

        wall_clock = time.perf_counter() - start

        return RunResult(
            method=method_name,
            seed=seed,
            champion_genome_id=champion_genome.genome_id,
            champion_fitness=champion_fitness,
            per_question_scores=per_q_scores,
            fitness_curve=fitness_curve,
            wall_clock_seconds=wall_clock,
            total_evaluations=self._eval_count,
            api_calls=self._call_count,
            metadata={
                "prompt_only": prompt_only,
                "allow_mutation": allow_mutation,
                "allow_crossover": allow_crossover,
                "compute_budget": self.config.compute_budget,
                "champion_genes": {
                    name: gene.value
                    for name, gene in champion_genome.genes.items()
                },
            },
        )

    @staticmethod
    def _refill_with_clones(
        population: Population, target_size: int, rng: random.Random
    ) -> None:
        """Fill *population* up to *target_size* with clones of random members.

        Used by the mutation-only baseline (no crossover).
        """
        elites = list(population.members)
        if not elites:
            raise RuntimeError("Cannot refill population: no elites available")

        children: list[Genome] = []
        while len(elites) + len(children) < target_size:
            parent = rng.choice(elites)
            # Make a deep copy with a fresh genome_id and incremented generation
            clone = parent.model_copy(deep=True)
            clone = clone.model_copy(
                update={
                    "genome_id": _new_genome_id(),
                    "generation": parent.generation + 1,
                    "parents": [parent.genome_id],
                }
            )
            children.append(clone)

        population.members = elites + children
        population.generation += 1

    # -- baselines that aren't evolutionary ---------------------------------

    async def _run_random_search(self, seed: int) -> RunResult:
        """Sample N*G random configs, return best on train set, evaluate on test."""
        self._reset_counters()
        start = time.perf_counter()
        rng = _make_seed_chain(seed)

        adapter = self._build_adapter()
        train_arena = self._build_arena(self.train_tasks)

        budget = self.config.compute_budget
        all_genomes: list[Genome] = []
        all_scores: list[float] = []

        for _ in range(budget):
            sub_seed = rng.randint(0, 2**31 - 1)
            pop = Population.spawn(
                1, seed=sub_seed, gene_template=self.gene_template
            )
            genome = pop.members[0]
            eval_result = await train_arena.evaluate(
                genome, adapter, self.train_tasks
            )
            self._eval_count += 1
            score = float(eval_result.fitness_score)
            all_genomes.append(genome)
            all_scores.append(score)

        # Champion = best on train
        best_idx = _argmax(all_scores)
        champion = all_genomes[best_idx]
        champion_fitness = all_scores[best_idx]

        # Fitness curve = running max over the sampled order
        fitness_curve = _running_max(all_scores)

        per_q_scores = await self._evaluate_on_test(champion)

        wall_clock = time.perf_counter() - start

        return RunResult(
            method="random_search",
            seed=seed,
            champion_genome_id=champion.genome_id,
            champion_fitness=champion_fitness,
            per_question_scores=per_q_scores,
            fitness_curve=fitness_curve,
            wall_clock_seconds=wall_clock,
            total_evaluations=self._eval_count,
            api_calls=self._call_count,
            metadata={
                "compute_budget": budget,
                "samples": budget,
                "champion_genes": {
                    name: gene.value for name, gene in champion.genes.items()
                },
            },
        )

    async def _run_static_best(self, seed: int) -> RunResult:
        """Pick the best of N (=population_size) random initial configs.

        This baseline uses only ``population_size`` train evaluations -- it
        does NOT consume the full compute budget. The remaining budget is
        recorded in metadata for honest reporting.
        """
        self._reset_counters()
        start = time.perf_counter()
        rng = _make_seed_chain(seed)

        adapter = self._build_adapter()
        train_arena = self._build_arena(self.train_tasks)

        population = Population.spawn(
            self.config.population_size,
            seed=rng.randint(0, 2**31 - 1),
            gene_template=self.gene_template,
        )

        scores: list[float] = []
        for genome in population.members:
            eval_result = await train_arena.evaluate(
                genome, adapter, self.train_tasks
            )
            self._eval_count += 1
            scores.append(float(eval_result.fitness_score))

        best_idx = _argmax(scores)
        champion = population.members[best_idx]
        champion_fitness = scores[best_idx]

        # Fitness curve has length 1 (only one "generation")
        fitness_curve = [champion_fitness]

        per_q_scores = await self._evaluate_on_test(champion)

        wall_clock = time.perf_counter() - start

        return RunResult(
            method="static_best",
            seed=seed,
            champion_genome_id=champion.genome_id,
            champion_fitness=champion_fitness,
            per_question_scores=per_q_scores,
            fitness_curve=fitness_curve,
            wall_clock_seconds=wall_clock,
            total_evaluations=self._eval_count,
            api_calls=self._call_count,
            metadata={
                "compute_budget": self.config.compute_budget,
                "compute_used": self.config.population_size,
                "champion_genes": {
                    name: gene.value for name, gene in champion.genes.items()
                },
            },
        )

    async def _run_static_ensemble(self, seed: int) -> RunResult:
        """Ensemble of N random configs: average their predictions on every task.

        For each test question, every member of the ensemble produces an
        output. The scorer is applied to each output independently, and
        the **mean per-question score** is reported.

        Champion fitness on the train set is computed similarly: every
        member is evaluated, and the mean train-set fitness is reported.
        """
        self._reset_counters()
        start = time.perf_counter()
        rng = _make_seed_chain(seed)

        adapter = self._build_adapter()
        train_arena = self._build_arena(self.train_tasks)

        population = Population.spawn(
            self.config.population_size,
            seed=rng.randint(0, 2**31 - 1),
            gene_template=self.gene_template,
        )

        # Train-set fitness for each member
        member_fitness: list[float] = []
        for genome in population.members:
            eval_result = await train_arena.evaluate(
                genome, adapter, self.train_tasks
            )
            self._eval_count += 1
            member_fitness.append(float(eval_result.fitness_score))

        ensemble_train_fitness = (
            sum(member_fitness) / len(member_fitness) if member_fitness else 0.0
        )
        fitness_curve = [ensemble_train_fitness]

        # Test set: per-question mean across the ensemble
        per_question_aggregate: list[list[float]] = [[] for _ in self.test_tasks]
        for genome in population.members:
            scores = await self._evaluate_on_test(genome)
            for i, s in enumerate(scores):
                per_question_aggregate[i].append(s)

        per_q_scores = [
            sum(col) / len(col) if col else 0.0 for col in per_question_aggregate
        ]

        # The "champion" of an ensemble is a synthetic notion -- we report
        # the genome ID of the highest-train-fitness member as a placeholder
        # so downstream code has *some* identifier.
        best_member_idx = _argmax(member_fitness) if member_fitness else 0
        placeholder_champion = population.members[best_member_idx]

        wall_clock = time.perf_counter() - start

        return RunResult(
            method="static_ensemble",
            seed=seed,
            champion_genome_id=placeholder_champion.genome_id,
            champion_fitness=ensemble_train_fitness,
            per_question_scores=per_q_scores,
            fitness_curve=fitness_curve,
            wall_clock_seconds=wall_clock,
            total_evaluations=self._eval_count,
            api_calls=self._call_count,
            metadata={
                "compute_budget": self.config.compute_budget,
                "ensemble_size": self.config.population_size,
                "member_fitness": member_fitness,
            },
        )

    # -- Bayesian optimisation (TPE) ----------------------------------------

    async def _run_bayesian_opt(self, seed: int) -> RunResult:
        """Bayesian optimisation via Tree-structured Parzen Estimator (TPE).

        Implementation notes
        --------------------
        * Same compute budget as other methods: ``population_size *
          generations`` LLM evaluations on the train set.
        * Warmup: the first ``_TPE_WARMUP`` iterations sample uniformly at
          random. After warmup, each iteration draws ``_TPE_N_CANDIDATES``
          random candidates and picks the one with the highest TPE-based
          Expected Improvement score.
        * Encoding: ENUM genes become categorical distributions, FLOAT genes
          become continuous (Gaussian KDE), SET genes become multi-hot
          binary vectors. TEXT / NUMERIC_VECTOR genes are encoded as
          categoricals over the index of the value in the raw template
          (fallback: treated as a single bucket).

        The acquisition score is::

            acq(x) = log(l(x) + eps) - log(g(x) + eps)

        where ``l`` is the KDE fit on the "good" history (top fraction by
        fitness) and ``g`` is the KDE fit on the "bad" history (the rest).
        """
        self._reset_counters()
        start = time.perf_counter()
        rng = _make_seed_chain(seed)
        np_rng = np.random.default_rng(seed)

        adapter = self._build_adapter()
        train_arena = self._build_arena(self.train_tasks)

        budget = self.config.compute_budget
        history_genomes: list[Genome] = []
        history_scores: list[float] = []
        history_encoded: list[dict[str, Any]] = []

        # Pre-compute encoder spec from the gene template so that the
        # encoding is consistent across candidates.
        encoder = _TPEEncoder(self.gene_template)

        for iteration in range(budget):
            if iteration < _TPE_WARMUP or len(history_scores) < 2:
                # Warmup: pure random sample.
                sub_seed = rng.randint(0, 2**31 - 1)
                pop = Population.spawn(
                    1, seed=sub_seed, gene_template=self.gene_template
                )
                candidate = pop.members[0]
            else:
                # Draw N candidates and pick the one with the best
                # acquisition score under the current TPE split.
                candidates: list[Genome] = []
                candidate_encoded: list[dict[str, Any]] = []
                for _ in range(_TPE_N_CANDIDATES):
                    sub_seed = rng.randint(0, 2**31 - 1)
                    pop = Population.spawn(
                        1, seed=sub_seed, gene_template=self.gene_template
                    )
                    g = pop.members[0]
                    candidates.append(g)
                    candidate_encoded.append(encoder.encode(g))

                good_mask, bad_mask = _split_history_mask(
                    history_scores, _TPE_SPLIT_FRACTION
                )
                if sum(good_mask) < 1 or sum(bad_mask) < 1:
                    # Not enough data to split -- fall back to the first
                    # candidate (still better than re-evaluating already
                    # seen points).
                    best_idx = 0
                else:
                    good_hist = [
                        h for h, m in zip(history_encoded, good_mask) if m
                    ]
                    bad_hist = [
                        h for h, m in zip(history_encoded, bad_mask) if m
                    ]
                    scores = [
                        encoder.tpe_acquisition(
                            enc,
                            good_hist,
                            bad_hist,
                            rng=np_rng,
                        )
                        for enc in candidate_encoded
                    ]
                    best_idx = _argmax(scores)
                candidate = candidates[best_idx]

            eval_result = await train_arena.evaluate(
                candidate, adapter, self.train_tasks
            )
            self._eval_count += 1
            score = float(eval_result.fitness_score)

            history_genomes.append(candidate)
            history_scores.append(score)
            history_encoded.append(encoder.encode(candidate))

        best_idx = _argmax(history_scores)
        champion = history_genomes[best_idx]
        champion_fitness = history_scores[best_idx]

        fitness_curve = _running_max(history_scores)
        per_q_scores = await self._evaluate_on_test(champion)

        wall_clock = time.perf_counter() - start

        return RunResult(
            method="bayesian_opt",
            seed=seed,
            champion_genome_id=champion.genome_id,
            champion_fitness=champion_fitness,
            per_question_scores=per_q_scores,
            fitness_curve=fitness_curve,
            wall_clock_seconds=wall_clock,
            total_evaluations=self._eval_count,
            api_calls=self._call_count,
            metadata={
                "compute_budget": budget,
                "tpe_warmup": _TPE_WARMUP,
                "tpe_n_candidates": _TPE_N_CANDIDATES,
                "tpe_split_fraction": _TPE_SPLIT_FRACTION,
                "champion_genes": {
                    name: gene.value for name, gene in champion.genes.items()
                },
            },
        )

    # -- Successive halving (Hyperband-style) -------------------------------

    async def _run_successive_halving(self, seed: int) -> RunResult:
        """Successive halving multi-fidelity search.

        Implementation
        --------------
        Starts with a large pool of random genomes, evaluates them at a
        cheap fidelity (a small subset of training tasks), keeps the top
        ``1/eta`` by score, then multiplies the fidelity by ``eta`` and
        repeats until the fidelity reaches the full training set.

        Budget is measured in *task evaluations* (each genome-on-task pair
        is one LLM call on the search set). The total budget equals
        ``population_size * generations * len(train_tasks)`` to stay in
        lock-step with the other baselines which perform
        ``population_size * generations`` full-train evaluations (each of
        which costs ``len(train_tasks)`` LLM calls). ``N_0`` is chosen so
        that every rung spends roughly an equal budget, matching the
        classical Hyperband analysis.
        """
        self._reset_counters()
        start = time.perf_counter()
        rng = _make_seed_chain(seed)

        adapter = self._build_adapter()
        full_train_arena = self._build_arena(self.train_tasks)

        n_train = len(self.train_tasks)
        if n_train == 0:
            raise RuntimeError(
                "successive_halving: train_tasks must be non-empty"
            )

        eta = _SH_ETA
        fidelity_max = n_train
        fidelity_min = min(_SH_FIDELITY_MIN, n_train)

        # Total task-level budget (LLM calls on search set).
        total_task_budget = (
            self.config.population_size
            * self.config.generations
            * n_train
        )

        # Build the fidelity schedule.
        schedule: list[int] = []
        f = fidelity_min
        while True:
            schedule.append(f)
            if f >= fidelity_max:
                break
            next_f = int(round(f * eta))
            if next_f <= f:
                next_f = f + 1
            if next_f >= fidelity_max:
                schedule.append(fidelity_max)
                break
            f = next_f
        # De-duplicate consecutive equal rungs (in case fidelity_min ==
        # fidelity_max).
        dedup_schedule: list[int] = []
        for fid in schedule:
            if not dedup_schedule or dedup_schedule[-1] != fid:
                dedup_schedule.append(fid)
        schedule = dedup_schedule
        n_rungs = len(schedule)

        # Choose N_0 so that each rung consumes approximately an equal
        # share of the budget. With halving ratio eta, rung i holds
        # ``N_0 / eta^i`` survivors evaluated at fidelity ``schedule[i]``.
        rung_cost_per_n0 = sum(
            schedule[i] / (eta ** i) for i in range(n_rungs)
        )
        if rung_cost_per_n0 <= 0:
            rung_cost_per_n0 = float(fidelity_max)
        n0 = max(2, int(round(total_task_budget / rung_cost_per_n0)))

        # Sample the initial pool.
        pop = Population.spawn(
            n0,
            seed=rng.randint(0, 2**31 - 1),
            gene_template=self.gene_template,
        )
        survivors: list[Genome] = list(pop.members)

        fitness_curve: list[float] = []
        last_fitness_at_rung: dict[str, float] = {}
        rung_details: list[dict[str, Any]] = []
        task_budget_left = total_task_budget

        for rung_idx, fidelity in enumerate(schedule):
            if not survivors:
                break

            # Random subset of training tasks for this fidelity.
            fidelity = min(fidelity, n_train)
            task_indices = rng.sample(range(n_train), fidelity)
            rung_tasks = [self.train_tasks[i] for i in task_indices]
            rung_arena = self._build_arena(rung_tasks)

            scored: list[tuple[Genome, float]] = []
            for genome in survivors:
                if task_budget_left < fidelity:
                    # Out of budget -- stop early, keep what we have.
                    break
                eval_result = await rung_arena.evaluate(
                    genome, adapter, rung_tasks
                )
                self._eval_count += 1
                score = float(eval_result.fitness_score)
                scored.append((genome, score))
                last_fitness_at_rung[genome.genome_id] = score
                task_budget_left -= fidelity

            if not scored:
                break

            scored.sort(key=lambda item: item[1], reverse=True)
            best_score = scored[0][1]
            fitness_curve.append(best_score)

            rung_details.append(
                {
                    "rung": rung_idx,
                    "fidelity": fidelity,
                    "n_evaluated": len(scored),
                    "best_score": best_score,
                }
            )

            if rung_idx == n_rungs - 1 or task_budget_left <= 0:
                survivors = [g for g, _ in scored]
                break

            # Keep the top 1/eta.
            keep = max(1, len(scored) // eta)
            survivors = [g for g, _ in scored[:keep]]

        if not survivors:
            raise RuntimeError(
                "successive_halving produced no survivors"
            )

        # Re-evaluate the finalists at full fidelity on the *entire* train
        # set to pick the champion, but only if budget allows. Otherwise,
        # pick the highest-scoring survivor from the last rung.
        champion: Genome | None = None
        champion_fitness = float("-inf")

        can_afford_final = task_budget_left >= len(survivors) * n_train
        if can_afford_final:
            for genome in survivors:
                eval_result = await full_train_arena.evaluate(
                    genome, adapter, self.train_tasks
                )
                self._eval_count += 1
                score = float(eval_result.fitness_score)
                task_budget_left -= n_train
                if score > champion_fitness:
                    champion_fitness = score
                    champion = genome
            if champion is not None:
                fitness_curve.append(champion_fitness)
        else:
            # Fallback: pick the survivor with the best last-rung score.
            scored_finalists = [
                (g, last_fitness_at_rung.get(g.genome_id, float("-inf")))
                for g in survivors
            ]
            scored_finalists.sort(key=lambda item: item[1], reverse=True)
            champion, champion_fitness = scored_finalists[0]

        if champion is None:
            # Shouldn't happen, but guard anyway.
            champion = survivors[0]
            champion_fitness = last_fitness_at_rung.get(
                champion.genome_id, 0.0
            )

        per_q_scores = await self._evaluate_on_test(champion)

        wall_clock = time.perf_counter() - start

        return RunResult(
            method="successive_halving",
            seed=seed,
            champion_genome_id=champion.genome_id,
            champion_fitness=float(champion_fitness),
            per_question_scores=per_q_scores,
            fitness_curve=fitness_curve,
            wall_clock_seconds=wall_clock,
            total_evaluations=self._eval_count,
            api_calls=self._call_count,
            metadata={
                "compute_budget": self.config.compute_budget,
                "task_budget": total_task_budget,
                "task_budget_remaining": max(0, task_budget_left),
                "eta": eta,
                "fidelity_min": fidelity_min,
                "fidelity_max": fidelity_max,
                "fidelity_schedule": list(schedule),
                "n0": n0,
                "rungs": rung_details,
                "champion_genes": {
                    name: gene.value for name, gene in champion.genes.items()
                },
            },
        )

    # ----------------------------------------------------------------------
    # Coordinate descent baseline
    # ----------------------------------------------------------------------

    async def _run_coordinate_descent(self, seed: int) -> RunResult:
        """Greedy coordinate descent over a single incumbent genome.

        Implementation notes
        --------------------
        * Starts from a random genome spawned with the per-seed RNG.
        * Iterates over genes in a fixed random order. For each gene, tries
          up to ``_CD_MAX_ALTERNATIVES`` alternative values drawn from the
          gene's domain and keeps the change if it improves the train-set
          fitness. For ENUM / SET genes the alternatives enumerate the
          option list; for FLOAT genes the alternatives are linearly
          spaced in [lo, hi].
        * Budget matching: evaluations are counted against
          ``compute_budget``. The loop stops as soon as the budget is
          exhausted, regardless of whether a full sweep completed.

        This baseline neutralizes the "maybe greedy local search would
        exploit the interaction structure" reviewer objection.
        """
        self._reset_counters()
        start = time.perf_counter()
        rng = _make_seed_chain(seed)

        adapter = self._build_adapter()
        train_arena = self._build_arena(self.train_tasks)
        budget = self.config.compute_budget

        # Initialize the incumbent
        incumbent = Population.spawn(
            1, seed=rng.randint(0, 2**31 - 1), gene_template=self.gene_template
        ).members[0]

        async def _score(genome: Genome) -> float:
            r = await train_arena.evaluate(genome, adapter, self.train_tasks)
            self._eval_count += 1
            return float(r.fitness_score)

        incumbent_fit = await _score(incumbent)
        fitness_curve: list[float] = [incumbent_fit]
        evals_used = 1

        # Fix a random order over genes so determinism is preserved.
        gene_names = list(self.gene_template.keys())
        rng.shuffle(gene_names)

        _CD_MAX_ALTERNATIVES = 4

        sweeps_completed = 0
        sweeps_without_improvement = 0

        while evals_used < budget and sweeps_without_improvement < 2:
            improved_this_sweep = False
            for gene_name in gene_names:
                if evals_used >= budget:
                    break
                spec = self.gene_template[gene_name]
                gtype = str(spec.get("type", "")).lower()

                # Build alternative values for this gene
                alternatives: list[Any] = []
                current_val = incumbent.genes[gene_name].value
                if gtype in {"enum"}:
                    options = list(spec.get("options") or [])
                    alternatives = [o for o in options if o != current_val]
                elif gtype in {"float"}:
                    lo, hi = spec.get("range", (0.0, 1.0))
                    # 4 linearly spaced candidates across the range
                    n_steps = _CD_MAX_ALTERNATIVES
                    step_vals = [
                        lo + (hi - lo) * (k + 0.5) / n_steps for k in range(n_steps)
                    ]
                    alternatives = [v for v in step_vals if abs(v - current_val) > 1e-9]
                elif gtype in {"set"}:
                    available = list(spec.get("available") or [])
                    cur_set = set(current_val or [])
                    # Each alternative toggles one element on/off
                    alternatives = []
                    for item in available:
                        if item in cur_set:
                            new_val = sorted(cur_set - {item})
                        else:
                            new_val = sorted(cur_set | {item})
                        if tuple(new_val) != tuple(sorted(current_val)):
                            alternatives.append(new_val)
                else:
                    continue  # TEXT / NUMERIC_VECTOR skipped for simplicity

                # Cap and shuffle so seed determines exploration order
                rng.shuffle(alternatives)
                alternatives = alternatives[:_CD_MAX_ALTERNATIVES]

                for alt_val in alternatives:
                    if evals_used >= budget:
                        break
                    # Make a candidate with the single-gene change
                    new_genes = dict(incumbent.genes)
                    old_gene = new_genes[gene_name]
                    new_genes[gene_name] = old_gene.model_copy(update={"value": alt_val})
                    candidate = incumbent.model_copy(
                        update={
                            "genes": new_genes,
                            "genome_id": _new_genome_id(),
                            "generation": incumbent.generation + 1,
                            "parents": [incumbent.genome_id],
                        }
                    )
                    cand_fit = await _score(candidate)
                    evals_used += 1
                    if cand_fit > incumbent_fit:
                        incumbent = candidate
                        incumbent_fit = cand_fit
                        improved_this_sweep = True
                        fitness_curve.append(incumbent_fit)
                        break  # accept improvement and move to next gene
                else:
                    fitness_curve.append(incumbent_fit)

            sweeps_completed += 1
            if improved_this_sweep:
                sweeps_without_improvement = 0
            else:
                sweeps_without_improvement += 1

        # Final per-test evaluation of the incumbent
        per_q_scores = await self._evaluate_on_test(incumbent)
        wall_clock = time.perf_counter() - start

        return RunResult(
            method="coordinate_descent",
            seed=seed,
            champion_genome_id=incumbent.genome_id,
            champion_fitness=incumbent_fit,
            per_question_scores=per_q_scores,
            fitness_curve=fitness_curve,
            wall_clock_seconds=wall_clock,
            total_evaluations=self._eval_count,
            api_calls=self._call_count,
            metadata={
                "compute_budget": self.config.compute_budget,
                "evals_used": evals_used,
                "sweeps_completed": sweeps_completed,
                "max_alternatives_per_gene": _CD_MAX_ALTERNATIVES,
                "champion_genes": {
                    name: gene.value for name, gene in incumbent.genes.items()
                },
            },
        )


# ---------------------------------------------------------------------------
# TPE helpers (module-level so they can be unit-tested independently)
# ---------------------------------------------------------------------------

_TPE_WARMUP: int = 5
_TPE_N_CANDIDATES: int = 5
_TPE_SPLIT_FRACTION: float = 0.5  # median split of history
_TPE_EPS: float = 1e-12
_TPE_BW_FLOOR: float = 1e-2

_SH_ETA: int = 3
_SH_FIDELITY_MIN: int = 5


def _split_history_mask(
    scores: list[float], fraction: float
) -> tuple[list[bool], list[bool]]:
    """Return (good_mask, bad_mask) splitting *scores* at the top *fraction*.

    Higher is better. Ties break toward "bad".
    """
    if not scores:
        return [], []
    n = len(scores)
    k = max(1, int(round(n * fraction)))
    k = min(k, n - 1) if n > 1 else 1
    # sort indices by score descending
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    good_set = set(ranked[:k])
    good_mask = [i in good_set for i in range(n)]
    bad_mask = [not m for m in good_mask]
    return good_mask, bad_mask


class _TPEEncoder:
    """Encode/score genomes for TPE acquisition.

    The encoder builds a per-gene schema once (from the gene template)
    and then:

    * ``encode(genome)`` returns a dict mapping gene name to a numeric
      representation suitable for KDE / categorical scoring.
    * ``tpe_acquisition(encoded, good_hist, bad_hist, rng)`` returns a
      scalar Expected-Improvement-style score.

    Gene handling
    -------------
    * FLOAT    -> normalised value in [0, 1]
    * ENUM     -> category index (int)
    * SET      -> multi-hot vector (numpy array of 0/1)
    * TEXT     -> category index over the provided templates (if any)
    * NUMERIC_VECTOR -> category index over the provided templates
    """

    def __init__(self, gene_template: dict[str, dict[str, Any]]) -> None:
        self._schema: dict[str, dict[str, Any]] = {}
        for gene_name, spec in gene_template.items():
            raw_type = spec.get("type")
            if isinstance(raw_type, GeneType):
                gtype = raw_type
            else:
                try:
                    gtype = GeneType(str(raw_type).lower())
                except ValueError:
                    continue

            if gtype == GeneType.FLOAT:
                lo, hi = spec["range"]
                self._schema[gene_name] = {
                    "kind": "float",
                    "lo": float(lo),
                    "hi": float(hi),
                }
            elif gtype == GeneType.ENUM:
                self._schema[gene_name] = {
                    "kind": "enum",
                    "options": list(spec["options"]),
                }
            elif gtype == GeneType.SET:
                self._schema[gene_name] = {
                    "kind": "set",
                    "available": list(spec["available"]),
                }
            elif gtype == GeneType.TEXT:
                templates = list(spec.get("templates", []))
                self._schema[gene_name] = {
                    "kind": "text",
                    "templates": templates,
                }
            elif gtype == GeneType.NUMERIC_VECTOR:
                templates = list(spec.get("templates", []))
                self._schema[gene_name] = {
                    "kind": "numeric_vector",
                    "templates": templates,
                }

    def encode(self, genome: Genome) -> dict[str, Any]:
        """Encode a genome into per-gene numeric buckets."""
        out: dict[str, Any] = {}
        for gene_name, schema in self._schema.items():
            gene = genome.genes.get(gene_name)
            if gene is None:
                out[gene_name] = None
                continue

            kind = schema["kind"]
            if kind == "float":
                lo = schema["lo"]
                hi = schema["hi"]
                span = hi - lo if hi > lo else 1.0
                out[gene_name] = float((float(gene.value) - lo) / span)
            elif kind == "enum":
                options = schema["options"]
                try:
                    out[gene_name] = options.index(gene.value)
                except ValueError:
                    out[gene_name] = 0
            elif kind == "set":
                available = schema["available"]
                value_set = set(gene.value or [])
                out[gene_name] = np.array(
                    [1.0 if item in value_set else 0.0 for item in available],
                    dtype=np.float64,
                )
            elif kind == "text":
                templates = schema["templates"]
                try:
                    out[gene_name] = templates.index(gene.value)
                except ValueError:
                    out[gene_name] = 0
            elif kind == "numeric_vector":
                templates = schema["templates"]
                idx = 0
                for i, tmpl in enumerate(templates):
                    if list(tmpl) == list(gene.value):
                        idx = i
                        break
                out[gene_name] = idx
        return out

    def tpe_acquisition(
        self,
        encoded: dict[str, Any],
        good_hist: list[dict[str, Any]],
        bad_hist: list[dict[str, Any]],
        rng: np.random.Generator,
    ) -> float:
        """Log-ratio acquisition score ``log(l(x)+eps) - log(g(x)+eps)``.

        The joint density factorises independently across genes.
        """
        if not good_hist or not bad_hist:
            return float(rng.random())

        log_l = 0.0
        log_g = 0.0
        for gene_name, schema in self._schema.items():
            val = encoded.get(gene_name)
            if val is None:
                continue

            kind = schema["kind"]
            if kind == "float":
                good_vals = [
                    h[gene_name]
                    for h in good_hist
                    if h.get(gene_name) is not None
                ]
                bad_vals = [
                    h[gene_name]
                    for h in bad_hist
                    if h.get(gene_name) is not None
                ]
                log_l += math.log(
                    _gaussian_kde_pdf(val, good_vals) + _TPE_EPS
                )
                log_g += math.log(
                    _gaussian_kde_pdf(val, bad_vals) + _TPE_EPS
                )
            elif kind in ("enum", "text", "numeric_vector"):
                if kind == "enum":
                    n_cats = max(1, len(schema["options"]))
                elif kind == "text":
                    n_cats = max(1, len(schema["templates"]))
                else:
                    n_cats = max(1, len(schema["templates"]))
                good_cats = [
                    h[gene_name]
                    for h in good_hist
                    if h.get(gene_name) is not None
                ]
                bad_cats = [
                    h[gene_name]
                    for h in bad_hist
                    if h.get(gene_name) is not None
                ]
                log_l += math.log(
                    _categorical_pmf(val, good_cats, n_cats) + _TPE_EPS
                )
                log_g += math.log(
                    _categorical_pmf(val, bad_cats, n_cats) + _TPE_EPS
                )
            elif kind == "set":
                # Independent-Bernoulli per slot.
                good_vecs = [
                    h[gene_name]
                    for h in good_hist
                    if h.get(gene_name) is not None
                ]
                bad_vecs = [
                    h[gene_name]
                    for h in bad_hist
                    if h.get(gene_name) is not None
                ]
                if good_vecs and bad_vecs:
                    good_arr = np.stack(good_vecs)
                    bad_arr = np.stack(bad_vecs)
                    good_p = (good_arr.sum(axis=0) + 1.0) / (
                        len(good_vecs) + 2.0
                    )
                    bad_p = (bad_arr.sum(axis=0) + 1.0) / (
                        len(bad_vecs) + 2.0
                    )
                    x = np.asarray(val, dtype=np.float64)
                    good_like = x * good_p + (1.0 - x) * (1.0 - good_p)
                    bad_like = x * bad_p + (1.0 - x) * (1.0 - bad_p)
                    log_l += float(
                        np.sum(np.log(good_like + _TPE_EPS))
                    )
                    log_g += float(
                        np.sum(np.log(bad_like + _TPE_EPS))
                    )

        return log_l - log_g


def _gaussian_kde_pdf(x: float, samples: list[float]) -> float:
    """Scalar Gaussian KDE density at *x*.

    Uses Silverman's rule for bandwidth with a small floor so degenerate
    histories (all-identical values) don't produce zero density.
    """
    if not samples:
        return 0.0
    n = len(samples)
    arr = np.asarray(samples, dtype=np.float64)
    std = float(arr.std(ddof=0))
    bw = max(_TPE_BW_FLOOR, 1.06 * std * (n ** (-1 / 5)))
    z = (x - arr) / bw
    kernels = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return float(kernels.mean() / bw)


def _categorical_pmf(
    value: int, samples: list[int], n_categories: int
) -> float:
    """Categorical PMF with Laplace smoothing."""
    if n_categories <= 0:
        return 0.0
    counts = 0
    for s in samples:
        if s == value:
            counts += 1
    return (counts + 1.0) / (len(samples) + float(n_categories))


# ---------------------------------------------------------------------------
# Helpers used by clone-based refill (kept module-level so they can be
# monkey-patched in tests if necessary).
# ---------------------------------------------------------------------------


def _new_genome_id() -> str:
    import uuid
    return uuid.uuid4().hex


__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    "KNOWN_METHODS",
    "RunResult",
]
