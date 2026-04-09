"""Top-level convenience API for breed.

This module provides ``evolve()`` and ``evolve_sync()`` -- the simplest
way to add evolutionary optimization to any agent system.

Usage::

    from breed import evolve

    # Define your agent function (takes config dict + task string, returns result string)
    async def my_agent(config: dict, task: str) -> str:
        # ... run your agent / swarm / pipeline ...
        return result

    # Define your gene template (what's evolvable about your agents)
    genes = {
        "system_prompt": {
            "type": "text",
            "templates": ["You are helpful.", "You are precise.", "You are creative."],
            "mutation_rate": 0.2,
        },
        "temperature": {
            "type": "float",
            "range": (0.0, 1.5),
            "mutation_rate": 0.1,
        },
    }

    # Evolve
    champion = await evolve(
        agent=my_agent,
        tasks=["Summarize this", "Analyze that"],
        scorer=lambda output, task: 1.0 if "good" in output else 0.0,
        genes=genes,
        generations=10,
        population_size=12,
    )

    # Use the champion config
    print(champion.to_dict())
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Sequence

from breed.adapters.callable_adapter import CallableAdapter
from breed.arenas.base import Task
from breed.arenas.custom import CustomArena
from breed.engine import BreedingConfig, BreedingEngine
from breed.events import CallbackObserver, EventBus, GenerationComplete
from breed.genome import Genome


async def evolve(
    agent: Callable[[dict[str, Any], str], Awaitable[str]],
    tasks: Sequence[str] | Sequence[Task],
    scorer: Callable[[str, Any], float],
    genes: dict[str, dict[str, Any]] | None = None,
    generations: int = 10,
    population_size: int = 20,
    elite_count: int = 4,
    mutation_rate: float = 0.15,
    seed: int | None = None,
    on_generation: Callable[[int, float, float], None] | None = None,
    lineage_file: str | None = None,
    results_dir: str = "breed_results",
    verbose: bool = False,
) -> Genome:
    """Evolve an agent function and return the champion genome.

    This is the primary entry point for breed. Provide your agent function,
    evaluation tasks, a scoring function, and optionally a custom gene
    template. breed handles everything else: population management,
    selection, crossover, mutation, lineage tracking.

    Parameters
    ----------
    agent:
        An async callable ``(genome_dict, task_str) -> result_str``.
        ``genome_dict`` is the serialized genome (use ``genome_dict["genes"]``
        to access gene values).  ``task_str`` is the evaluation task.
    tasks:
        Evaluation tasks.  Can be plain strings or :class:`Task` objects.
        Every genome in the population is evaluated on all tasks each generation.
    scorer:
        A callable ``(agent_output, expected) -> float`` returning a score
        in [0, 1] where higher is better.  ``expected`` comes from
        ``Task.expected`` (``None`` if tasks are plain strings).
    genes:
        Custom gene template defining what's evolvable.  Keys are gene names,
        values are dicts with ``type``, ``mutation_rate``, and type-specific
        fields.  If ``None``, uses the default 10-gene forecasting template.

        Example::

            {
                "system_prompt": {"type": "text", "templates": [...], "mutation_rate": 0.2},
                "temperature": {"type": "float", "range": (0.0, 1.5), "mutation_rate": 0.1},
                "model": {"type": "enum", "options": ["gpt-4o", "claude-sonnet-4-6"], "mutation_rate": 0.05},
                "tools": {"type": "set", "available": ["search", "code", "browse"], "mutation_rate": 0.1},
            }

    generations:
        Number of evolutionary generations.
    population_size:
        Number of agents per generation.
    elite_count:
        Number of top agents that survive to the next generation unchanged.
    mutation_rate:
        Probability of mutation per genome per generation.
    seed:
        RNG seed for reproducibility.
    on_generation:
        Optional callback ``(generation, best_fitness, avg_fitness)`` called
        after each generation completes.
    lineage_file:
        Path to JSONL lineage log.  Defaults to ``breed_results/lineage.jsonl``.
    results_dir:
        Directory for results.
    verbose:
        If True, print generation summaries to stdout.

    Returns
    -------
    Genome
        The champion genome -- the highest-fitness genome seen across all
        generations.  Use ``champion.to_dict()["genes"]`` to extract the
        winning configuration.
    """
    # Build tasks
    task_objects: list[Task] = []
    for i, t in enumerate(tasks):
        if isinstance(t, Task):
            task_objects.append(t)
        else:
            task_objects.append(Task(task_id=str(i), prompt=str(t)))

    # Build components
    adapter = CallableAdapter(agent)
    arena = CustomArena(tasks=task_objects, scorer=scorer)

    config = BreedingConfig(
        population_size=population_size,
        elite_count=elite_count,
        generations=generations,
        mutation_rate=mutation_rate,
        seed=seed,
        lineage_file=lineage_file or "lineage.jsonl",
        results_dir=results_dir,
        num_tasks=len(task_objects),
    )

    # Set up event bus
    event_bus = EventBus()

    if on_generation is not None or verbose:
        async def _on_gen_event(event: Any) -> None:
            if not isinstance(event, GenerationComplete):
                return
            if on_generation is not None:
                on_generation(event.generation, event.best_fitness, event.avg_fitness)
            if verbose:
                print(
                    f"  Gen {event.generation:3d}  |  "
                    f"best {event.best_fitness:.4f}  "
                    f"avg {event.avg_fitness:.4f}  "
                    f"worst {event.worst_fitness:.4f}  "
                    f"pop {event.population_size}"
                )

        observer = CallbackObserver({"*": [_on_gen_event]})
        await event_bus.add_observer(observer)

    engine = BreedingEngine(
        config=config,
        arena=arena,
        adapter=adapter,
        event_bus=event_bus,
        gene_template=genes,
    )

    champion = await engine.run()
    return champion


def evolve_sync(
    agent: Callable[[dict[str, Any], str], Awaitable[str]],
    **kwargs: Any,
) -> Genome:
    """Synchronous wrapper around :func:`evolve`.

    Same parameters as ``evolve()``, but blocks until evolution is complete.
    Use this when you're not in an async context::

        from breed import evolve_sync
        champion = evolve_sync(my_agent, tasks=[...], scorer=my_scorer)
    """
    return asyncio.run(evolve(agent, **kwargs))
