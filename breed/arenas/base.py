"""Base arena interface for agent evaluation domains."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from breed.adapters.base import Adapter, AgentResult
from breed.genome import Genome


@dataclass(frozen=True)
class Task:
    """A single evaluation task.

    Attributes:
        task_id: Unique identifier for the task.
        prompt: The question or instruction for the agent.
        expected: Ground truth answer (if available).
        metadata: Arbitrary extra data attached to the task.
    """

    task_id: str
    prompt: str
    expected: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalResult:
    """Result of evaluating one genome across all tasks.

    Attributes:
        genome_id: Which genome was evaluated.
        fitness_score: Overall fitness in [0, 1], higher is better.
        fitness_breakdown: Per-metric breakdown of the fitness score.
        task_results: Raw agent outputs for each task.
        cost_tokens: Total tokens consumed across all tasks.
    """

    genome_id: str
    fitness_score: float
    fitness_breakdown: dict[str, float] = field(default_factory=dict)
    task_results: list[AgentResult] = field(default_factory=list)
    cost_tokens: int = 0


class Arena(ABC):
    """Base class for evaluation arenas.

    An arena defines a domain of evaluation: it generates tasks, runs agents
    on those tasks via an adapter, and computes fitness scores.
    """

    @abstractmethod
    async def generate_tasks(self, count: int, seed: int | None = None) -> list[Task]:
        """Generate evaluation tasks.

        Args:
            count: Number of tasks to generate.
            seed: Optional RNG seed for reproducibility.

        Returns:
            A list of Task objects.
        """
        ...

    @abstractmethod
    async def evaluate(
        self, genome: Genome, adapter: Adapter, tasks: list[Task]
    ) -> EvalResult:
        """Evaluate a genome on the given tasks using the adapter.

        Args:
            genome: The genome encoding the agent's behaviour.
            adapter: Translates the genome into a runnable agent.
            tasks: The tasks to evaluate on.

        Returns:
            An EvalResult with fitness score and per-task outputs.
        """
        ...

    async def evaluate_population(
        self,
        genomes: Sequence[Genome],
        adapter: Adapter,
        tasks: list[Task],
    ) -> list[EvalResult]:
        """Evaluate all genomes on the same tasks. Default: sequential.

        Args:
            genomes: The population of genomes to evaluate.
            adapter: Translates each genome into a runnable agent.
            tasks: The shared task set.

        Returns:
            A list of EvalResult, one per genome.
        """
        results: list[EvalResult] = []
        for genome in genomes:
            result = await self.evaluate(genome, adapter, tasks)
            results.append(result)
        return results
