"""Custom arena: user-provided tasks and scoring function.

Allows callers to supply their own tasks and a scorer callable that
takes (agent_output, expected) and returns a fitness float in [0, 1].
"""

from __future__ import annotations

from typing import Any, Callable

from breed.adapters.base import Adapter, AgentResult
from breed.arenas.base import Arena, EvalResult, Task
from breed.genome import Genome


class CustomArena(Arena):
    """Arena with user-provided tasks and scoring function.

    Attributes:
        tasks: The fixed set of tasks to evaluate on.
        scorer: A callable ``(agent_output: str, expected: Any) -> float``
                that returns a score in [0, 1].
    """

    def __init__(
        self,
        tasks: list[Task],
        scorer: Callable[[str, Any], float],
    ) -> None:
        """Initialise with tasks and a scorer.

        Args:
            tasks: Pre-built Task objects to evaluate on.
            scorer: Takes (agent_output, expected) and returns a float in [0, 1].
        """
        self._tasks = list(tasks)
        self._scorer = scorer

    async def generate_tasks(self, count: int, seed: int | None = None) -> list[Task]:
        """Return the stored tasks (ignores count and seed).

        The custom arena uses a fixed task set, so this simply returns
        the tasks provided at construction time.

        Args:
            count: Ignored.
            seed: Ignored.

        Returns:
            The stored list of tasks.
        """
        return list(self._tasks)

    async def evaluate(
        self, genome: Genome, adapter: Adapter, tasks: list[Task]
    ) -> EvalResult:
        """Evaluate the genome using the custom scorer on each task.

        Args:
            genome: The genome defining the agent.
            adapter: Translates the genome into a runnable agent.
            tasks: The tasks to evaluate on.

        Returns:
            EvalResult with average score as fitness.
        """
        scores: list[float] = []
        agent_results: list[AgentResult] = []
        total_tokens = 0

        for task in tasks:
            result = await adapter.run(genome, task.prompt)
            agent_results.append(result)
            total_tokens += result.cost_tokens

            if result.success:
                score = self._scorer(result.output, task.expected)
                score = max(0.0, min(1.0, score))
            else:
                score = 0.0

            scores.append(score)

        fitness = sum(scores) / len(scores) if scores else 0.0

        return EvalResult(
            genome_id=genome.genome_id,
            fitness_score=fitness,
            fitness_breakdown={"custom_score": fitness},
            task_results=agent_results,
            cost_tokens=total_tokens,
        )
