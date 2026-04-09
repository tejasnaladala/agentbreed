"""Abstract base class for all adapters and the AgentResult data class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from breed.genome import Genome


@dataclass(frozen=True)
class AgentResult:
    """Immutable result from running an agent on a task.

    Attributes:
        output: The text output produced by the agent.
        metadata: Arbitrary extra data (e.g. model id, stop reason).
        cost_tokens: Total tokens consumed (prompt + completion).
        duration_ms: Wall-clock execution time in milliseconds.
        error: Error description if the run failed, otherwise None.
    """

    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    cost_tokens: int = 0
    duration_ms: int = 0
    error: str | None = None

    @property
    def success(self) -> bool:
        """True when the run completed without error."""
        return self.error is None


class Adapter(ABC):
    """Translates a Genome into a runnable agent and executes it on a task."""

    @abstractmethod
    async def run(self, genome: Genome, task: str) -> AgentResult:
        """Run the agent defined by *genome* on the given *task* string.

        Args:
            genome: The genome encoding the agent's behavioral parameters.
            task: The task prompt / question for the agent to handle.

        Returns:
            An :class:`AgentResult` with the agent's output (or error).
        """
        ...

    def build_system_prompt(self, genome: Genome) -> str:
        """Construct a system prompt from the genome's text genes.

        Combines ``prompt_template``, ``decision_heuristic``, and
        ``calibration_rule`` into a single multi-paragraph prompt.
        """
        parts: list[str] = []
        for gene_name in ("prompt_template", "decision_heuristic", "calibration_rule"):
            gene = genome.genes.get(gene_name)
            if gene is not None:
                parts.append(str(gene.value))
        return "\n\n".join(parts)
