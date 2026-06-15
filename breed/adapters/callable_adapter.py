"""Escape-hatch adapter: wraps any user-provided async callable."""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

from breed.adapters.base import Adapter, AgentResult
from breed.genome import Genome


class CallableAdapter(Adapter):
    """Adapter wrapping any async callable: ``(genome_dict, task) -> result_str``.

    This is the simplest way to plug a custom agent into the breeding loop.
    The callable receives the genome serialised as a plain dict (via
    :meth:`Genome.to_dict`) and the raw task string, and must return a
    string result.

    Example::

        async def my_agent(genome_dict: dict, task: str) -> str:
            return f"Answer based on {genome_dict['genes']['temperature']}"

        adapter = CallableAdapter(my_agent)
        result = await adapter.run(genome, "What is 2+2?")
    """

    def __init__(
        self, fn: Callable[[dict[str, Any], str], Awaitable[str]]
    ) -> None:
        self._fn = fn

    async def run(self, genome: Genome, task: str) -> AgentResult:
        """Invoke the wrapped callable and return an :class:`AgentResult`.

        Exceptions raised by the callable are caught and surfaced via
        :attr:`AgentResult.error` rather than propagated.
        """
        start_ns = time.monotonic_ns()
        try:
            output = await self._fn(genome.to_dict(), task)
            duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000
            return AgentResult(output=output, duration_ms=duration_ms)
        except Exception as exc:
            duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000
            return AgentResult(
                output="",
                duration_ms=duration_ms,
                error=f"{type(exc).__name__}: {exc}",
            )
