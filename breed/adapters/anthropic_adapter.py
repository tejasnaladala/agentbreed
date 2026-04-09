"""Adapter for the Anthropic Messages API (optional dependency)."""

from __future__ import annotations

import time
from typing import Any

from breed.adapters.base import Adapter, AgentResult
from breed.genome import Genome

# Guard the import so the module is always importable even without the SDK.
_anthropic_available = True
_anthropic_import_error: str | None = None

try:
    import anthropic  # type: ignore[import-untyped]
except ImportError:
    _anthropic_available = False
    _anthropic_import_error = (
        "The 'anthropic' package is required for AnthropicMessagesAdapter. "
        "Install it with: pip install anthropic"
    )


class AnthropicMessagesAdapter(Adapter):
    """Adapter using the Anthropic Messages API.

    Requires the ``anthropic`` package.  If not installed, the class can
    still be imported but instantiation raises :class:`ImportError`.

    Args:
        model: Anthropic model identifier (e.g. ``"claude-sonnet-4-6"``).
        api_key: Optional API key.  Falls back to the ``ANTHROPIC_API_KEY``
            environment variable when *None*.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
    ) -> None:
        if not _anthropic_available:
            raise ImportError(_anthropic_import_error)

        self._model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def run(self, genome: Genome, task: str) -> AgentResult:
        """Send the task to the Anthropic Messages API.

        The system prompt is built from the genome's text genes.  The
        ``temperature`` gene controls sampling temperature.  The
        ``tool_policy`` gene is forwarded as tool name stubs (no actual
        tool execution occurs -- this merely signals to the model which
        tools are conceptually available).
        """
        system_prompt = self.build_system_prompt(genome)

        # Temperature from genome (default 0.7 if gene absent).
        temperature_gene = genome.genes.get("temperature")
        temperature = float(temperature_gene.value) if temperature_gene else 0.7

        # Tool stubs from the tool_policy SET gene.
        tools: list[dict[str, Any]] = []
        tool_gene = genome.genes.get("tool_policy")
        if tool_gene is not None and isinstance(tool_gene.value, list):
            for tool_name in tool_gene.value:
                tools.append(
                    {
                        "name": tool_name,
                        "description": f"Tool: {tool_name}",
                        "input_schema": {
                            "type": "object",
                            "properties": {},
                        },
                    }
                )

        start_ns = time.monotonic_ns()
        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": 4096,
                "system": system_prompt,
                "temperature": temperature,
                "messages": [{"role": "user", "content": task}],
            }
            if tools:
                kwargs["tools"] = tools

            response = await self._client.messages.create(**kwargs)
            duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000

            # Extract text blocks from the response.
            text_parts = [
                block.text
                for block in response.content
                if hasattr(block, "text")
            ]
            output = "\n".join(text_parts)

            cost_tokens = (
                response.usage.input_tokens + response.usage.output_tokens
            )

            return AgentResult(
                output=output,
                metadata={
                    "model": response.model,
                    "stop_reason": response.stop_reason,
                },
                cost_tokens=cost_tokens,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000
            return AgentResult(
                output="",
                duration_ms=duration_ms,
                error=f"{type(exc).__name__}: {exc}",
            )
