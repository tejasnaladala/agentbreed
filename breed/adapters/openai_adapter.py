"""Adapter for the OpenAI Chat Completions API (optional dependency)."""

from __future__ import annotations

import time

from breed.adapters.base import Adapter, AgentResult
from breed.genome import Genome

# Guard the import so the module is always importable even without the SDK.
_openai_available = True
_openai_import_error: str | None = None

try:
    import openai  # type: ignore[import-untyped]
except ImportError:
    _openai_available = False
    _openai_import_error = (
        "The 'openai' package is required for OpenAIAdapter. "
        "Install it with: pip install openai"
    )


class OpenAIAdapter(Adapter):
    """Adapter using the OpenAI Chat Completions API.

    Requires the ``openai`` package.  If not installed, the class can
    still be imported but instantiation raises :class:`ImportError`.

    Args:
        model: OpenAI model identifier (e.g. ``"gpt-4o"``).
        api_key: Optional API key.  Falls back to the ``OPENAI_API_KEY``
            environment variable when *None*.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
    ) -> None:
        if not _openai_available:
            raise ImportError(_openai_import_error)

        self._model = model
        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def run(self, genome: Genome, task: str) -> AgentResult:
        """Send the task to the OpenAI Chat Completions API.

        The system prompt is built from the genome's text genes.  The
        ``temperature`` gene controls sampling temperature.
        """
        system_prompt = self.build_system_prompt(genome)

        # Temperature from genome (default 0.7 if gene absent).
        temperature_gene = genome.genes.get("temperature")
        temperature = float(temperature_gene.value) if temperature_gene else 0.7

        start_ns = time.monotonic_ns()
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task},
                ],
            )
            duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000

            choice = response.choices[0]
            output = choice.message.content or ""

            cost_tokens = 0
            if response.usage is not None:
                cost_tokens = (
                    response.usage.prompt_tokens
                    + response.usage.completion_tokens
                )

            return AgentResult(
                output=output,
                metadata={
                    "model": response.model,
                    "finish_reason": choice.finish_reason,
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
