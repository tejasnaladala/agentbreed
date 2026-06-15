"""Abstract model-client interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class ModelCallResult:
    """Per-call result shared across all clients."""
    text: str
    input_tokens: int
    output_tokens: int
    wall_time_ms: int
    model: str
    seed: Optional[int] = None
    temperature: float = 0.0
    finish_reason: str = ""
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.text != ""


class ModelClient(abc.ABC):
    """Abstract async interface for an LLM client."""

    model_name: str = ""

    @abc.abstractmethod
    async def generate(
        self,
        *,
        system: Optional[str],
        user: str,
        max_tokens: int,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> ModelCallResult:
        """Issue one generation call and return a ModelCallResult."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Clean up any client state (connections, pools)."""
