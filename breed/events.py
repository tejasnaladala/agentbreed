"""Event / observer system for the evolutionary breeding pipeline.

Provides frozen dataclass event types for every significant lifecycle moment,
an async Observer protocol, and concrete observer implementations for headless
(JSON lines), programmatic (callbacks), and composite fan-out usage.  Stubs
are included for Rich terminal and WebSocket observers.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Union, runtime_checkable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _class_to_snake(name: str) -> str:
    """Convert *PascalCase* class name to *snake_case*."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


# ---------------------------------------------------------------------------
# Event dataclasses  (all frozen / immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentBorn:
    """A new agent has been created."""

    genome_id: str
    generation: int
    parents: list[str]
    timestamp: str = field(default_factory=_utc_now_iso)

    @property
    def event_type(self) -> str:
        return _class_to_snake(type(self).__name__)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "event_type": self.event_type}


@dataclass(frozen=True)
class AgentDied:
    """An agent has been culled from the population."""

    genome_id: str
    generation: int
    fitness_score: float
    timestamp: str = field(default_factory=_utc_now_iso)

    @property
    def event_type(self) -> str:
        return _class_to_snake(type(self).__name__)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "event_type": self.event_type}


@dataclass(frozen=True)
class AgentEvaluated:
    """An agent has been scored by the fitness evaluator."""

    genome_id: str
    generation: int
    fitness_score: float
    fitness_breakdown: dict[str, float]
    timestamp: str = field(default_factory=_utc_now_iso)

    @property
    def event_type(self) -> str:
        return _class_to_snake(type(self).__name__)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "event_type": self.event_type}


@dataclass(frozen=True)
class CrossoverEvent:
    """A crossover operation produced a child genome."""

    parent_a_id: str
    parent_b_id: str
    child_id: str
    operator: str
    timestamp: str = field(default_factory=_utc_now_iso)

    @property
    def event_type(self) -> str:
        return _class_to_snake(type(self).__name__)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "event_type": self.event_type}


@dataclass(frozen=True)
class MutationEvent:
    """A mutation was applied to a genome."""

    genome_id: str
    operator: str
    gene_name: str | None
    timestamp: str = field(default_factory=_utc_now_iso)

    @property
    def event_type(self) -> str:
        return _class_to_snake(type(self).__name__)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "event_type": self.event_type}


@dataclass(frozen=True)
class GenerationComplete:
    """An entire generation has been evaluated."""

    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    population_size: int
    champion_id: str
    timestamp: str = field(default_factory=_utc_now_iso)

    @property
    def event_type(self) -> str:
        return _class_to_snake(type(self).__name__)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "event_type": self.event_type}


@dataclass(frozen=True)
class ChampionChanged:
    """The overall champion genome has changed."""

    old_champion_id: str | None
    new_champion_id: str
    generation: int
    fitness_score: float
    timestamp: str = field(default_factory=_utc_now_iso)

    @property
    def event_type(self) -> str:
        return _class_to_snake(type(self).__name__)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "event_type": self.event_type}


@dataclass(frozen=True)
class BreedingStarted:
    """A breeding run has begun."""

    total_generations: int
    population_size: int
    arena_type: str
    timestamp: str = field(default_factory=_utc_now_iso)

    @property
    def event_type(self) -> str:
        return _class_to_snake(type(self).__name__)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "event_type": self.event_type}


@dataclass(frozen=True)
class BreedingComplete:
    """A breeding run has finished."""

    final_generation: int
    champion_id: str
    champion_fitness: float
    total_evaluations: int
    timestamp: str = field(default_factory=_utc_now_iso)

    @property
    def event_type(self) -> str:
        return _class_to_snake(type(self).__name__)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "event_type": self.event_type}


@dataclass(frozen=True)
class ImmigrantsInjected:
    """Random immigrants were injected into the population."""

    count: int
    generation: int
    timestamp: str = field(default_factory=_utc_now_iso)

    @property
    def event_type(self) -> str:
        return _class_to_snake(type(self).__name__)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "event_type": self.event_type}


# ---------------------------------------------------------------------------
# Union type for all events
# ---------------------------------------------------------------------------

BreedingEvent = Union[
    AgentBorn,
    AgentDied,
    AgentEvaluated,
    CrossoverEvent,
    MutationEvent,
    GenerationComplete,
    ChampionChanged,
    BreedingStarted,
    BreedingComplete,
    ImmigrantsInjected,
]


# ---------------------------------------------------------------------------
# Observer protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Observer(Protocol):
    """Any object that can receive breeding events."""

    async def on_event(self, event: BreedingEvent) -> None: ...


# ---------------------------------------------------------------------------
# Concrete observers
# ---------------------------------------------------------------------------

class JSONObserver:
    """Appends each event as a JSON line to a file.  Ideal for headless / CI runs."""

    def __init__(self, filepath: str | Path) -> None:
        self._filepath = Path(filepath)

    async def on_event(self, event: BreedingEvent) -> None:
        line = json.dumps(event.to_dict(), default=str) + "\n"
        with self._filepath.open("a", encoding="utf-8") as fh:
            fh.write(line)


class CallbackObserver:
    """Dispatches events to registered async callbacks.

    Callbacks are keyed by *event_type* string (snake_case class name).
    The special key ``"*"`` matches every event type.
    """

    def __init__(
        self,
        callbacks: dict[str, list[Callable[..., Any]]] | None = None,
    ) -> None:
        self._callbacks: dict[str, list[Callable[..., Any]]] = dict(callbacks or {})

    def register(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Register a callback for *event_type* (or ``"*"`` for all)."""
        self._callbacks.setdefault(event_type, []).append(callback)

    async def on_event(self, event: BreedingEvent) -> None:
        et = event.event_type
        targets = list(self._callbacks.get(et, []))
        targets.extend(self._callbacks.get("*", []))
        for cb in targets:
            result = cb(event)
            if asyncio.iscoroutine(result):
                await result


class CompositeObserver:
    """Fans out every event to a list of child observers."""

    def __init__(self, observers: list[Observer] | None = None) -> None:
        self._observers: list[Observer] = list(observers or [])

    def add(self, observer: Observer) -> None:
        """Append an observer to the fan-out list."""
        self._observers.append(observer)

    async def on_event(self, event: BreedingEvent) -> None:
        for observer in self._observers:
            await observer.on_event(event)


class WebObserver:
    """Push events to the Breeding Pit web UI via WebSocket.

    TODO: Implement WebSocket transport using FastAPI / uvicorn integration.
    """

    pass  # noqa: WPS420


class RichObserver:
    """Render live breeding progress in a Rich terminal dashboard.

    TODO: Implement Rich Live display with generation table and fitness sparklines.
    """

    pass  # noqa: WPS420


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class EventBus:
    """Thread-safe async event bus for broadcasting events to observers.

    Uses an :class:`asyncio.Lock` to ensure safe concurrent add/remove/emit.
    """

    def __init__(self) -> None:
        self._observers: list[Observer] = []
        self._lock = asyncio.Lock()

    async def add_observer(self, observer: Observer) -> None:
        """Register *observer* to receive future events."""
        async with self._lock:
            self._observers.append(observer)

    async def remove_observer(self, observer: Observer) -> None:
        """Un-register *observer*.  Silently ignores observers not present."""
        async with self._lock:
            try:
                self._observers.remove(observer)
            except ValueError:
                pass

    async def emit(self, event: BreedingEvent) -> None:
        """Broadcast *event* to every registered observer."""
        async with self._lock:
            snapshot = list(self._observers)
        for observer in snapshot:
            await observer.on_event(event)
