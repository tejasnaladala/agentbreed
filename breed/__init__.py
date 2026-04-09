"""breed -- Evolve any AI agent through Darwinian selection.

Quick start::

    from breed import evolve

    champion = await evolve(
        agent=my_agent_fn,
        tasks=["task 1", "task 2"],
        scorer=lambda output, task: score(output),
        genes=my_gene_template,
        generations=10,
    )
"""

__version__ = "0.1.0"

from breed.api import evolve, evolve_sync
from breed.genome import Gene, GeneType, Genome, create_genome_from_template, create_random_genome
from breed.engine import BreedingEngine, BreedingConfig
from breed.population import Population
from breed.adapters.base import Adapter, AgentResult
from breed.adapters.callable_adapter import CallableAdapter
from breed.arenas.base import Arena, Task, EvalResult
from breed.arenas.custom import CustomArena
from breed.arenas.forecasting import ForecastingArena
from breed.arenas.coding import CodingArena
from breed.events import EventBus, CallbackObserver, JSONObserver
from breed.lineage import LineageTracker

__all__ = [
    # Top-level API
    "evolve",
    "evolve_sync",
    # Core
    "Genome",
    "Gene",
    "GeneType",
    "Population",
    "BreedingEngine",
    "BreedingConfig",
    # Adapters
    "Adapter",
    "AgentResult",
    "CallableAdapter",
    # Arenas
    "Arena",
    "Task",
    "EvalResult",
    "CustomArena",
    "ForecastingArena",
    "CodingArena",
    # Events
    "EventBus",
    "CallbackObserver",
    "JSONObserver",
    # Lineage
    "LineageTracker",
    # Factories
    "create_random_genome",
    "create_genome_from_template",
]
