"""Evaluation arenas for agent fitness testing."""

from breed.arenas.base import Arena, EvalResult, Task
from breed.arenas.coding import CodingArena
from breed.arenas.custom import CustomArena
from breed.arenas.forecasting import ForecastingArena

__all__ = [
    "Arena",
    "CodingArena",
    "CustomArena",
    "EvalResult",
    "ForecastingArena",
    "Task",
]
