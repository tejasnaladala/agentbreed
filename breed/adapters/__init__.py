"""Adapters translate genomes into framework-specific agent configs."""

from breed.adapters.base import Adapter, AgentResult
from breed.adapters.callable_adapter import CallableAdapter

__all__ = ["Adapter", "AgentResult", "CallableAdapter"]
