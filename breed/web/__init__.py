"""Web UI for lineage exploration and the Breeding Pit dashboard.

Quick start::

    uvicorn breed.web.app:app --port 8420
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from breed.web.app import ConnectionManager, EngineState, WebObserver

__all__ = [
    "get_app",
    "get_web_observer",
]


def get_app():
    """Return the FastAPI application instance (lazy import)."""
    from breed.web.app import app

    return app


def get_web_observer():
    """Create and return a new :class:`WebObserver` wired to the global state."""
    from breed.web.app import WebObserver

    return WebObserver()
