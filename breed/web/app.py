"""FastAPI application for the Breeding Pit web UI.

Serves the static HTML/JS/CSS dashboard and provides WebSocket + REST
endpoints for real-time breeding event streaming and lineage exploration.

Run standalone::

    uvicorn breed.web.app:app --port 8420
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:
    raise ImportError(
        "FastAPI is required for the Breeding Pit web UI. "
        "Install it with:  pip install 'fastapi[standard]' uvicorn"
    ) from exc

from breed.events import BreedingEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_WEB_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _WEB_DIR / "static"

# ---------------------------------------------------------------------------
# Shared state (module-level singletons)
# ---------------------------------------------------------------------------


@dataclass
class EngineState:
    """Mutable snapshot of the breeding engine state exposed via REST."""

    status: str = "idle"
    generation: int = 0
    population_size: int = 0
    champion_id: str | None = None
    champion_fitness: float = 0.0
    total_generations: int = 0
    arena_type: str = ""
    paused: bool = False


_engine_state = EngineState()

# Store generation history for the fitness chart endpoint
_generation_history: list[dict[str, Any]] = []

# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Track active WebSocket connections and broadcast messages."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)
        logger.info("WebSocket client connected (%d total)", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            try:
                self._connections.remove(ws)
            except ValueError:
                pass
        logger.info("WebSocket client disconnected (%d total)", len(self._connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send a JSON message to every connected client."""
        payload = json.dumps(message, default=str)
        async with self._lock:
            stale: list[WebSocket] = []
            for ws in self._connections:
                try:
                    await ws.send_text(payload)
                except Exception:
                    stale.append(ws)
            for ws in stale:
                try:
                    self._connections.remove(ws)
                except ValueError:
                    pass

    @property
    def client_count(self) -> int:
        return len(self._connections)


_manager = ConnectionManager()

# ---------------------------------------------------------------------------
# WebObserver -- the real implementation replacing the stub in events.py
# ---------------------------------------------------------------------------


class WebObserver:
    """Push breeding events to all connected WebSocket clients.

    Also updates the shared ``EngineState`` so REST endpoints stay current.
    Implements the ``Observer`` protocol from ``breed.events``.
    """

    def __init__(
        self,
        manager: ConnectionManager | None = None,
        state: EngineState | None = None,
    ) -> None:
        self._manager = manager or _manager
        self._state = state or _engine_state

    async def on_event(self, event: BreedingEvent) -> None:
        """Handle an incoming breeding event."""
        payload = event.to_dict()

        # Update shared engine state based on event type
        self._update_state(payload)

        # Broadcast to all connected WebSocket clients
        await self._manager.broadcast(payload)

    def _update_state(self, payload: dict[str, Any]) -> None:
        et = payload.get("event_type", "")

        if et == "breeding_started":
            self._state.status = "running"
            self._state.total_generations = payload.get("total_generations", 0)
            self._state.population_size = payload.get("population_size", 0)
            self._state.arena_type = payload.get("arena_type", "")
            self._state.generation = 0

        elif et == "generation_complete":
            self._state.generation = payload.get("generation", 0)
            self._state.population_size = payload.get("population_size", 0)
            self._state.champion_id = payload.get("champion_id")
            self._state.champion_fitness = payload.get("best_fitness", 0.0)
            _generation_history.append({
                "generation": payload.get("generation", 0),
                "best_fitness": payload.get("best_fitness", 0.0),
                "avg_fitness": payload.get("avg_fitness", 0.0),
                "worst_fitness": payload.get("worst_fitness", 0.0),
                "population_size": payload.get("population_size", 0),
                "champion_id": payload.get("champion_id"),
            })

        elif et == "champion_changed":
            self._state.champion_id = payload.get("new_champion_id")
            self._state.champion_fitness = payload.get("fitness_score", 0.0)

        elif et == "breeding_complete":
            self._state.status = "complete"
            self._state.champion_id = payload.get("champion_id")
            self._state.champion_fitness = payload.get("champion_fitness", 0.0)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Breeding Pit",
    description="Real-time evolutionary agent breeding dashboard",
    version="0.1.0",
)

# Serve static files (index.html, etc.)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Optional: keep a reference to a LineageTracker if one is attached
_lineage_tracker: Any = None


def attach_lineage_tracker(tracker: Any) -> None:
    """Attach a ``LineageTracker`` instance so REST endpoints can query it."""
    global _lineage_tracker
    _lineage_tracker = tracker


def get_engine_state() -> EngineState:
    """Return the module-level engine state (for external mutation)."""
    return _engine_state


def get_connection_manager() -> ConnectionManager:
    """Return the module-level connection manager."""
    return _manager


# ---------------------------------------------------------------------------
# Routes -- Static
# ---------------------------------------------------------------------------


@app.get("/")
async def root() -> FileResponse:
    """Serve the Breeding Pit dashboard."""
    return FileResponse(str(_STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Routes -- WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Stream breeding events to the client in real time."""
    await _manager.connect(ws)
    try:
        while True:
            # Keep connection alive; client can also send commands
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                cmd = msg.get("command")
                if cmd:
                    await _handle_ws_command(cmd, msg)
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        await _manager.disconnect(ws)
    except Exception:
        await _manager.disconnect(ws)


async def _handle_ws_command(cmd: str, msg: dict[str, Any]) -> None:
    """Handle commands sent from the client over WebSocket."""
    if cmd == "pause":
        _engine_state.paused = True
        _engine_state.status = "paused"
    elif cmd == "resume":
        _engine_state.paused = False
        _engine_state.status = "running"


# ---------------------------------------------------------------------------
# Routes -- REST API
# ---------------------------------------------------------------------------


@app.get("/api/status")
async def api_status() -> JSONResponse:
    """Return current engine status."""
    return JSONResponse({
        "status": _engine_state.status,
        "generation": _engine_state.generation,
        "total_generations": _engine_state.total_generations,
        "population_size": _engine_state.population_size,
        "champion_id": _engine_state.champion_id,
        "champion_fitness": _engine_state.champion_fitness,
        "arena_type": _engine_state.arena_type,
        "paused": _engine_state.paused,
        "connected_clients": _manager.client_count,
        "generation_history": _generation_history,
    })


@app.get("/api/lineage")
async def api_lineage() -> JSONResponse:
    """Return the full lineage tree as JSON."""
    if _lineage_tracker is None:
        return JSONResponse(
            {"error": "No lineage tracker attached"},
            status_code=404,
        )
    tree = _lineage_tracker.get_tree()
    serialized = {
        gid: {
            "genome_id": node.genome_id,
            "generation": node.generation,
            "fitness_score": node.fitness_score,
            "parents": node.parents,
            "children": node.children,
        }
        for gid, node in tree.items()
    }
    return JSONResponse(serialized)


@app.get("/api/lineage/{genome_id}")
async def api_lineage_genome(genome_id: str) -> JSONResponse:
    """Return the ancestry of a specific genome."""
    if _lineage_tracker is None:
        return JSONResponse(
            {"error": "No lineage tracker attached"},
            status_code=404,
        )
    ancestry = _lineage_tracker.get_lineage(genome_id)
    return JSONResponse([r.to_dict() for r in ancestry])


@app.get("/api/generation/{gen}")
async def api_generation(gen: int) -> JSONResponse:
    """Return all lineage records for a generation."""
    if _lineage_tracker is None:
        return JSONResponse(
            {"error": "No lineage tracker attached"},
            status_code=404,
        )
    records = _lineage_tracker.load()
    gen_records = [r.to_dict() for r in records if r.generation == gen]
    return JSONResponse(gen_records)


@app.get("/api/champion")
async def api_champion() -> JSONResponse:
    """Return the current champion genome info."""
    if _engine_state.champion_id is None:
        return JSONResponse(
            {"error": "No champion yet"},
            status_code=404,
        )
    result: dict[str, Any] = {
        "champion_id": _engine_state.champion_id,
        "fitness_score": _engine_state.champion_fitness,
        "generation": _engine_state.generation,
    }
    # If lineage tracker is available, include full record
    if _lineage_tracker is not None:
        ancestry = _lineage_tracker.get_lineage(_engine_state.champion_id)
        if ancestry:
            result["record"] = ancestry[0].to_dict()
    return JSONResponse(result)


@app.post("/api/command")
async def api_command(body: dict[str, Any]) -> JSONResponse:
    """Send commands to the engine (pause, resume, etc.)."""
    cmd = body.get("command", "")
    if cmd == "pause":
        _engine_state.paused = True
        _engine_state.status = "paused"
        return JSONResponse({"ok": True, "status": "paused"})
    elif cmd == "resume":
        _engine_state.paused = False
        _engine_state.status = "running"
        return JSONResponse({"ok": True, "status": "running"})
    else:
        return JSONResponse(
            {"error": f"Unknown command: {cmd}"},
            status_code=400,
        )
