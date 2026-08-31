"""Security regression tests for the disabled CodingArena executor."""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest
from unittest.mock import AsyncMock

from breed.adapters.base import Adapter, AgentResult
from breed.arenas import coding
from breed.genome import Genome


_DISABLED_MESSAGE = r"CodingArena is disabled.*OS-level sandbox"


def _assert_submission_rejected(code: str) -> None:
    with pytest.raises(coding.CodingArenaDisabledError, match=_DISABLED_MESSAGE):
        coding._run_test(code, "candidate", [], True)


async def test_evaluate_fails_before_requesting_generated_code() -> None:
    arena = coding.CodingArena()
    tasks = await arena.generate_tasks(count=1, seed=0)
    adapter = AsyncMock(spec=Adapter)
    adapter.run.return_value = AgentResult(output="def candidate(): return True")

    with pytest.raises(coding.CodingArenaDisabledError, match=_DISABLED_MESSAGE):
        await arena.evaluate(Genome(genome_id="security-test"), adapter, tasks)

    adapter.run.assert_not_awaited()


def test_submission_cannot_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def tracked_getenv(key: str, default: str | None = None) -> str | None:
        calls.append(key)
        return default

    monkeypatch.setattr(os, "getenv", tracked_getenv)
    _assert_submission_rejected(
        "import os\nos.getenv('AGENTBREED_SECRET')\ndef candidate(): return True"
    )

    assert calls == []


def test_submission_cannot_read_files(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def tracked_read_text(path: Path, *args: object, **kwargs: object) -> str:
        calls.append(path)
        return "private"

    monkeypatch.setattr(Path, "read_text", tracked_read_text)
    _assert_submission_rejected(
        "from pathlib import Path\n"
        "Path('private.txt').read_text()\n"
        "def candidate(): return True"
    )

    assert calls == []


def test_submission_cannot_spawn_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    def tracked_run(*args: object, **kwargs: object) -> object:
        calls.append(args)
        return object()

    monkeypatch.setattr(subprocess, "run", tracked_run)
    _assert_submission_rejected(
        "import subprocess\n"
        "subprocess.run(['python', '-c', 'pass'])\n"
        "def candidate(): return True"
    )

    assert calls == []


def test_submission_cannot_access_network(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    def tracked_connection(*args: object, **kwargs: object) -> object:
        calls.append(args)
        return object()

    monkeypatch.setattr(socket, "create_connection", tracked_connection)
    _assert_submission_rejected(
        "import socket\n"
        "socket.create_connection(('127.0.0.1', 9))\n"
        "def candidate(): return True"
    )

    assert calls == []


def test_nonterminating_submission_is_never_started() -> None:
    payload = "while True:\n    pass\n\ndef candidate():\n    return True"
    script = (
        "from breed.arenas.coding import _run_test\n"
        f"_run_test({payload!r}, 'candidate', [], True)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )

    assert completed.returncode != 0
    assert "CodingArenaDisabledError" in completed.stderr
