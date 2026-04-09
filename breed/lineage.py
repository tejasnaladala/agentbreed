"""Immutable lineage tracker using append-only JSONL.

Every evaluated genome gets a single-line JSON record appended to the log
file.  The tracker never overwrites or modifies existing records -- it is
designed for crash-safe, auditable evolutionary history.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from breed.genome import Genome


# ---------------------------------------------------------------------------
# Single lineage record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LineageRecord:
    """One snapshot of a genome at evaluation time."""

    genome_id: str
    generation: int
    parents: list[str]
    fitness_score: float
    fitness_breakdown: dict[str, float]
    survived: bool
    crossover_operator: str | None
    mutations_applied: list[str]
    genome_snapshot: dict[str, Any]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "generation": self.generation,
            "parents": list(self.parents),
            "fitness_score": self.fitness_score,
            "fitness_breakdown": dict(self.fitness_breakdown),
            "survived": self.survived,
            "crossover_operator": self.crossover_operator,
            "mutations_applied": list(self.mutations_applied),
            "genome_snapshot": dict(self.genome_snapshot),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageRecord:
        return cls(
            genome_id=data["genome_id"],
            generation=data["generation"],
            parents=data.get("parents", []),
            fitness_score=data["fitness_score"],
            fitness_breakdown=data.get("fitness_breakdown", {}),
            survived=data.get("survived", False),
            crossover_operator=data.get("crossover_operator"),
            mutations_applied=data.get("mutations_applied", []),
            genome_snapshot=data.get("genome_snapshot", {}),
            timestamp=data.get("timestamp", ""),
        )


# ---------------------------------------------------------------------------
# Lineage tree node (for visualization)
# ---------------------------------------------------------------------------

@dataclass
class LineageNode:
    """A node in the lineage tree."""

    genome_id: str
    generation: int
    fitness_score: float
    parents: list[str]
    children: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class LineageTracker:
    """Append-only JSONL lineage log.

    Parameters
    ----------
    filepath:
        Path to the ``.jsonl`` file.  Created on first write if it does
        not exist.
    """

    def __init__(self, filepath: str | Path) -> None:
        self._filepath = Path(filepath)

    @property
    def filepath(self) -> Path:
        return self._filepath

    # -- writing -------------------------------------------------------------

    def log(
        self,
        genome: Genome,
        fitness_score: float,
        fitness_breakdown: dict[str, float] | None = None,
        survived: bool = True,
        crossover_op: str | None = None,
        mutations_applied: Sequence[str] | None = None,
    ) -> LineageRecord:
        """Append a single evaluation record.

        Parameters
        ----------
        genome:
            The evaluated genome.
        fitness_score:
            Composite fitness score.
        fitness_breakdown:
            Per-metric scores (e.g. ``{"accuracy": 0.8, "calibration": 0.9}``).
        survived:
            Whether this genome was selected into the next generation.
        crossover_op:
            Name of the crossover operator that produced this genome (``None``
            for Gen-0 random genomes).
        mutations_applied:
            List of mutation operator names applied to this genome.

        Returns
        -------
        LineageRecord
            The record that was written.
        """
        # Build genome snapshot -- Genome is a Pydantic model, use to_dict()
        snapshot = genome.to_dict()

        record = LineageRecord(
            genome_id=genome.genome_id,
            generation=genome.generation,
            parents=list(genome.parents),
            fitness_score=fitness_score,
            fitness_breakdown=dict(fitness_breakdown or {}),
            survived=survived,
            crossover_operator=crossover_op,
            mutations_applied=list(mutations_applied or []),
            genome_snapshot=snapshot,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self._filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

        return record

    # -- reading -------------------------------------------------------------

    def load(self) -> list[LineageRecord]:
        """Read every record from the JSONL file.

        Returns an empty list when the file does not exist.
        """
        if not self._filepath.exists():
            return []

        records: list[LineageRecord] = []
        with open(self._filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(LineageRecord.from_dict(json.loads(line)))
        return records

    # -- queries -------------------------------------------------------------

    def get_champion(self, generation: int) -> LineageRecord | None:
        """Return the highest-fitness record for *generation*.

        Returns ``None`` when the generation has no records.
        """
        records = self.load()
        gen_records = [r for r in records if r.generation == generation]
        if not gen_records:
            return None
        return max(gen_records, key=lambda r: r.fitness_score)

    def get_lineage(self, genome_id: str) -> list[LineageRecord]:
        """Trace the ancestry of *genome_id* back to Gen 0.

        Returns a list ordered from the target genome back to its oldest
        known ancestor.
        """
        records = self.load()
        by_id: dict[str, LineageRecord] = {}
        for r in records:
            # Keep the first (earliest) record per id
            if r.genome_id not in by_id:
                by_id[r.genome_id] = r

        ancestry: list[LineageRecord] = []
        visited: set[str] = set()
        queue = [genome_id]

        while queue:
            gid = queue.pop(0)
            if gid in visited:
                continue
            visited.add(gid)
            rec = by_id.get(gid)
            if rec is None:
                continue
            ancestry.append(rec)
            for parent_id in rec.parents:
                if parent_id not in visited:
                    queue.append(parent_id)

        return ancestry

    def get_tree(self) -> dict[str, LineageNode]:
        """Return the full lineage as a tree of ``LineageNode`` objects.

        The returned dict is keyed by genome_id.
        """
        records = self.load()

        nodes: dict[str, LineageNode] = {}
        for r in records:
            if r.genome_id not in nodes:
                nodes[r.genome_id] = LineageNode(
                    genome_id=r.genome_id,
                    generation=r.generation,
                    fitness_score=r.fitness_score,
                    parents=list(r.parents),
                )

        # Populate children links
        for node in nodes.values():
            for parent_id in node.parents:
                if parent_id in nodes:
                    parent_node = nodes[parent_id]
                    if node.genome_id not in parent_node.children:
                        parent_node.children.append(node.genome_id)

        return nodes
