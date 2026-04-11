"""Benchmark abstractions shared across ForecastBench / LiveCodeBench / GPQA Diamond."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


BENCHMARK_IDS = ("forecastbench", "livecodebench_v6", "gpqa_diamond")


@dataclass(frozen=True)
class BenchmarkItem:
    """A single scorable benchmark item.

    `payload` is an opaque dict that each benchmark scorer knows how to interpret.
    For ForecastBench: includes the question text and resolved ground-truth probability.
    For LCB: includes the problem statement and reference test cases.
    For GPQA: includes the question, options, and ground-truth letter.
    """
    benchmark: str
    item_id: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class BenchmarkScore:
    """Per-item score produced by the scorer."""
    item_id: str
    score: float  # 0.0 to 1.0; higher is better
    parse_ok: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkSplits:
    train: Tuple[BenchmarkItem, ...]
    test: Tuple[BenchmarkItem, ...]
    snapshot_sha256: str
    snapshot_date: str  # ISO-8601 date string


class Benchmark(abc.ABC):
    """Abstract base class for a benchmark."""

    benchmark_id: str = ""

    @abc.abstractmethod
    def load_splits(
        self,
        *,
        snapshot_path: Optional[str] = None,
        refresh: bool = False,
    ) -> BenchmarkSplits:
        """Load (or download) the train/test splits.

        If `snapshot_path` is provided, load from that frozen snapshot (preferred).
        If not, download the latest and write a snapshot for reproducibility.
        """

    @abc.abstractmethod
    def score_one(
        self,
        item: BenchmarkItem,
        raw_response: str,
    ) -> BenchmarkScore:
        """Score a single agent response against the item's ground truth."""

    @abc.abstractmethod
    def contamination_probe(
        self,
        items: Sequence[BenchmarkItem],
        *,
        llm_callable,
    ) -> List[str]:
        """Run a contamination probe and return the list of item_ids to drop."""

    # -----------------------------------------------------------------------
    # Default utility: mean score over a set of (item, response) pairs
    # -----------------------------------------------------------------------
    def score_many(
        self,
        pairs: Iterable[Tuple[BenchmarkItem, str]],
    ) -> Tuple[float, Tuple[BenchmarkScore, ...]]:
        scores: List[BenchmarkScore] = []
        total = 0.0
        n = 0
        for item, resp in pairs:
            s = self.score_one(item, resp)
            scores.append(s)
            total += s.score
            n += 1
        mean = total / n if n > 0 else 0.0
        return mean, tuple(scores)
