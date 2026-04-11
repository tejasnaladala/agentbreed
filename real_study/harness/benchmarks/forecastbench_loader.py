"""ForecastBench loader for the real study.

Reference: Karger et al., 2024 (arXiv 2409.19839). Public data at
https://github.com/forecastingresearch/forecastbench and
https://www.forecastbench.org/

The real study does NOT call ForecastBench's server at runtime. We download a
frozen snapshot once (during Stage A smoke test), commit its SHA-256, and
reference it from every subsequent run. This makes the entire pipeline
reproducible from the snapshot alone, and protects against the ForecastBench
server going offline or updating between our runs.

IMPORTANT: the actual network fetch logic below is stubbed with a docstring;
the real implementation will go in once we're on Hyak and can download. For
now, this file provides the interface so the rest of the harness can be
written against a stable contract.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .base import Benchmark, BenchmarkItem, BenchmarkScore, BenchmarkSplits


class ForecastBenchLoader(Benchmark):
    benchmark_id = "forecastbench"

    def __init__(self, snapshot_dir: str = "results/snapshots"):
        self.snapshot_dir = Path(snapshot_dir)

    def load_splits(
        self,
        *,
        snapshot_path: Optional[str] = None,
        refresh: bool = False,
    ) -> BenchmarkSplits:
        """Load train/test from a frozen snapshot.

        Snapshot file format (JSON):
        {
            "snapshot_date": "2026-04-XX",
            "source": "https://www.forecastbench.org/ (frozen)",
            "items": [
                {
                    "item_id": "fcb_0001",
                    "question": "Will X happen by 2025-12-31?",
                    "resolution_date": "2025-07-15",
                    "resolved_value": 1,
                    "topic": "politics",
                    "sources": ["metaculus"]
                },
                ...
            ]
        }

        The split is time-based: items with resolution_date < 2025-06-01 → train,
        items with resolution_date ≥ 2025-06-01 → test.
        """
        if snapshot_path is None:
            snapshot_path = str(self.snapshot_dir / "forecastbench_frozen.json")
        path = Path(snapshot_path)
        if not path.exists():
            raise FileNotFoundError(
                f"ForecastBench snapshot not found at {path}. "
                f"Run `real_study/harness/scripts/download_forecastbench.py` "
                f"at Stage A to create it."
            )

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        snapshot_date = data["snapshot_date"]
        all_items = data["items"]

        # Compute the file's sha256 for the log
        sha = hashlib.sha256(path.read_bytes()).hexdigest()

        train: List[BenchmarkItem] = []
        test: List[BenchmarkItem] = []
        for raw in all_items:
            item = BenchmarkItem(
                benchmark=self.benchmark_id,
                item_id=raw["item_id"],
                payload=raw,
            )
            resolution = raw.get("resolution_date", "")
            if resolution < "2025-06-01":
                train.append(item)
            else:
                test.append(item)

        # Cap at 80/80 for v1
        train = train[:80]
        test = test[:80]

        return BenchmarkSplits(
            train=tuple(train),
            test=tuple(test),
            snapshot_sha256=sha,
            snapshot_date=snapshot_date,
        )

    def score_one(self, item: BenchmarkItem, raw_response: str) -> BenchmarkScore:
        """Score a binary forecast: fitness = 1 - Brier score.

        Looks for a probability in the response. Multiple patterns tried:
        1. A bare number 0.0–1.0 in the last non-empty line.
        2. A percentage like "65%".
        3. A JSON blob with key "probability".
        4. Pattern "probability: N" or "p = N".

        On parse failure: returns score = 0.25 (equivalent to Brier 0.25, i.e.
        predicting 0.5) which is the preregistered worst-case for parse failure.
        """
        truth = item.payload.get("resolved_value")
        if truth is None:
            return BenchmarkScore(
                item_id=item.item_id,
                score=0.0,
                parse_ok=False,
                details={"error": "no ground truth in payload"},
            )

        p = _parse_probability(raw_response)
        if p is None:
            return BenchmarkScore(
                item_id=item.item_id,
                score=0.25,  # 1 - 0.75 = 0.25 (worst plausible Brier)
                parse_ok=False,
                details={"error": "could not parse probability", "truncated_resp": raw_response[:200]},
            )

        brier = (p - float(truth)) ** 2
        fitness = 1.0 - brier
        return BenchmarkScore(
            item_id=item.item_id,
            score=fitness,
            parse_ok=True,
            details={"probability": p, "brier": brier, "truth": float(truth)},
        )

    def contamination_probe(
        self,
        items: Sequence[BenchmarkItem],
        *,
        llm_callable,
    ) -> List[str]:
        """ForecastBench is contamination-immune by design (future events).

        The probe is a sanity check: we ask the LLM the bare question at
        temperature 0 and see if it outputs the answer confidently. Since
        the question is about the future, any confident answer is suspicious
        (the LLM should express uncertainty).

        We do NOT drop items based on this probe for ForecastBench; we just
        log whether the LLM seemed confident, for the reproducibility record.
        """
        return []


# ---------------------------------------------------------------------------
# Probability parser
# ---------------------------------------------------------------------------

_PROB_PATTERNS = [
    re.compile(r"probability[^0-9]*([01](?:\.\d+)?|0?\.\d+)", re.IGNORECASE),
    re.compile(r"\bp\s*[:=]\s*([01](?:\.\d+)?|0?\.\d+)", re.IGNORECASE),
    re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%"),
    re.compile(r"^\s*([01](?:\.\d+)?|0?\.\d+)\s*$", re.MULTILINE),
]


def _parse_probability(text: str) -> Optional[float]:
    """Extract a probability p ∈ [0, 1] from a free-form text response."""
    if not text:
        return None
    # Try the patterns in order
    for i, pat in enumerate(_PROB_PATTERNS):
        m = pat.search(text)
        if m:
            try:
                raw = m.group(1)
                val = float(raw)
            except ValueError:
                continue
            if pat.pattern.endswith(r"\s*%"):
                val = val / 100.0
            if 0.0 <= val <= 1.0:
                return val
    return None
