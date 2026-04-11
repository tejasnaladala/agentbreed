"""Runner that wires a (benchmark, model, method, seed) cell into one atomic run.

The runner is the only place where per-run state is assembled. It:
1. Loads the benchmark snapshot (train/test splits).
2. Instantiates the model client.
3. Constructs a fitness_fn closure that evaluates a genome on the train split
   by issuing LLM calls via the client.
4. Runs the search method with a fixed budget.
5. Evaluates the champion on the test split.
6. Writes an atomic JSON record to results/{run_id}.json with everything needed
   for reproducibility.

Design goals:
- **Atomic writes.** The JSON is first written to `{run_id}.json.tmp` then
  renamed to `{run_id}.json`. Slurm arrays can crash mid-cell without corrupting
  anything.
- **Deterministic run_id.** `sha256((method, benchmark, model, seed, code_sha, prereg_sha))`.
- **Graceful degradation.** If an LLM call fails, the fitness function logs
  the failure and scores it as worst-case. The search method continues.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .benchmarks.base import Benchmark, BenchmarkItem, BenchmarkSplits
from .genome import Genome, genome_content_hash
from .models.base import ModelClient
from .search.base import FitnessFn, SearchContext, SearchMethod, SearchResult


@dataclass
class RunSpec:
    benchmark: str
    model_name: str
    method: str
    seed: int
    population_size: int = 20
    generations: int = 15
    code_sha: str = ""
    prereg_sha: str = ""
    snapshot_sha256: str = ""
    quantization: str = "fp16"
    tp_size: int = 1
    gpu_type: str = "H100"


@dataclass
class RunRecord:
    run_id: str
    spec: Dict[str, Any]
    started_at: str
    completed_at: str = ""
    champion_genome: Dict[str, Any] = field(default_factory=dict)
    champion_hash: str = ""
    champion_test_score: float = 0.0
    champion_train_score: float = 0.0
    per_generation_best: List[float] = field(default_factory=list)
    total_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_vllm_wall_time_s: float = 0.0
    parse_failures_train: int = 0
    parse_failures_test: int = 0
    notes: Dict[str, Any] = field(default_factory=dict)


def make_run_id(spec: RunSpec) -> str:
    key = "|".join(
        [
            spec.benchmark,
            spec.model_name,
            spec.method,
            str(spec.seed),
            spec.code_sha or "nosha",
            spec.prereg_sha or "noprereg",
            spec.snapshot_sha256 or "nosnap",
            str(spec.population_size),
            str(spec.generations),
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def get_code_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return out.decode().strip()
    except Exception:
        return ""


class Runner:
    def __init__(
        self,
        *,
        benchmark: Benchmark,
        splits: BenchmarkSplits,
        model: ModelClient,
        search_method: SearchMethod,
        spec: RunSpec,
        results_dir: str = "real_study/results",
        prompt_builder=None,  # callable: (genome, item) -> (system, user, max_tokens, stop)
    ):
        self.benchmark = benchmark
        self.splits = splits
        self.model = model
        self.search_method = search_method
        self.spec = spec
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_builder = prompt_builder
        # running totals
        self._input_tokens = 0
        self._output_tokens = 0
        self._vllm_time_s = 0.0
        self._llm_calls = 0
        self._parse_failures_train = 0
        self._parse_failures_test = 0

    async def _evaluate_genome_on_items(
        self,
        genome: Genome,
        items: Tuple[BenchmarkItem, ...],
        *,
        is_train: bool,
    ) -> float:
        """Evaluate one genome on a set of benchmark items, returning mean fitness."""
        # Build prompts for each item, call the model, score.
        scores: List[float] = []
        for item in items:
            if self.prompt_builder is None:
                # Minimal default: just pass the payload as a user message
                system = None
                user = json.dumps(item.payload, ensure_ascii=False)[:2000]
                max_tokens = 256
                stop = None
            else:
                system, user, max_tokens, stop = self.prompt_builder(genome, item)

            temperature = float(genome.get("temperature"))
            result = await self.model.generate(
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=self.spec.seed,
                stop=stop,
            )
            self._llm_calls += 1
            self._input_tokens += result.input_tokens
            self._output_tokens += result.output_tokens
            self._vllm_time_s += result.wall_time_ms / 1000.0

            bs = self.benchmark.score_one(item, result.text)
            scores.append(bs.score)
            if not bs.parse_ok:
                if is_train:
                    self._parse_failures_train += 1
                else:
                    self._parse_failures_test += 1

        return sum(scores) / max(1, len(scores))

    def make_fitness_fn(self) -> FitnessFn:
        """Closure that evaluates a single genome on the train split."""
        async def _fitness(genome: Genome) -> Tuple[float, int]:
            mean = await self._evaluate_genome_on_items(
                genome, self.splits.train, is_train=True
            )
            return mean, 1  # one genome-level evaluation
        return _fitness

    async def run(self) -> RunRecord:
        run_id = make_run_id(self.spec)
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        ctx = SearchContext(
            benchmark=self.spec.benchmark,
            seed=self.spec.seed,
            budget_llm_calls=self.spec.population_size * self.spec.generations,
            population_size=self.spec.population_size,
            generations=self.spec.generations,
        )

        fitness_fn = self.make_fitness_fn()
        search_result: SearchResult = await self.search_method.run(
            ctx=ctx, fitness_fn=fitness_fn
        )

        # Evaluate champion on test set
        champion_test = await self._evaluate_genome_on_items(
            search_result.champion, self.splits.test, is_train=False
        )
        completed = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        record = RunRecord(
            run_id=run_id,
            spec=asdict(self.spec),
            started_at=started,
            completed_at=completed,
            champion_genome={
                "benchmark": search_result.champion.benchmark,
                "values": [(n, _jsonable(v)) for n, v in search_result.champion.values],
            },
            champion_hash=genome_content_hash(search_result.champion),
            champion_test_score=champion_test,
            champion_train_score=search_result.champion_fitness_search,
            per_generation_best=search_result.per_generation_best,
            total_llm_calls=self._llm_calls,
            total_input_tokens=self._input_tokens,
            total_output_tokens=self._output_tokens,
            total_vllm_wall_time_s=round(self._vllm_time_s, 3),
            parse_failures_train=self._parse_failures_train,
            parse_failures_test=self._parse_failures_test,
            notes={
                "method_notes": search_result.notes,
                "n_search_evals": len(search_result.all_evaluated),
            },
        )

        self._write_atomic(run_id, record)
        return record

    def _write_atomic(self, run_id: str, record: RunRecord) -> None:
        path = self.results_dir / f"{run_id}.json"
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(asdict(record), f, indent=2, default=_jsonable)
        os.replace(tmp, path)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, frozenset):
        return sorted(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, tuple):
        return list(obj)
    return obj
