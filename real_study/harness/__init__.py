"""real_study.harness — production harness for the real-LLM study.

Layout:
- genome/      typed genome schema + validator
- benchmarks/  per-benchmark loaders and scorers (ForecastBench, LCB v6, GPQA)
- models/      vLLM client + model registry
- search/      8 confirmatory search methods (budget-matched)
- stats/       LMM, TOST, Sobol, paired bootstrap, figure generation
- slurm/       Slurm job templates and submit scripts
- scripts/     Runnable entry points (smoke, pilot, main, analysis)
- tests/       unit tests

The harness is deliberately separate from `breed/`. Any code reused from `breed/`
is imported explicitly and the import is logged so we know what's shared. Scientific
claims in the real paper depend ONLY on `real_study/` execution; no pilot artifacts
leak in silently.
"""

from __future__ import annotations

__version__ = "0.1.0"
