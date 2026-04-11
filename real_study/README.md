# Real Study — Agent Configuration Search on Real LLMs

This directory contains the design, preregistration, harness, and (eventually) results of the NeurIPS-targeted real-LLM study on agent configuration search. It is intentionally isolated from `../paper/` (the synthetic pilot) so no pilot claim can silently leak into the real paper.

## Navigation

| File | Purpose |
|---|---|
| `docs/01_research_brief.md` | Phase 2 — three framing options, evaluated against each other, primary + backup picked |
| `docs/02_benchmark_suite.md` | Phase 3 — ForecastBench / LiveCodeBench v6 / GPQA Diamond with exact protocols |
| `docs/03_model_suite.md` | Phase 4 — Qwen3-32B + Llama-3.3-70B (+ optional Qwen3-14B) with vLLM stack |
| `docs/04_genome_schema.md` | Phase 5 — formal genome with 9 scientifically-legible genes |
| `docs/05_methods.md` | Phase 6 — 9-method ladder with budget-matching rules |
| `docs/06_analysis_plan.md` | Phase 7 — preregistered statistical plan with explicit formulas |
| `docs/07_compute_plan.md` | Phase 8 — Hyak klone GPU plan, stages, wall-clock estimates |
| `docs/08_hypotheses.md` | Phase 9 — H1–H4 with estimators and decision rules |
| `docs/09_execution_stages.md` | Phase 11 — smoke → pilot → main → exploratory stage gates |
| `docs/10_paper_outline.md` | Phase 12 — NeurIPS-caliber paper outline |
| `docs/11_risk_memo.md` | Phase 14 — blunt list of what could still kill the paper |
| `preregistration_real_v1.md` | **THE LOCK.** Committed before any real-LLM run. Hypotheses, methods, benchmarks, estimators all frozen. |
| `harness/` | Implementation: benchmark loaders, model clients, search algorithms, runner, analysis |
| `logs/` | Per-run JSONL logs (gitignored contents, structure committed) |
| `results/` | Per-run artifact storage (gitignored contents, structure committed) |

## Principles

1. **No silent changes.** If the preregistration says X and we need to do Y, we add a dated amendment with a justification and explicitly label any affected analysis as post-hoc.
2. **Budget-matched by construction.** Every method gets exactly `population_size × generations` LLM calls on the search set, plus `|test_set|` on the test set. Auxiliary compute (TPE fits, SH partial-fidelity calls) is reported separately but not counted.
3. **Atomic per-run logging.** Every (benchmark, model, method, seed) is a single atomic JSON file keyed by a SHA-256 hash of the run config. No shared mutable state.
4. **Mixed-effects aggregation.** The primary cross-benchmark inference is a linear mixed model, not a pooled t-test.
5. **Equivalence testing for H2.** The operator-null hypothesis is tested with equivalence bounds, not by failure-to-reject-H0.
6. **Sobol at n_base ≥ 2048.** Never at the broken 512 sample size from the pilot.
7. **Reproducibility-first.** Every Slurm job script, every config, every seed is committed. The harness can re-run any cell on demand.
8. **Real benchmarks only.** No synthetic analogs in the paper body. The synthetic pilot exists as prior work in the related work section, labeled as such.

## Non-negotiables from the pilot retrospective

- The word "preregistered" applies only to runs that happened **after** the lock commit of `preregistration_real_v1.md`.
- The prior preregistration at `../paper/01_corpus/preregistration.md` is preserved as historical record and is the original intent of the real-LLM study, but it is superseded for the purposes of this paper by `preregistration_real_v1.md`, which is locked with explicit reference to the pilot and the amendment rationale.
- No baseline method is added after runs have started.
- No decision rule is reinterpreted after data is observed.
- If H1 or H2 is rejected, the paper publishes a null result honestly.
