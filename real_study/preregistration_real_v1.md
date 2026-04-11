# Preregistration — Real-LLM Agent Configuration Search Study v1

**Status:** LOCKED — committed before any real-LLM confirmatory run.
**Date locked:** 2026-04-10
**Principal investigator:** Tejas (solo author)
**Project:** `agentbreed` / `breed` library.
**Target venue (primary):** NeurIPS 2026 Evaluations and Datasets track.
**Target venue (backup):** NeurIPS 2026 main track (only if evidence supports a clean surprising finding).

## 0. Relationship to prior preregistration

This document supersedes, for the purposes of this paper, the original preregistration at `../paper/01_corpus/preregistration.md` (dated 2026-04-10, committed earlier on the same day). The original preregistration was not satisfied by the synthetic pilot work that followed it: the pilot used synthetic agents rather than the preregistered real benchmarks and models, applied decision rules that did not match the locked definitions, and ran Sobol at an insufficient sample size. The original preregistration is preserved as historical record and is a valid scientific plan — what it lacked was execution.

This document describes the real-LLM study that was always the intended target. It restates the plan, corrects the decision rules for the real study's benchmark and method count, and binds the author to a new locked protocol before any real-LLM run.

## 1. Background and motivation

See `docs/01_research_brief.md`. The question is: when optimizing LLM agent configurations across multiple axes, what actually contributes to the gain — the search operator, or the richness of the search space?

We test this with preregistered hypotheses H1–H3 on three real benchmarks across two real open-weight models. Full details are in `docs/02_benchmark_suite.md`, `docs/03_model_suite.md`, `docs/04_genome_schema.md`, `docs/05_methods.md`, `docs/06_analysis_plan.md`.

## 2. Hypotheses

### Primary (confirmatory)

- **H1 — Dimensionality effect.** Mean paired difference `full_evolution − prompt_only_evolution` on the test set, estimated via the primary LMM contrast, has its 95% CI upper bound below `−0.05`.

- **H2 — Operator null (equivalence).** All 21 pairwise comparisons in `{full_evolution, mutation_only, crossover_only, random_search, coordinate_descent, bayesian_opt, successive_halving}` establish TOST equivalence at margin `Δ_eq = 0.03`, Holm-corrected at α = 0.05.

- **H3 — Epistasis mechanism.** Sobol decomposition at `n_base = 2048` on at least 2 of 6 `(benchmark, model)` cells produces `Σ_{i<j} S_{ij} ≥ 0.10` with a valid total-variance decomposition.

### Secondary (exploratory, for reporting)

- **S1** cross-model transfer (retention ≥ 0.5)
- **S2** cross-benchmark non-transfer (retention < 0.3)
- **S3** monotonic dimensionality sweep
- **S4** category-specific effects (semantic > compute-budget)
- **S5** ranking stability under wall-clock normalization

## 3. Benchmarks

Frozen:
1. **ForecastBench** (Karger et al., 2024, arXiv 2409.19839). Frozen snapshot taken at Stage A. 80 train / 80 test binary questions resolved after the later of 2025-06-01 and the model's training cutoff.
2. **LiveCodeBench v6** (Jain et al., 2024). Problems released 2024-12-01 to 2025-04-30. 80 train / 40 test time-split.
3. **GPQA Diamond** (Rein et al., 2023). All 198 items. 80 train / 80 test random split with seed 42. Remaining 38 held out for sanity-check.

No benchmark may be added, removed, or modified after this preregistration is locked. The snapshot hashes are committed to `results/snapshots/` at Stage A.

## 4. Models

Frozen:
1. **Qwen3-32B** (Alibaba, April 2026) — primary mid-size.
2. **Llama-3.3-70B-Instruct** (Meta, December 2024) — primary large.
3. **Qwen3-14B** (Alibaba, April 2026) — optional scaling reference; included in the confirmatory matrix only if Stage-B pilot shows wall-clock slack.

Inference: vLLM ≥ 0.7.x, fp16 by default, AWQ 4-bit as fallback. Seeds passed to vLLM for best-effort determinism. Context cap 8k tokens.

No additional models may be added to the primary matrix post-lock. A closed-model sanity check (e.g. GPT-5) may be reported as an **exploratory** add-on only, clearly labeled.

## 5. Methods

Confirmatory method set (|M| = 8):
1. `full_evolution`
2. `mutation_only`
3. `crossover_only`
4. `random_search`
5. `coordinate_descent`
6. `bayesian_opt` (TPE)
7. `successive_halving`
8. `prompt_only_evolution`

Budget: `population_size = 20`, `generations = 15`, so `B = 300` search-set calls per cell. Plus `|test_set|` test calls.

Auxiliary compute (TPE fits, SH bracket arithmetic) is reported but not counted against `B`.

`best_of_20_init` is the **budget-scaling anchor** — tested at multiple budgets, reported in the budget-scaling secondary analysis only, NOT in the primary matrix.

No methods may be added post-lock. If `coordinate_descent` turns out to be pathologically slow in practice, it may be dropped — and if so, the drop will be documented in the amendment log and the preregistration will be re-frozen before any affected analysis runs.

## 6. Experimental design

### Main confirmatory matrix (E1)

`|M| × |B| × |L| × n_seeds = 8 × 3 × 2 × 15 = 720 cells`.

Per cell: 300 train calls + 40 or 80 test calls.

Total LLM calls in main matrix: ~220,000.

### Decisive dimensionality sweep (E_Decisive)

`2 methods (full_evolution, random_search) × 2 benchmarks (ForecastBench, LiveCodeBench v6) × 2 models × 6 K-values × 15 seeds = 720 cells`.

K ∈ {1, 2, 3, 5, 7, 9} from the canonical sweep order (see `docs/04_genome_schema.md` §3).

### H3 Sobol decomposition

2 of 6 `(b, l)` cells × `n_base = 2048` Saltelli points × 80 search items = ~90,000 evaluations.

### Transfer matrix

Champion-only evaluations for S1 and S2. No search loop; just evaluate source-cell champions on target cells. ~10,800 calls total.

### Exploratory (post-confirmatory)

- Budget-scaling curves
- Category ablations
- Sensitivity to seed count
- Sensitivity to wall-clock normalization
- Qualitative failure-mode review

None of these affect the primary decision rules.

## 7. Primary endpoint

For each `(m, b, l, s)` cell, the primary endpoint is:

**`y_{m,b,l,s}` = mean metric over the test set** (1 − Brier for forecasting, pass@1 for coding, accuracy for reasoning).

Paired differences are computed within-`(b, l, s)`.

## 8. Statistical tests

See `docs/06_analysis_plan.md` for full details. Summary:

- **Primary aggregate test:** LMM `y ~ method + (1|benchmark) + (1|model) + (1|benchmark:model) + (1|benchmark:model:seed)` fit via REML.
- **H1:** LMM contrast + 95% Wald CI.
- **H2:** TOST on LMM pairwise contrasts with Holm-Bonferroni across 21 pairs.
- **H3:** Saltelli-Sobol with validity check.
- **Per-cell supplementary:** paired bootstrap (10,000 resamples) with Holm across 6 cells.

## 9. Decision rules

### H1
- **Confirmed:** upper CI < −0.05.
- **Mixed:** upper CI ∈ [−0.05, 0.0].
- **Rejected:** upper CI ≥ 0.0.

### H2
- **Confirmed:** all 21 pairs establish TOST equivalence at Holm-corrected α = 0.05.
- **Partially confirmed:** ≥ 16/21 pairs equivalent.
- **Rejected:** < 16/21 equivalent OR any pair CI has absolute endpoint > 0.05.

### H3
- **Confirmed:** `Σ S_{ij} ≥ 0.10` on ≥ 2 valid cells.
- **Rejected:** `Σ S_{ij} < 0.10` on ≥ 4 cells.
- **Inconclusive:** otherwise.

## 10. Stopping rules

- All 720 main-matrix cells must complete before primary inference runs.
- No optional stopping, no interim analyses of primary endpoints.
- Sensitivity analyses and figures may be generated continuously during Stage E, but primary inference must not be run until Stage F.

## 11. Pivot rules

If Stage B pilot (ForecastBench × Qwen3-32B × 8 methods × 5 seeds) yields:
- `full_evolution ≤ prompt_only_evolution` on ≥ 3 of 5 seeds: pause, check scoring code, check parser, check the frozen snapshot. Do not proceed until the anomaly is explained.
- `full_evolution` ties `random_search` cleanly (difference < 0.02 on all 5 seeds): Stage C proceeds; this is actually the expected pattern under the operator-null hypothesis.
- Mean fitness across all methods is below 0.3 on ForecastBench: the benchmark is too hard for Qwen3-32B or the parser is broken. Investigate before Stage C.

## 12. Reproducibility commitments

- All code open-sourced at submission time (in the main `agentbreed` repo).
- All seeds fixed and documented.
- All benchmark snapshots committed with SHA-256.
- All model weights referenced by HuggingFace revision string.
- All run outputs are atomic per-cell JSONs keyed by SHA-256 of run config.
- Docker container provided for the analysis pipeline.
- Slurm scripts provided for the execution pipeline.

## 13. Authors' prior beliefs

Before running any real-LLM confirmatory experiment:

- P(H1 confirmed) = 0.65
- P(H2 confirmed) = 0.40
- P(H3 confirmed) = 0.50
- P(all three confirmed) = 0.18
- P(H1 confirmed, H2 rejected) = 0.35
- P(H1 rejected) = 0.15
- P(S1 confirmed) = 0.55
- P(S2 confirmed) = 0.70

## 14. Amendments

Any amendment must:
1. Be committed to git with a timestamped entry below.
2. Explicitly state what was changed and why.
3. Clearly label any affected analysis as post-hoc in the paper.

### Amendment log

(none)

## 15. Commitment

By committing this file to the git repository, the author locks the experimental plan described above. The author agrees to:

- Report all preregistered primary and secondary analyses, whether or not they produce a "clean" result.
- Publish a null result honestly if H1 or H2 or H3 is rejected.
- Document every deviation from this plan as a dated amendment.
- Not silently reinterpret any decision rule.
- Not add or remove methods, benchmarks, models, or hypotheses after the lock date.

This preregistration is the scientific record that governs the paper.
