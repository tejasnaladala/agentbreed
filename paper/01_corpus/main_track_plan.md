# Main-Track Upgrade Plan

This document is the locked strategic plan for converting the workshop-level
result into a main-track submission. It is the canonical reference for all
experimental design decisions.

## Thesis

**Primary:** "The Search Space is the Story" — epistasis, not operators, is
the load-bearing variable in LLM agent configuration search. Multi-component
search dominates prompt-only search across benchmarks and models, but
within the space of multi-component methods (full evolution, mutation-only,
crossover-only, random search, Bayesian optimization, successive halving),
operator choice is indistinguishable at matched compute. We quantify the
mechanism via functional ANOVA decomposition of the fitness landscape.

**Backup:** Benchmark/artifact paper releasing the typed-genome framework
and reproducibility stack, suitable for NeurIPS D&B track.

## Headline claims (to be confirmed by the full study)

1. **H1 (dimensionality):** Full-evolution minus prompt-only-evolution > 0
   on at least 3 out of 4 benchmarks with paired bootstrap 95% CI excluding
   zero on each benchmark.

2. **H2 (operator null):** Among {full_evolution, mutation_only,
   crossover_only, random_search, bayesian_opt, successive_halving},
   pairwise differences have 95% CI containing zero on at least 3 out of
   4 benchmarks at n=20 seeds.

3. **H3 (epistasis mechanism):** Sum of pairwise interaction Sobol indices
   >= 0.10 on at least 2 benchmarks, providing a mechanistic explanation
   for the dimensionality effect.

## Experiment packet

- E1: 4 benchmarks x 4 models x 9+ methods x 20 seeds primary matrix.
- E2: Ablations (gene dropout, population size, mutation rate,
  crossover operator variants, diversity mechanisms).
- E3: Functional ANOVA / Sobol decomposition on the fitness landscape.
- E_Decisive: Controlled dimensionality sweep K in {1,2,3,5,7,9} with
  full_evolution vs random_search at matched compute.
- E_Transfer: Champion-transfer matrix across models and benchmarks.
- Stress tests: budget scaling, noise injection, contamination control,
  negative controls.

## Statistical plan

- Paired bootstrap (10,000 resamples) primary; Wilcoxon secondary;
  paired t for reference.
- Holm-Bonferroni across all H1 + H2 pairwise comparisons.
- Mixed-effects model for aggregate inference across benchmark x model
  pairs: score ~ method + (1|benchmark) + (1|model) + (1|seed:bench:model).
- Minimum 20 seeds per cell. Power analysis: d_z = 0.8 detectable at 95%
  power with n=20.
- Preregistration locked before any real-LLM experiment runs.

## Benchmark portfolio

**Essential:**
- ForecastBench (forecasting, time-based split, Brier).
- HumanEval + MBPP with LiveCodeBench contamination filter (coding,
  pass@1).
- Metaculus historical (second forecasting source for cross-dataset
  generalization).

**Strong recommendation:**
- GPQA Diamond (knowledge/reasoning).

**Aspirational:**
- WebArena Lite (tool-use).

## Model strategy

- Primary search: Llama-3.3-70B-Instruct (4-bit AWQ) and Qwen-2.5-72B-Instruct
  (4-bit AWQ) via vLLM.
- Evaluation-only (champion transfer): Claude Sonnet 4.6 and GPT-4o.
- No fine-tuning. Configuration search only.

## Execution phases

- Phase A (Week 1): Audit & freeze.
- Phase B (Weeks 2-3): Benchmark integration.
- Phase C (Weeks 2-4): Large-scale infrastructure.
- Phase D (Week 5): Pilot experiments.
- Phase E (Weeks 6-8): Full main study.
- Phase F (Weeks 8-10): Analysis and paper.
- Phase G (Weeks 9-11): Artifact release.

## Minimal viable package

- 2 benchmarks (ForecastBench + HumanEval).
- 2 open-weight models.
- 9 methods.
- 20 seeds.
- Approximate cost: ~500 H200-hours, ~$100k.
- Time: 4 weeks from preregistration to draft.

## Non-negotiable

- Preregistration signed before experiments.
- No post-hoc modification of primary endpoints.
- Report all methods on all benchmarks; no cherry-picking.
- Honest null result publication if H1 or H2 fails.
- Open source code and data at submission time.
