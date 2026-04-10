# Preregistration: Search Space Dimensionality vs. Operator Choice in LLM Agent Configuration Search

**Status:** LOCKED — committed before any confirmatory experiment runs.
**Date locked:** 2026-04-10
**Project:** agentbreed / `breed` library.

This document locks the confirmatory experimental plan. Any deviation
from this plan must be documented as post-hoc exploratory analysis and
labeled as such in the paper.

## 1. Background and motivation

Prior workshop-scale results on a 50-question synthetic forecasting
benchmark showed that (a) multi-component evolutionary search
statistically significantly outperforms prompt-only evolutionary search
(d_z = 1.45, Holm-adjusted p = 0.027), while (b) no pairwise comparison
between multi-component search methods (full evolution, mutation-only,
crossover-only, random search, best random init, static ensemble)
reached Holm-corrected significance at n = 8 seeds. This preregistration
tests whether those findings reproduce and generalize.

## 2. Hypotheses

### Primary hypotheses (confirmatory)

**H1 — Dimensionality effect.**
On each benchmark, full_evolution's mean test-set score exceeds
prompt_only_evolution's mean test-set score, with the paired bootstrap
95% CI of the difference excluding zero. The effect holds on at least
three out of four benchmarks.

**H2 — Operator null.**
For each pair (A, B) in the set of multi-component methods
{full_evolution, mutation_only, crossover_only, random_search,
bayesian_opt, successive_halving}, the absolute value of the paired
mean difference is <= 0.03 (3 percentage points of fitness) with the
paired bootstrap 95% CI containing zero. The null holds on at least
three out of four benchmarks.

**H3 — Epistasis mechanism.**
Functional ANOVA on the fitness landscape yields
`sum over (i < j) of S_{ij} >= 0.10`, i.e., at least 10% of fitness
variance is explained by two-way gene interactions, on at least two
benchmarks.

### Secondary hypotheses (exploratory)

**S1.** Champion configurations from one model transfer (retain at
least 80% of their fitness improvement over random_search) when
evaluated on a different model.

**S2.** Champion configurations from one benchmark do NOT transfer
(retain less than 50% of their fitness improvement) when evaluated on
a different benchmark.

**S3.** The dimensionality effect in H1 scales monotonically with the
number of gene axes K in {1, 2, 3, 5, 7, 9}.

## 3. Benchmarks

**Essential:**
1. ForecastBench (Karger et al., 2024, arXiv 2409.19839) — binary
   forecasting with time-based split. Metric: 1 - Brier score.
2. HumanEval (Chen et al., 2021) filtered to problems not in the
   pretraining set (use LiveCodeBench release date filter).
   Metric: pass@1.
3. MBPP sanitized (Austin et al., 2021). Metric: pass@1.
4. Metaculus historical binary questions (resolved 2022-2025).
   Metric: 1 - Brier score.

**If inaccessible:** substitute Autocast (Zou et al., NeurIPS 2022,
arXiv 2206.15474) for Metaculus.

**Split protocol:** strict time-based splits for all forecasting
benchmarks; random but frozen splits for coding benchmarks.
No method may touch the test set until all configuration decisions are
locked.

## 4. Models

**Primary (full search):**
- Llama-3.3-70B-Instruct (4-bit AWQ quantization via vLLM)
- Qwen-2.5-72B-Instruct (4-bit AWQ quantization via vLLM)

**Evaluation-only (champion transfer):**
- Claude Sonnet 4.6 via Anthropic API
- GPT-4o via OpenAI API

All model calls use `temperature = 0.0` for determinism except when
temperature is the evolved parameter of interest.

## 5. Methods / baselines

Nine methods in the primary suite:

1. `full_evolution` — population-based search with tournament selection,
   uniform+component-swap+weighted crossover, all six mutation operators,
   random immigrants.
2. `mutation_only` — same as full evolution but crossover disabled.
3. `crossover_only` — same as full evolution but mutation rate = 0.
4. `random_search` — population_size x generations random samples.
5. `best_random_init` — single population_size random sample, return best.
6. `static_ensemble` — prediction averaging over population_size random
   samples.
7. `prompt_only_evolution` — full evolution with all genes except
   prompt_template frozen at spawn-time values.
8. `bayesian_opt` — Gaussian Process / TPE Bayesian optimization on the
   genome space (via scikit-optimize or equivalent) with budget matched
   to population_size x generations.
9. `successive_halving` — Hyperband-style multi-fidelity search with
   budget matched.

Budget matching rule: every method receives exactly
`population_size x generations` LLM search-set evaluations plus
`|test_set|` final-evaluation calls. Any method requiring auxiliary
compute (GP fitting, etc.) has that cost reported but not counted
against the LLM budget.

## 6. Experimental design

### Primary experiment (E1)

Full factorial: 4 benchmarks x 2 primary search models x 9 methods x 20 seeds = 1,440 runs.

Hyperparameters (fixed):
- population_size = 20
- generations = 15
- elite_count = 4
- mutation_rate = 0.20
- immigration_rate = 0.10
- selection_method = tournament (size 3)

### Decisive dimensionality sweep (E_Decisive)

K in {1, 2, 3, 5, 7, 9} gene axes, full_evolution vs random_search,
on ForecastBench + HumanEval x 2 models x 20 seeds = 960 runs.

### Epistasis decomposition (E3)

Quasi-Monte Carlo Sobol sampling with 2,048 design points per
(benchmark, model) on ForecastBench and HumanEval x 2 models = 8,192
evaluations. Computes first-order, total-order, and all pairwise
interaction Sobol indices.

### Ablations (E2)

- Gene dropout: k in {0, 1, 3, 5, 7, 8} genes frozen, 3 benchmarks,
  2 models, 10 seeds = 360 runs.
- Mutation rate: {0.0, 0.05, 0.10, 0.20, 0.30}, 2 benchmarks, 2 models,
  10 seeds = 200 runs.
- Population size at matched budget: P in {5, 10, 20, 40, 80},
  2 benchmarks, 2 models, 10 seeds = 200 runs.

### Champion transfer (E_Transfer)

Champion from (model_i, benchmark_j) evaluated on (model_k, benchmark_l)
for all 16 combinations, per benchmark, per method.

## 7. Primary endpoints

For each (benchmark, model, method, seed):

- **Test-set mean score** (1 - Brier for forecasting, pass@1 for coding).

Reported with 95% paired bootstrap CI across seeds, with Holm-Bonferroni
correction across all H1 + H2 pairwise comparisons.

## 8. Statistical tests

- **Primary:** Paired bootstrap (10,000 resamples) for each pairwise
  method comparison at the seed level.
- **Secondary:** Wilcoxon signed-rank test.
- **Reference:** Paired t-test with Cohen's d_z effect size.
- **Aggregate:** Mixed-effects linear model
  `score ~ method + (1|benchmark) + (1|model) + (1|benchmark:model:seed)`
  fit via `statsmodels.MixedLM`.
- **Correction:** Holm-Bonferroni across all pairwise comparisons;
  Benjamini-Hochberg FDR as sensitivity check.

## 9. Decision rules

- **H1 is confirmed** if full_evolution beats prompt_only_evolution with
  paired bootstrap 95% CI excluding zero on at least 3 of 4 benchmarks.
- **H1 is rejected** if this condition fails on 2 or more benchmarks.
- **H1 is mixed** if confirmed on exactly 2 of 4 benchmarks.
- **H2 is confirmed** if all pairwise differences among
  multi-component methods have |mean| <= 0.03 and 95% CI containing
  zero on at least 3 of 4 benchmarks.
- **H2 is rejected** if any pairwise difference is > 0.05 with 95% CI
  excluding zero on the majority of benchmarks.
- **H3 is confirmed** if functional ANOVA yields
  sum_{i<j} S_{ij} >= 0.10 on at least 2 benchmarks.

## 10. Stopping rules

- All 1,440 primary runs (E1) must complete before primary inference.
- No optional stopping. No interim analyses of primary endpoints.
- Sensitivity analyses and exploratory figures may be computed
  continuously but will be labeled as such in the paper.

## 11. What counts as a pivot

If pilot (E1 on 1 benchmark x 1 model x 5 seeds) yields:
- Full evolution <= prompt_only evolution: pivot to Thesis B
  (benchmark paper).
- Full evolution > random search with p < 0.01 AND d_z > 0.5: reconsider
  whether the "operator null" frame is correct.
- Epistasis Sobol sum < 0.05 on the pilot: drop the "epistasis mechanism"
  frame and use "additive dimensionality" mechanism instead.

## 12. Reproducibility commitments

- All code open-sourced at submission time.
- All seeds fixed and documented.
- All random number streams seeded deterministically.
- All configuration files version-controlled.
- Docker image provided for bit-reproducible synthetic experiments.
- Documented non-determinism sources for real-LLM experiments.
- Full per-run JSONL logs released alongside the paper.

## 13. Authors' prior beliefs

Before running E1 on real LLMs, the principal investigator's prior
distribution over outcomes is:

- P(H1 confirmed) = 0.70
- P(H2 confirmed) = 0.55
- P(H3 confirmed) = 0.45
- P(all three confirmed) = 0.25
- P(pivot to Thesis B required) = 0.15

These priors are recorded to assess calibration after the fact.

## 14. Amendments

Any amendment to this preregistration must:
1. Be committed to git with a timestamped entry below.
2. Explicitly state what was changed and why.
3. Clearly label any affected analysis as post-hoc.

### Amendment log
(none)
