# Phase 6 — Baselines and Methods Ladder

## Design principles

1. **Budget-matched by construction.** Every method receives exactly `B = population_size × generations` LLM evaluations on the train/search set, plus `|test_set|` evaluations at the very end. Auxiliary compute (TPE density fits, surrogate training if any) is reported separately but does not count against `B`.
2. **No post-hoc additions.** The method list is frozen in the preregistration. If a reviewer asks "why not X?" the answer is either "because X was not in our preregistered set" or "we included X as an exploratory analysis clearly labeled as such."
3. **Each method must produce a single champion genome at the end**, selected by mean search-set fitness, which is then evaluated on the held-out test set.
4. **Deterministic given seed.** Every method's trajectory is reproducible from `(method, benchmark, model, seed)`.

## The 9 methods (v1)

### 1. `full_evolution`

**Description:** population-based evolutionary search. Tournament selection (size 3), with three crossover operators (uniform, component-swap, fitness-weighted), six type-aware mutation operators (enum flip, float perturb, set toggle, category swap, zero-out, identity), random immigrant injection at 10% rate, and elitism of the top-4 parents each generation.

**Hyperparameters (fixed, preregistered):**
- `population_size = 20`
- `generations = 15`
- `elite_count = 4`
- `mutation_rate = 0.20`
- `crossover_rate = 0.90`
- `immigration_rate = 0.10`
- `tournament_size = 3`

**Budget:** 20 × 15 = 300 search calls per benchmark item → 300 × 80 = 24,000 per cell.

### 2. `mutation_only`

**Description:** same as full_evolution but `crossover_rate = 0`. Offspring are produced by cloning selected parents and applying mutation operators.

### 3. `crossover_only`

**Description:** same as full_evolution but `mutation_rate = 0`. Offspring from crossover only. Non-obvious but important: without mutation, the population can lose diversity quickly — this is a feature, not a bug, for testing "does mutation carry the signal?"

### 4. `random_search`

**Description:** sample 300 uniformly random genomes and return the best on the search set. Zero structure — this is the null operator.

### 5. `best_of_20_init`

**Description:** sample 20 uniformly random genomes, evaluate them on the search set, return the best. This isolates "search-space quality without any search" — it is the budget-matched equivalent of "good hyperparameter defaults + random search with budget = population_size."

**Budget:** 20 × 80 = 1,600 search calls. **This method uses a smaller budget by design** — it is tested at multiple budgets as a budget-scaling experiment.

### 6. `coordinate_descent`

**Description:** greedy one-gene-at-a-time local search. Start from a random genome; for each gene in the canonical sweep order, try all legal values (discretized FLOAT as 13 values), fix the best; iterate until no gene changes improve search-set fitness or budget is exhausted. If the budget is exhausted mid-iteration, stop gracefully.

**Deterministic start:** the initial genome is seeded by `seed` and fixed across restarts.

### 7. `bayesian_opt` (TPE)

**Description:** Tree-structured Parzen Estimator over the encoded genome space. Uses `optuna` as the backend (mature, well-tested, produces reproducible results with `RandomSampler(seed=N)` bootstrapping then TPE takeover after 5 warm-up trials).

**Hyperparameters:**
- `n_startup_trials = 5`
- `n_ei_candidates = 24`
- `multivariate = True`

**Budget:** 300 evaluations — 5 warm-up random + 295 TPE-guided.

### 8. `successive_halving`

**Description:** Hyperband-style multi-fidelity with `η = 3` and fidelity floor 5 train items. Start with 81 candidates at 5 items each (405 calls), then 27 candidates at 15 items each (405 calls), then 9 candidates at 45 items each (405 calls), then 3 candidates at 80 items each (240 calls) = **1455 calls** — too many. Adjust.

**Adjusted budget:** tune the SH bracket so total calls = 300 exactly. One valid bracket:
- 27 candidates at 3 items = 81 calls
- 9 candidates at 9 items = 81 calls
- 3 candidates at 27 items = 81 calls
- 1 candidate at 57 items = 57 calls
- Total: 300 calls ✓

### 9. `prompt_only_evolution`

**Description:** full_evolution with all genes except `system_prompt` frozen at their spawn-time random values. The search axis is reduced to the 5 values of `system_prompt`.

This is the **H1 anchor** — the comparison of full_evolution vs. prompt_only_evolution is what H1 is measuring.

## Method compatibility table

| Method | Evolutionary | Uses crossover | Uses mutation | Population-based | Multi-fidelity | Learned component |
|---|---|---|---|---|---|---|
| full_evolution | ✓ | ✓ | ✓ | ✓ | — | — |
| mutation_only | ✓ | — | ✓ | ✓ | — | — |
| crossover_only | ✓ | ✓ | — | ✓ | — | — |
| random_search | — | — | — | — | — | — |
| best_of_20_init | — | — | — | partial | — | — |
| coordinate_descent | — | — | — | — | — | — |
| bayesian_opt | — | — | — | — | — | TPE |
| successive_halving | — | — | — | partial | ✓ | — |
| prompt_only_evolution | ✓ | ✓ | ✓ | ✓ | — | — |

## Budget-matching audit

All methods except `best_of_20_init` receive exactly 300 search-set evaluations. `best_of_20_init` receives 20 and is used only in the budget-scaling secondary analysis.

For the main matrix, `best_of_20_init` is **excluded** — it is a budget-varying method and belongs in a separate experiment. The primary 9-method matrix uses `{full_evolution, mutation_only, crossover_only, random_search, coordinate_descent, bayesian_opt, successive_halving, prompt_only_evolution}` at matched budget = 8 methods. Plus optionally `best_of_20_init` at reduced budget for budget-scaling = 9 distinct lines on a scaling plot.

**Decision: the confirmatory matrix is 8 methods at matched budget. `best_of_20_init` is a budget-scaling exploration only.** This is cleaner than the pilot's "9 methods but some at different budgets" mess.

## Auxiliary compute (reported but not counted)

- `bayesian_opt`: TPE density fits at each iteration. ~0.1 s per iteration on CPU. Total: ~30 s per cell.
- `successive_halving`: the multi-fidelity bracket arithmetic and rank sorting. Negligible.
- `full_evolution`, `mutation_only`, `crossover_only`: population breeding, selection, mutation, crossover. Negligible.
- `coordinate_descent`: nothing auxiliary.
- `random_search`: nothing auxiliary.

All "auxiliary compute" is CPU-side, measured in seconds per cell, and dwarfed by the LLM call cost (which is measured in hours per cell).

## Selection rule for the champion

All methods:
1. During search, each genome is evaluated on the train/search set; fitness = mean metric over the 80 search items.
2. The champion is the argmax of search-set fitness over all genomes ever evaluated during the run.
3. The champion is then evaluated on the test set (40 or 80 items) — this is the reported final score.

**Ties are broken by (a) first-evaluated, (b) lexicographic hash of the genome.** Deterministic given seed.

## Normalizations reported in the paper

For every pairwise method comparison, we report three normalizations:

1. **LLM-call budget normalization** (primary): all methods at 300 calls, as defined above. This is what the confirmatory analysis uses.
2. **Wall-clock normalization** (secondary): each method is also run at a wall-clock cap. Methods with lower per-call cost (e.g. coordinate descent is fully sequential so concurrency is low) get fewer calls. Reported as a supplementary analysis.
3. **Token budget normalization** (secondary): instead of counting calls, count total output tokens. This is fairer for methods that happen to issue longer prompts. Reported as a supplementary analysis.

The primary claim uses normalization (1). If (2) or (3) show a different pattern, that is discussed as a sensitivity analysis in §7 of the paper.

## What we are NOT including (v1)

- **SMAC / DEHB / BOHB** — mature AutoML baselines. Interesting but: (a) adds three more implementations to debug, (b) overlaps conceptually with our TPE + SH combo, (c) the preregistration must be frozen. Deferred to v2 or included as post-hoc exploration if compute permits.
- **Reinforcement-learning-from-search-trajectories.** Belongs in learned-search paper v2.
- **LLM-guided crossover (semantic merging via an LLM call).** Breaks the "no auxiliary LLM calls" rule and balloons budget. Deferred to v2.
- **Multi-objective search** (e.g. fitness + cost Pareto). Interesting extension; not in the v1 confirmatory set.

## Frozen before first run

Once this document is committed as part of the preregistration lock, the method list **cannot be changed without a dated amendment** to the preregistration. Adding a new method mid-study invalidates the preregistered analysis for that new method.
