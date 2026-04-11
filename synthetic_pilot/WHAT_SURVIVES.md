# What Survives the Pilot → Real Study Handoff

Components from the `../paper/` pilot that are directly reusable in `../real_study/`. Each item has a provenance tag and an explicit reuse contract.

## Code — reusable as-is

### `breed/genome/` (typed genome representation)
- **Why it survives:** the typed-genome abstraction (TEXT, ENUM, FLOAT, SET, NUMERIC_VECTOR with constraint metadata, lineage, content hash) is orthogonal to whether the agent is synthetic or real. The 431 unit tests cover this code.
- **Reuse contract:** the real-study harness imports `breed.genome.*` unchanged. No behavioral delta.
- **Reviewer risk:** none.

### `breed/population.py` (tournament + elitism + breeding)
- **Why it survives:** the infinite-loop bug (elites collapsed to one identity) was fixed in commit `dd364ca`. Behavior is correct for the generic genome case.
- **Reuse contract:** unchanged. Real-study methods call `Population.breed(...)` directly.
- **Reviewer risk:** low — covered by unit tests.

### `breed/analysis/epistasis.py` + `breed/analysis/schema_preservation.py` + `breed/analysis/landscape.py`
- **Why it survives:** these are generic landscape-analysis utilities (Sobol indices, fitness-distance correlation, PCA-based effective dimensionality, typed Hamming distance). They operate on genomes + fitness arrays, not on any specific agent implementation.
- **Reuse contract:** used for post-hoc landscape characterization in the real study. Sobol must be run at `n_base ≥ 2048` per the preregistered protocol, not 512.
- **Reviewer risk:** **the Sobol estimator had numerical issues at n_base = 512 in the pilot.** Before reusing, add a unit test that asserts `sum(first_order_indices) + sum(pairwise_indices) ≤ 1.05` on a known-additive synthetic landscape. If that test fails, the estimator itself is buggy, not just under-sampled.

### Baseline implementations (`breed/experiment.py`: random_search, static_best, static_ensemble, bayesian_opt, successive_halving, coordinate_descent)
- **Why they survive:** generic search loops with no synthetic-specific assumptions. `bayesian_opt` uses TPE over the encoded genome, `successive_halving` uses η=3 with fidelity floor 5.
- **Reuse contract:** pulled into `real_study/harness/search/` via thin wrappers that accept a real-LLM `agent_fn`.
- **Reviewer risk:** **coordinate_descent was added post-hoc and is not in the original preregistration.** If the real study includes it, the new preregistration must list it explicitly in the methods set from day one.

### Test infrastructure (`tests/` — 431 tests)
- **Why it survives:** high-coverage unit tests for all of the above. The synthetic-agent-specific tests are clearly separated and will be excluded.
- **Reuse contract:** `real_study/harness/tests/` adds new tests for real-LLM-specific concerns (rate limiting, schema parsing, cost accounting). Pilot tests continue to run for `breed.*`.

### `breed/datasets/builtin.py` (120 forecasting questions)
- **Why it partially survives:** only as an integration-test fixture for local runs without an LLM call. **It will not appear in the real paper as a benchmark.** The real study uses ForecastBench.
- **Reuse contract:** relabel the file as a smoke-test fixture, not a benchmark. Add a module docstring that says so.

## Methodology — partially reusable

### The paper's core question
> "When we optimize LLM agent configurations across multiple axes, is the load-bearing variable the choice of search operator, or the richness of the configuration space itself?"

This is the **right question.** It survives unchanged.

### The dimensionality-sweep experiment design (`E_decisive`)
- **Why it survives:** the controlled K ∈ {1, 2, 3, 5, 7, 9} sweep, holding everything else constant, is a clean causal design.
- **Reuse contract:** run this design on **real benchmarks × real models** in the real study. The sweep K values and per-cell seed count carry forward.

### The 9-method matrix with matched compute
- **Why it survives:** the idea of running all methods at `population_size × generations` LLM calls is the right way to compare operators fairly.
- **Reuse contract:** preserved. Budget matching is a non-negotiable in the new preregistration.

### Paired bootstrap + Holm correction per-benchmark
- **Why it survives:** appropriate for n=20 seeds. Correct multiplicity handling.
- **Reuse contract:** preserved, but now combined with a **preregistered mixed-effects model** for the cross-benchmark aggregate (which the pilot silently replaced with a pooled t-test).

## What does NOT survive

### The "preregistered main-track study" framing
Dead. The pilot was not the main-track execution, will never be called that, and the abstract + intro + reproducibility section of `../paper/05_draft/paper_main_track.md` must not appear in the real-study paper verbatim.

### H1 / H2 / H3 confirmation claims from the pilot
Dead. The pilot's "H1 confirmed at p = 1e-14" cannot be cited as evidence in the real study. The real study re-tests from scratch.

### The "3 synthetic domains" framing
Dead. The synthetic coding and knowledge domains share a forecasting-flavored heuristic vocabulary, which means the pilot's three-domain independence claim is false. Any real-study reviewer inspecting `multi_domain_agent.py` would spot this.

### The Sobol-based H3 claim
Dead. The pilot's `n_base = 512` decomposition produced invalid variance shares. The real study runs Sobol at `n_base = 2048` on real data and reports H3 honestly — confirmed, mixed, or rejected, no "qualitative" mode.

### The cross-domain transfer matrix
Dead as a causal claim. The pilot's transfer matrix has an internally-inconsistent structure (coding-champion beats forecasting-champion on the forecasting test set). The real study re-derives transfer from scratch on real benchmarks and models.

### The "coordinate descent" baseline (in current form)
Restricted. The real study can include coordinate descent **only if** it is declared in the new preregistration from day one.

## Summary

About 60% of the pilot code survives (genome, population, analysis utilities, baseline search methods, test infrastructure). Zero percent of the pilot's scientific claims survive. The pilot's core question, dimensionality-sweep design, budget-matching rule, and paired-bootstrap statistical approach survive. Everything else is rebuilt from scratch.
