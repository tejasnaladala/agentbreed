# Phase 7 — Statistical Analysis Plan

This is the **preregistered analysis plan**. Every analysis described here is **confirmatory**; anything not described here is **exploratory** and will be labeled as such in the paper. No post-hoc modifications of the estimators, cell definitions, or decision rules are permitted without a dated amendment to the preregistration.

## 0. Notation

- `M` = set of methods (|M| = 8 confirmatory).
- `B` = set of benchmarks (|B| = 3): {ForecastBench, LiveCodeBench_v6, GPQA_Diamond}.
- `L` = set of models (|L| = 2): {Qwen3-32B, Llama-3.3-70B}. (Qwen3-14B optional, not in confirmatory.)
- `n_seeds = 15` seeds per `(m, b, l)` cell.
- `y_{m,b,l,s}` = the test-set mean fitness of method `m` on benchmark `b` with model `l` and seed `s`.
- `Δ_{m→m', b, l, s} = y_{m, b, l, s} - y_{m', b, l, s}` = paired difference between two methods at the same `(b, l, s)`.

**Unit of analysis for the primary tests:** the `(benchmark, model, seed)` cell. Pair the two methods being compared at that cell to get a single difference.

**Seeds are paired across methods.** The same `(b, l, s)` triple gives one `y` per method; thus `Δ` is always a matched pair. The randomness is in population initialization, crossover/mutation, train/test split (no — splits are frozen across seeds).

## 1. Primary hypotheses

(See Phase 9 for the full hypothesis list; this section gives estimators and decision rules.)

### H1 — Dimensionality effect

**Statement.** Across all benchmarks × models, the paired difference between `full_evolution` and `prompt_only_evolution` has a mean that exceeds a practical threshold.

**Practical threshold.** `Δ_min = 0.05` on the 0–1 fitness scale. A benefit of less than 5 percentage points (absolute) is not considered practically meaningful.

**Primary estimator (confirmatory).** A linear mixed-effects model (LMM) fit via REML:

```
y_{m,b,l,s} ~ method + (1 | benchmark) + (1 | model) + (1 | benchmark:model) + (1 | benchmark:model:seed)
```

Where:
- `method` is a fixed effect (categorical, baseline = `full_evolution`).
- `benchmark`, `model`, and their interaction are crossed random intercepts.
- The seed-within-cell random intercept accounts for paired structure.

**Fit with:** `statsmodels.MixedLM.from_formula(...).fit(reml=True, method='lbfgs')`.

**Decision rule for H1:**
- Extract the fixed-effect contrast `β_{prompt_only_evolution} - β_{full_evolution}` (so negative = `full` beats `prompt_only`).
- Compute the 95% Wald CI of this contrast.
- **H1 is confirmed if:** the contrast is negative and the upper end of its 95% CI is below `-0.05` (i.e. the full-minus-prompt-only gap is ≥ 0.05 with 95% confidence).
- **H1 is mixed if:** the contrast is negative and the upper CI is between `-0.05` and `0.0`.
- **H1 is rejected if:** the upper CI is above `0.0`.

**Supplementary per-cell analysis** (for reporting, not for the decision rule): paired bootstrap (10,000 resamples, resample seeds within cell) for each `(b, l)` cell. Report the 95% CI of the mean paired difference. Holm-Bonferroni correction across the `|B| × |L| = 6` per-cell tests.

### H2 — Operator null (equivalence)

**Statement.** For each pair of methods in `{full_evolution, mutation_only, crossover_only, random_search, coordinate_descent, bayesian_opt, successive_halving}`, the mean paired difference is bounded in absolute value by a practical-equivalence margin.

**Practical-equivalence margin.** `Δ_eq = 0.03` on the 0–1 fitness scale. We consider two methods "practically equivalent" if their mean paired difference is within ±0.03.

**Primary estimator (confirmatory).** Two one-sided tests (TOST) on the paired differences from the LMM fit above.

For each pairwise comparison `(m, m')`:
- Fit the LMM with `method` as a categorical fixed effect.
- Extract the contrast `β_m - β_{m'}` and its standard error `SE`.
- Compute two one-sided t-statistics:
  - `t_lower = (β_m - β_{m'} - (-Δ_eq)) / SE`
  - `t_upper = (β_m - β_{m'} - Δ_eq) / SE`
- Equivalence is established if both `t_lower > t_crit(1 - α)` and `t_upper < -t_crit(1 - α)` at `α = 0.05`.
- Equivalently, check that the 90% CI of the contrast is fully within `[-Δ_eq, +Δ_eq]`.

**Multiple-comparison correction.** H2 has `C(7, 2) = 21` pairwise comparisons. Apply **Holm-Bonferroni** across all 21 TOST p-values (we use `max(p_lower, p_upper)` as the pair's p-value for Holm adjustment).

**Decision rule for H2:**
- **H2 is confirmed** if all 21 pairwise tests establish equivalence at Holm-corrected α = 0.05.
- **H2 is partially confirmed** if ≥ 16/21 pairs are equivalent.
- **H2 is rejected** if fewer than 16/21 pairs are equivalent, OR any single pair has a 95% CI whose absolute endpoints exceed 0.05.

**Important correction from the pilot:** we do NOT claim H2 is confirmed just because the paired difference is "not significantly different from zero" — failure to reject H0 is not equivalence. We explicitly use TOST, which is the right test for "practically equivalent."

### H3 — Epistasis mechanism

**Statement.** At least 10% of fitness variance in the landscape is explained by two-way gene interactions.

**Estimator.** Saltelli-Sobol variance decomposition at `n_base = 2048` per `(benchmark, model)` cell using a 9-dimensional input space (the full genome).

For each `(b, l)` cell:
- Sample a base Sobol sequence of `n_base = 2048` points in `[0, 1]^9`.
- Construct the `A`, `B`, `AB_i` matrices per Saltelli's estimator.
- Evaluate the agent fitness on the search set for each point (80 train items per sample, same as main method cost; total Sobol calls per cell ≈ 2048 × 2 + 9 × 2048 × 2 ≈ 45,056 evaluations).
- Compute first-order indices `S_i` and total-order indices `S_{T,i}` via the standard estimator.
- Compute pairwise indices `S_{ij}` via a second-order Saltelli estimator (or, if computationally prohibitive, estimate the sum `Σ_{i<j} S_{ij}` via `Σ_i S_{T,i} - Σ_i S_i`, which equals the sum of all higher-order interactions and is an upper bound on pairwise interactions; we report both).
- **Validity check:** assert `Σ_i S_i + Σ_{i<j} S_{ij} + Σ_{i<j<k} S_{ijk} + ... ≤ 1.05`. If the check fails, the decomposition is invalid and H3 is marked **untestable at this sample size**; we either re-run at `n_base = 4096` or drop H3 from confirmatory.

**Decision rule for H3:**
- **H3 is confirmed** if `Σ_{i<j} S_{ij} ≥ 0.10` on at least 2 of `|B| × |L| = 6` cells, AND the validity check passes on at least those 2 cells.
- **H3 is rejected** if `Σ_{i<j} S_{ij} < 0.10` on ≥ 4 cells (and validity checks pass).
- **H3 is inconclusive** otherwise.

**Note on pilot H3 failure.** The pilot ran at `n_base = 512` which produced `Σ S > 1.0`, making the decomposition invalid. At `n_base = 2048` the variance of the Saltelli estimator is ~4× smaller; we expect validity but will test it explicitly before claiming confirmation.

**Cost.** H3 costs 2 `(b, l)` cells × 45,056 evaluations = ~90,000 additional evaluations per the 2-of-6 cell claim. This is about 4 GPU-days on 70B, 2 on 32B. Budget-justified.

## 2. Secondary hypotheses

### S1 — Cross-model transfer

**Statement.** The champion from model `l` on benchmark `b` retains at least 50% of its improvement over random search when evaluated on a different model `l'` on the same benchmark.

**Retention metric:**

```
R_{l→l', b} = (y_{champion(l), b, l'} - y_{random, b, l'}) / (y_{champion(l), b, l} - y_{random, b, l})
```

Where `y_{champion(l), b, l'}` is the champion of model `l` evaluated on model `l'` on benchmark `b`'s test set, and `y_{random, b, l}` is the mean test-set score of `random_search` on `(b, l)` across seeds.

**Decision rule.**
- **S1 confirmed** if `R ≥ 0.5` on ≥ 5 of 6 `(l→l', b)` combinations (taking the mean over 5 transfer seeds).
- **S1 rejected** if `R < 0.5` on ≥ 3 of 6 combinations.

**Note:** the retention metric is sign-sensitive. If `y_{champion} - y_{random}` is near zero on the source cell, `R` is ill-defined; we flag and report raw fitness instead.

### S2 — Cross-benchmark transfer

**Statement.** The champion from benchmark `b` on model `l` does NOT transfer well to a different benchmark `b'` on the same model (retention metric below 0.30).

**Retention metric:** same as S1 but across benchmarks.

**Decision rule:**
- **S2 confirmed** if `R < 0.30` on ≥ 5 of 6 `(b→b', l)` combinations.
- **S2 rejected** otherwise.

### S3 — Monotonic dimensionality

**Statement.** Test fitness of `full_evolution` on the dimensionality-reduced templates is non-decreasing in `K` for all `K ∈ {1, 2, 3, 5, 7, 9}` on at least one benchmark × model cell.

**Estimator.** Per cell, fit `y(K) ~ K` with monotonicity enforced via an isotonic regression. Compare to an unconstrained linear fit with a likelihood-ratio test. If the monotonic fit has RSS within 10% of the linear fit, we declare "non-decreasing."

**Decision rule:**
- **S3 confirmed** if monotonicity holds (as above) on ≥ 4 of `|B| × |L| = 6` cells.
- **S3 rejected** otherwise.

## 3. Multiplicity correction

### Family-wise rate across hypotheses

The three primary hypotheses are tested at `α = 0.05` **each**, with no additional correction across hypotheses. Rationale: they test distinct substantive questions (dimensionality, operator equivalence, epistasis) and are not redundant.

### Within-hypothesis correction

- **H1:** one primary LMM contrast (no correction needed for the confirmatory rule). The supplementary per-cell bootstrap analysis uses Holm-Bonferroni across 6 cells.
- **H2:** 21 pairwise TOST tests, Holm-Bonferroni.
- **H3:** 6 cells, each tested independently against the 0.10 threshold. No Holm needed for threshold tests; the decision rule is already "at least 2 of 6" which is the correction.

### Cross-hypothesis

Not applied. H1, H2, H3 are distinct substantive hypotheses; correcting across them would over-correct.

## 4. Power and sensitivity analysis

Prior to runs, we fix:
- `n_seeds = 15`
- `|B| × |L| = 6` cells per comparison
- Observed cell-level paired fitness SD (from pilot) ≈ 0.05 on forecasting, ≈ 0.10 on coding, assumed 0.08 on GPQA Diamond.

### H1 power

With 15 seeds paired across 6 cells (90 paired observations) and cell-level SD in range [0.05, 0.10], the LMM contrast has ~95% power to detect a mean paired difference of 0.05 (the practical threshold) at α = 0.05.

### H2 power (equivalence)

TOST at α = 0.05 with SD ≈ 0.05, `n = 90` paired: we can establish equivalence at ±0.03 with power 0.80 if the true difference is ≤ 0.01. This is tight — pushing `Δ_eq` to 0.03 rather than 0.05 is deliberate because 0.05 is too loose to be a "null result."

### H3 power

Sobol at `n_base = 2048` gives variance-index estimates with ~5% relative error for indices of magnitude ≥ 0.05. We can detect `Σ_{i<j} S_{ij} ≥ 0.10` with confidence if the true value is ≥ 0.12.

## 5. Exploratory analyses (labeled as such in the paper)

- Per-benchmark breakdowns (not cross-benchmark LMM).
- Sensitivity to seed count (re-run main LMM with seeds 5, 10, 15 and check stability).
- Budget-scaling curves (main matrix methods at `B ∈ {100, 300, 1000}`).
- Category-ablations (freeze one genome category at defaults, re-run 3 main methods per cell).
- Learning-curve heatmaps (per-generation fitness trajectory).
- Failed-run forensics (qualitative review of runs where the agent outputs parse errors > 10%).
- Correlation between `S_{T,i} - S_i` (unexplained-by-first-order variance) and gene categories.

**None of these modify the primary decision rules.** They are reported in the paper's appendix as exploratory.

## 6. What we will report for every primary comparison

For each preregistered pairwise comparison, the paper reports:

1. Mean paired difference `Δ̄` (LMM fixed effect).
2. Standard error from the LMM.
3. 95% Wald CI of the difference.
4. Cohen's `d_z` computed as `Δ̄ / SD(Δ)` using the residual SD from the LMM.
5. For H2: TOST p-value at `Δ_eq = 0.03`, Holm-corrected p-value.
6. Raw per-cell means (for reference) in the appendix.

## 7. Implementation

All analysis is a single reproducible script: `real_study/harness/stats/analysis.py`. Inputs: the `results/` directory of per-run JSONs. Outputs: `results/analysis_real_v1.json` and LaTeX-ready tables.

The script has unit tests on synthetic data with a known `(β, SE)` that verify the LMM contrast extraction and the TOST implementation are correct. Unit tests must pass before the script is used for the confirmatory analysis.

## 8. What invalidates the analysis plan

The analysis plan is considered "followed" iff:
- All listed estimators are computed as defined.
- No additional methods or benchmarks are added post-hoc.
- No seeds are dropped without explicit documentation of the reason.
- The LMM converges on the primary fit; if it fails to converge, we fall back to a fixed-effects OLS with clustered SEs and document the fallback.
- All data is logged atomically per-run before the analysis is run.

Any deviation from this plan, at any point after the preregistration is locked, must be documented as a dated amendment in `preregistration_real_v1.md` and the affected analysis labeled "post-hoc."
