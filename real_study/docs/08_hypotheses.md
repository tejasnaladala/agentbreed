# Phase 9 — Hypothesis List for the Real Study

This is the substantive hypothesis list. For estimators, decision rules, multiplicity correction, and power analysis, see `06_analysis_plan.md` — this document focuses on **what we are testing and why**.

## Primary hypotheses (confirmatory)

### H1 — Dimensionality effect

**Informal:** Searching over multiple agent-configuration axes beats searching over only the prompt.

**Formal:** Across the three benchmarks and two models, the mean paired difference `full_evolution − prompt_only_evolution` on the test set is negative (full wins) and its 95% LMM confidence interval has its upper bound below `−0.05`.

**Why this is the right target:** This is the single most practically relevant question for agent-optimization practitioners. If H1 fails, the entire "search beyond prompts" literature needs to be re-examined.

**Pilot prior:** the synthetic pilot found a very large effect (d_z = 1.32 pooled). We expect the real effect to be smaller — possibly d_z ≈ 0.5 — because real LLMs have more inherent structure that the synthetic landscape could not capture.

---

### H2 — Operator null (equivalence)

**Informal:** Fancy evolutionary operators (crossover, mutation, TPE, Hyperband, coordinate descent) are practically equivalent to random search at matched budget once the search space is rich.

**Formal:** All 21 pairwise comparisons within the set `{full_evolution, mutation_only, crossover_only, random_search, coordinate_descent, bayesian_opt, successive_halving}` satisfy TOST equivalence with margin `Δ_eq = 0.03`, Holm-corrected at α = 0.05.

**Why this is the right target:** this is the paper's most surprising candidate finding. If H2 holds, it has concrete implications for practitioners (don't waste effort on fancy operators; invest in search-space design). If H2 fails, the paper reports which operators actually do differ and in what direction.

**Pilot prior:** the synthetic pilot showed the null holds directionally but with a 0.054 gap on `full vs crossover_only` that exceeded the preregistered 0.03 margin. Under proper equivalence testing, that would be inconclusive, not confirmed. The real study is designed to give a clean answer one way or the other.

---

### H3 — Epistasis mechanism

**Informal:** The fitness landscape of agent configurations has meaningful two-way gene interactions — i.e., the fitness contribution of one gene depends on another gene's value.

**Formal:** Sobol decomposition at `n_base = 2048` on the 9-dimensional genome space, computed on real-LLM fitness on at least 2 of the 6 `(benchmark, model)` cells, yields `Σ_{i<j} S_{ij} ≥ 0.10` with a valid (≤ 1.05) total-variance decomposition.

**Why this is the right target:** H3 is the **mechanistic explanation** for H1. If the landscape has no pairwise interactions, dimensionality helps only via additive contributions, in which case the H1 effect should be flat in K (no interaction surplus beyond the sum of gene marginals). If H3 holds, it explains why multi-component search can uncover configurations that single-component search cannot.

**Pilot prior:** the pilot's Sobol at n=512 was numerically broken. The real study's n=2048 should give a valid decomposition. The author prior is P(H3 confirmed) ≈ 0.50 — genuinely uncertain.

## Secondary hypotheses (exploratory)

### S1 — Cross-model transfer

**Informal:** A champion from one model evaluated on a different model retains at least half of its improvement over random search.

**Formal:** `R_{l→l', b} ≥ 0.50` on ≥ 5 of the 6 `(source_model, target_model, benchmark)` combinations, where R is defined in `06_analysis_plan.md` §2.

**Why this is the right target:** if champions transfer across models, it means agent configurations encode model-independent scaffolding; if they don't, optimization is model-specific. Both answers are scientifically interesting; the status quo in the literature is ambiguous.

### S2 — Cross-benchmark transfer

**Informal:** A champion from one benchmark does NOT transfer to a different benchmark.

**Formal:** `R_{b→b', l} < 0.30` on ≥ 5 of 6 combinations.

**Why this is the right target:** the pilot already suggested this strongly. Confirming it on real data matters because it changes how practitioners should think about deploying agents: you cannot reuse the same genome across task families. Note the direction: we are testing that the champion does NOT transfer. This is a **positive prediction** of non-transfer, which requires a different framing than H2's equivalence test — we use the retention metric directly.

### S3 — Monotonic dimensionality

**Informal:** As we add more searchable gene axes, the attainable test fitness is non-decreasing.

**Formal:** Over the dimensionality sweep `K ∈ {1, 2, 3, 5, 7, 9}`, the fitted isotonic regression of `full_evolution`'s test fitness on K has RSS within 10% of an unconstrained linear fit.

**Why this is the right target:** non-monotonicity would be evidence that the search space has local overfitting or that some genes actively hurt. The null expectation is monotonicity; finding a violation would be highly surprising and worth investigating.

### S4 — Category-specific effects (new)

**Informal:** Different gene categories contribute unequally; specifically, the semantic category (system_prompt, decomposition_style, self_critique) contributes more to fitness than the compute-budget category (temperature, prompt_token_budget).

**Formal:** With semantic genes frozen at defaults, `full_evolution`'s test fitness drops by more than when compute-budget genes are frozen at defaults, on at least 4 of 6 cells.

**Why this is the right target:** practitioners want to know "where should I invest my agent-design effort?" If semantic genes dominate, the answer is "rewrite your scaffolds." If compute-budget genes dominate, the answer is "tune temperature."

### S5 — Wall-clock vs LLM-call budget ordering

**Informal:** The method ranking under LLM-call budget is approximately the same as under wall-clock budget.

**Formal:** The Spearman rank correlation between method rankings under the two normalizations is ≥ 0.90 on ≥ 4 of 6 cells.

**Why this is the right target:** reviewers will ask "what if you had used wall-clock?" We answer it preemptively.

## Dropped hypotheses (from pilot)

- **"Three domains with independent interactions"** — dropped because the synthetic domains shared a heuristic vocabulary. The real study uses three genuinely different task families with different reasoning requirements.
- **"Transfer matrix asymmetry"** — the pilot's claim that forecasting has a "wide basin" and coding has a "narrow" one was a pilot artifact. The real study reports the transfer matrix without making basin-shape claims.
- **"SOTA comparison"** — never claimed anyway; explicitly out of scope.
- **"Champions generalize to production"** — outside the scope of this paper.

## Author prior beliefs (recorded for calibration)

Before running the real study:

- P(H1 confirmed) = **0.65**  (pilot effect was large, but real LLMs may have more structure)
- P(H2 confirmed) = **0.40**  (equivalence margin of 0.03 is tight; 21 pairwise tests is a strong correction)
- P(H3 confirmed) = **0.50**  (landscape structure is genuinely unknown at real-LLM fidelity)
- P(all three primary confirmed) = **0.18**
- P(H1 confirmed, H2 rejected) = **0.35**  (this would be the "full evolution does win on real data" outcome)
- P(H1 rejected) = **0.15**  (this would be a big surprise; mostly comes from variance inflation on real benchmarks)
- P(S1 confirmed, transfer across models works) = **0.55**
- P(S2 confirmed, no cross-benchmark transfer) = **0.70**

These priors are recorded before the lock to assess calibration post-hoc. They are author judgment, not formal elicitation.

## What a clean paper looks like under different outcomes

| Scenario | Paper framing |
|---|---|
| H1 ✓, H2 ✓, H3 ✓ | "Dimensionality dominates operators, with epistatic mechanism." Main-track credible. |
| H1 ✓, H2 ✓, H3 ✗ | "Dimensionality dominates operators; we cannot attribute this to pairwise epistasis." Still strong. Main-track marginal, E&D certain. |
| H1 ✓, H2 ✗, H3 ✓ | "Full evolution wins, and the landscape has epistasis — here is which operators help most." Different paper (more about operator measurement), E&D certain. |
| H1 ✓, H2 ✗, H3 ✗ | "Multi-component search helps, operators matter, and here is what we measured." Solid E&D paper. |
| H1 ✗, H2 ✓, H3 varies | "Surprisingly, multi-component search does NOT beat prompt-only on real benchmarks." **Best negative-result paper.** Main-track candidate if the null is tight. |
| H1 ✗, H2 ✗ | "Operator choice matters, and dimensionality does not — here is what we found." Less clean, but honestly a contribution. |
| All three rejected | "The pilot's finding does not replicate on real data." Workshop / arXiv-only contribution, paper pivots to a methodology paper about preregistered evaluation. |
