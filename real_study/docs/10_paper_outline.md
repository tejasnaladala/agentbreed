# Phase 12 — Paper Outline (NeurIPS 2026 E&D primary, main-track backup)

## Title (placeholder)

**"Which components of agent configuration search actually matter? A preregistered evaluation on real benchmarks and open-weight LLMs"**

Alternative titles to consider after results:
- "Search space or search operator? A preregistered study of agent configuration optimization" (neutral)
- "Operators do less than expected: a preregistered evaluation of agent configuration search" (if H2 holds)
- "Dimensionality matters, operators don't: new evidence from real LLM agents" (if both H1 ✓ and H2 ✓ with strong margins)

## Abstract (3 paragraphs, ~300 words)

Paragraph 1: **What we measured.** We ran a preregistered evaluation of 8 agent-configuration search methods — full evolution, mutation-only, crossover-only, random search, coordinate descent, Bayesian optimization (TPE), successive halving, and prompt-only evolution — on three real benchmarks (ForecastBench, LiveCodeBench v6, GPQA Diamond) with two open-weight models (Qwen3-32B and Llama-3.3-70B). 720 confirmatory runs at matched compute, 15 seeds per cell, preregistered decision rules, linear mixed-effects aggregation.

Paragraph 2: **What we found.** [filled in after runs — TEMPLATE:]
- H1 (dimensionality): [confirmed/mixed/rejected] with contrast = [N ± M]
- H2 (operator null, equivalence-tested): [confirmed/partial/rejected] across [K/21] pairs
- H3 (epistasis mechanism): [confirmed/inconclusive/rejected]
- S1 (cross-model transfer): [outcome]
- S2 (cross-benchmark non-transfer): [outcome]

Paragraph 3: **Why it matters.** State the practical prescription, hedged to the evidence. Release code, preregistration, per-run logs, and analysis scripts.

## 1. Introduction (~1.5 pages)

- **Motivation:** LLM agents have many configurable components. Agent-optimization systems report gains from evolving multiple components. But nobody has cleanly separated "better operator" from "bigger search space" from "more budget."
- **The question we answer:** When you spend compute on agent configuration search, what is actually doing the work?
- **Our contribution:**
  1. A preregistered, contamination-controlled evaluation on real benchmarks and real open-weight models.
  2. Equivalence-testing for operator comparisons (not the misleading "fail to reject null" logic common in the literature).
  3. A controlled dimensionality sweep that isolates search-space size as an independent variable.
  4. A variance decomposition of the fitness landscape that makes the epistasis claim testable rather than asserted.
  5. Full per-run release: code, preregistration, snapshots, logs.
- **Scope:** three task families × two model scales. We do NOT claim generality beyond this.

## 2. Related work (~1 page)

- Evolutionary prompt optimization: EvoPrompt, PromptBreeder, DSPy/GEPA.
- Multi-component agent search: AgentSquare, EvoAgentX.
- Self-improving agents: Darwin-Gödel Machine, AlphaEvolve, FunSearch.
- Classical NAS negative results: Regularized Evolution, AutoML-Zero.
- AutoML baselines for reference: TPE, Hyperband, SMAC.
- What makes our paper different: all prior work either runs on synthetic or benchmarks-of-convenience, does not preregister, or does not separate space from operator.

## 3. Preliminaries (~1 page)

### 3.1 Agent configuration as a typed genome
9-gene schema in four categories. Legal value sets per benchmark. See §Appendix for full schema.

### 3.2 The search space at hand
~2 × 10^6 legal configurations per benchmark. Large enough that exhaustive is infeasible.

### 3.3 The question, formalized
"Given a fixed LLM-call budget B, does the choice of search operator matter, or is it the search space dimensionality K that dominates?"

### 3.4 Notation
(cell, paired difference, LMM contrast, TOST margin, Sobol indices)

## 4. Methods (~1.5 pages)

### 4.1 Benchmarks
- ForecastBench (frozen snapshot, time-split, 80/80)
- LiveCodeBench v6 (post-training-cutoff filter, 80/40)
- GPQA Diamond (random split seed 42, 80/80)
- Contamination protocol (probe + drop)

### 4.2 Models
Qwen3-32B (primary mid), Llama-3.3-70B (primary large). Inference via vLLM, matched compute budgets.

### 4.3 Search methods (8)
Full evolution, mutation-only, crossover-only, random search, coordinate descent, TPE, successive halving, prompt-only evolution. All at `population_size × generations = 300` LLM calls, matched budget.

### 4.4 Preregistration
Locked 2026-04-10 at `preregistration_real_v1.md`. Decision rules for H1, H2, H3, S1, S2 specified in the document.

### 4.5 Statistical protocol
Mixed-effects LMM for aggregate inference; TOST for operator equivalence; Holm-Bonferroni correction; Saltelli-Sobol for variance decomposition.

## 5. Main results (~2 pages)

### 5.1 H1 — Dimensionality effect
LMM contrast, 95% CI, interpretation against the preregistered decision rule.
Table 1: per-(benchmark, model) paired difference full vs prompt-only.
Figure 1: forest plot of paired differences.

### 5.2 H2 — Operator null (equivalence)
21 pairwise TOST results, Holm-corrected.
Table 2: full pairwise equivalence matrix.
Figure 2: pairwise equivalence bounds visualized.

### 5.3 H3 — Epistasis mechanism
Sobol first-order, total-order, pairwise indices per `(benchmark, model)`.
Table 3: Sobol decomposition per cell.
Figure 3: `Σ S_{ij}` vs benchmark; bar chart with validity check annotations.

### 5.4 Dimensionality sweep (E_decisive)
Monotonicity test for isotonic fit.
Figure 4: test fitness vs K for full_evolution and random_search on both benchmarks × both models. **Candidate headline figure.**

## 6. Transfer and scaling (~1 page)

### 6.1 Cross-model transfer (S1)
Retention metric values, 6 combinations, pass/fail against S1 threshold.
Table 4: retention matrix.

### 6.2 Cross-benchmark transfer (S2)
Retention metric values across benchmarks.
Table 5: cross-benchmark matrix.

### 6.3 Model-scale scaling (S3 optional add-on)
If Qwen3-14B is in the matrix: how does the dimensionality effect scale with model size?

## 7. Budget and cost analyses (~0.5 pages)

### 7.1 Budget-scaling curves
Test fitness vs `B ∈ {100, 300, 1000}` for the three most informative methods.

### 7.2 LLM-call vs wall-clock normalization
Rank correlation between method ranking under the two normalizations.

## 8. Discussion (~1 page)

Practical takeaways, phrased to match whatever the data shows. Sample:
- "Invest in genome diversity, not operator novelty, up to at least K=9."
- "At matched budget, random search is a reasonable default baseline; fancy operators should justify their complexity against it."
- "Cross-benchmark champions don't transfer — deploy per-task-family."

If results are mixed: lean into the nuance. "Operators matter for coding but not for forecasting." "H2 holds for fp16 but not for AWQ."

## 9. Limitations (~0.75 pages)

Be exhaustive. Explicitly list:
- Three benchmarks is a meaningful but narrow slice of real-world tasks.
- Two models × two scales is not the full scaling picture.
- Synthetic pilot results are referenced only as prior work.
- Decoding stochasticity at non-zero temperature is an additional variance source.
- Equivalence margin of 0.03 is author-chosen; sensitivity analysis at 0.02 and 0.05 reported in appendix.
- H3 Sobol indices are estimates; their CI is not computed at `n_base = 2048`.
- No real external tool use in v1; belongs in v2.
- `coordinate_descent` is new to the preregistered set in v1 (not in the original pilot's preregistration); labeled explicitly.

## 10. Reproducibility (~0.5 pages)

- Git SHA of the main experiment at paper submission.
- Preregistration SHA at lock time.
- Snapshot SHAs for all benchmarks.
- vLLM version, Python version, key library versions.
- Docker image for the analysis pipeline.
- Slurm scripts committed in `real_study/harness/slurm/`.
- Per-run JSON logs in the released artifact bundle.

## 11. Release artifact

- Tar.gz of `real_study/results/` (≈ 2 GB compressed).
- Slurm scripts, Python harness, stats scripts, analysis notebooks.
- README with "how to reproduce every table and figure."

## Appendices

- A. Full genome schema table.
- B. Per-cell results (all 720 main-matrix cells, mean + SD + n).
- C. Sobol details (first-order, total-order, pairwise, validity).
- D. Preregistration verbatim.
- E. Amendment log (if any).
- F. Compute accounting.
- G. Qualitative examples: best and worst-performing configurations per benchmark.

## Length budget

NeurIPS main track: 9 pages body + unlimited refs/appendix.
NeurIPS E&D: 9 pages body + unlimited refs/appendix (same page limit as of 2025).
Budget:
- §1 Intro: 1.5 pages
- §2 Related: 1 page
- §3 Preliminaries: 1 page
- §4 Methods: 1.5 pages
- §5 Main results: 2 pages
- §6 Transfer/scaling: 1 page
- §7 Budget analyses: 0.5 pages
- §8 Discussion: 1 page
- §9 Limitations: 0.75 pages (larger than usual — intentional)
- §10 Reproducibility: 0.5 pages
Total: ~10.75 pages before cutting. Needs a light edit to fit 9 pages.

Cuttable content:
- Related work can compress to 0.75 pages.
- §7 budget analyses can move to appendix.
- §6.3 scaling can move to appendix if Qwen3-14B is not included.

Target after edits: 9 pages body, 5–8 pages appendix.
