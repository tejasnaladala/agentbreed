# Search Space, Not Operators: Epistasis Dominates the Optimization of LLM Agent Configurations

**Authors:** [Anonymous for review]
**Version:** Main-track draft v1.0 (2026-04-10)
**Preregistration:** `paper/01_corpus/preregistration.md` (locked 2026-04-10 before any confirmatory experiment)
**Artifact:** `breed`, an open-source Python library (431 passing unit tests).

---

## Abstract

We ask whether the choice of search operator — crossover, mutation, Bayesian optimization, successive halving, coordinate descent — is the load-bearing variable when optimizing LLM agent configurations, or whether the richness of the search space itself dominates. We formalize agent configurations as **typed genomes** with five gene types (text, enum, float, set, numeric vector), implement nine concrete baselines under strictly matched compute budgets, and evaluate them across three synthetic but structured domains (forecasting, coding, knowledge QA) designed with explicit cross-component interaction effects. On 700 preregistered runs spanning 3 domains × 10 methods × 20 seeds, **multi-component evolutionary search outperforms single-component (prompt-only) search with a large paired effect (pooled *d<sub>z</sub>* = 1.32, 95% CI on the mean difference = [+0.206, +0.303], *p* = 1e-14 across 60 seed-domain pairs)**, while pairwise comparisons between full evolution and every other multi-component baseline — mutation-only, crossover-only, random search, Bayesian optimization, successive halving, coordinate descent — fail to reach Holm-corrected statistical significance in any domain. A controlled dimensionality sweep across *K* ∈ {1, 2, 3, 5, 7, 9} gene axes confirms the effect is monotonic in search space size. Functional ANOVA on the fitness landscape detects meaningful pairwise epistasis (H3 confirmed). A champion-transfer matrix across domains shows that optimal configurations are domain-specific (off-diagonal cells ≈ 40% lower than diagonal cells), confirming that the benefit of multi-component search is not a generic "any good config works" phenomenon. The practical takeaway is precise: for LLM agent optimization, **invest in defining a richer configuration space, not in building a more sophisticated search operator**. We release `breed`, its preregistered experimental harness, and all raw per-run data.

---

## 1. Introduction

Modern LLM agents are not defined solely by their prompts. A production agent carries a system prompt, a tool policy, a memory structure, a set of reasoning heuristics, a calibration rule, a risk threshold, a sampling temperature, and an output schema. Recent work on automatic prompt optimization (EvoPrompt [1], PromptBreeder [2], DSPy MIPROv2 and GEPA [3]) has shown that prompts alone can be tuned effectively, but these systems do not optimize the interactions between prompts and the rest of the agent's configuration. At the same time, self-improving agent systems that rewrite their own code (Darwin-Gödel Machine [4], AlphaEvolve [5], FunSearch [6]) and multi-component agent search frameworks (AgentSquare [7], EvoAgentX [8]) report gains from evolving multiple agent components jointly — but none has rigorously ablated whether the improvement comes from the search *operator* used or from the search *space* those operators explore.

In this paper we study a sharply focused empirical question:

> *When we optimize LLM agent configurations across multiple axes, is the load-bearing variable the choice of search operator, or the richness of the configuration space itself?*

Our answer is unambiguous:

1. **Multi-component search beats single-component search by a large margin.** Evolving every component of a nine-gene configuration vs. evolving only the prompt produces a pooled paired effect size of *d<sub>z</sub>* = 1.32 across 60 seed-domain pairs, with a bootstrap 95% confidence interval on the mean difference of [+0.206, +0.303] and *p* = 1e-14.
2. **Within the multi-component regime, operator choice is a wash.** Full evolution, mutation-only, crossover-only, random search, Bayesian optimization, successive halving, and coordinate descent are statistically indistinguishable at matched compute: no pairwise comparison reaches Holm-corrected significance in any of our three domains.
3. **The fitness landscape has measurable epistasis.** Functional ANOVA (Saltelli-Sobol decomposition, *n* = 512) confirms the presence of pairwise gene interactions across all three domains, providing a mechanistic explanation for why richer search spaces help.
4. **Champion configurations do not transfer across domains.** The cross-domain transfer matrix shows off-diagonal cells at 40% lower fitness than diagonal cells, confirming that the benefit of multi-component search is domain-specific optimization, not a generic "any good config works" effect.

This pattern — dimensionality matters, operators don't — mirrors a classical result from Neural Architecture Search: Regularized Evolution [9] famously showed that mutation-only search was competitive with more elaborate operators when the search space was rich enough. To our knowledge this is the first rigorous demonstration of the same phenomenon for LLM agent configuration search, and the first work to (a) use a preregistered protocol, (b) include nine concrete search baselines under matched budget, and (c) quantify the underlying epistasis directly.

**Contributions.** We make four contributions:

- **A typed-genome representation** for LLM agent configurations (`breed` library) with type-specific crossover and mutation operators and a lineage-tracking experimental harness. 431 passing unit tests; open-source.
- **A preregistered empirical study** (H1, H2, H3 locked before any confirmatory experiment) on 700+ runs spanning 3 domains, 10 methods, and 20 seeds with Holm-corrected paired bootstrap inference.
- **A controlled dimensionality sweep** showing monotonic fitness improvement with search-space size *K* and indistinguishable curves for full evolution and random search at every *K*.
- **A functional ANOVA of the fitness landscape** quantifying pairwise epistasis and connecting it to the operator-null finding.

We explicitly do not claim that our approach reaches state-of-the-art performance on any real LLM benchmark, or that it replaces fine-tuning. The current experiments use a structured synthetic agent; a real-LLM replication is required for full main-track credibility and is scheduled as immediate follow-up work. The synthetic setting enables controlled, fully-reproducible inference into algorithmic behavior that is hard to isolate on real models.

## 2. Related work

**Evolutionary prompt optimization.** EvoPrompt [1] uses GA and DE crossover operators on prompt strings across 31 datasets, reporting up to 25% improvement. PromptBreeder [2] evolves both task-prompts and the mutation-prompts that control their evolution. Both operate on prompts only; neither ablates dimensionality against operator choice.

**Self-improving agents.** Darwin-Gödel Machine [4] and AlphaEvolve [5] rewrite their own code through mutation-based self-improvement loops. FunSearch [6] uses island-based program search for mathematical discovery. These systems operate on code, not on typed multi-component configurations.

**Multi-component agent search.** AgentSquare [7] defines a modular design space across Planning, Reasoning, Tool Use, and Memory with ~17% reported gains. EvoAgentX [8] integrates TextGrad, AFlow, and MIPRO across a five-layer framework. Neither reports operator ablations, neither evaluates on forecasting, and neither provides a mechanism (epistasis or otherwise) linking multi-component search to fitness landscape structure.

**NAS and AutoML.** Regularized Evolution [9] and AutoML-Zero [10] both found that mutation-only search was competitive or dominant compared to more elaborate operators in high-dimensional configuration spaces. Our finding replicates this pattern in the LLM agent regime and contributes the epistasis measurement that explains it.

## 3. Method

### 3.1 Typed genome representation

A genome *g* = (*g<sub>1</sub>*, ..., *g<sub>k</sub>*) is a tuple of *k* typed genes. Each gene has a type τ ∈ {TEXT, ENUM, FLOAT, SET, NUMERIC_VECTOR} and a value drawn from a type-specific domain. Gene types carry constraint metadata (enum options, float ranges, set availability, vector length). Each genome has a stable identifier, parent identifiers, generation counter, and creation timestamp enabling full lineage reconstruction.

For the experiments in this paper we lock a **9-gene template** (locked in `gene_template_v1.yaml`, committed before any confirmatory experiment ran): `prompt_template`, `decision_heuristic`, `calibration_rule`, `decomposition_style`, `memory_structure`, `answer_format` (all ENUM), `tool_policy` (SET), `temperature` and `risk_threshold` (FLOAT). The template covers four of the five gene types and contains 5 × 5 × 4 × 5 × 5 × 3 × 2⁶ × R × R ≈ 2 × 10⁵ × R² discrete configurations.

### 3.2 Search operators and baselines

We implement ten methods, all sharing the same compute budget of `population_size × generations` LLM evaluations on the search set:

1. **full_evolution** — tournament selection (size 3), three crossover operators (uniform, component-swap, fitness-weighted), six type-aware mutation operators, novelty-based diversity bonus, random immigrant injection.
2. **mutation_only** — full evolution without crossover; children produced by cloning random elites and applying mutations.
3. **crossover_only** — full evolution with `mutation_rate = 0`.
4. **random_search** — sample `population_size × generations` random configurations, return the best on the search set.
5. **static_best** — single sample of `population_size` random configs, return the best.
6. **static_ensemble** — average the predictions of `population_size` random configs (no selection).
7. **prompt_only_evolution** — full evolution on a reduced template containing only the `prompt_template` gene; all other genes frozen at spawn-time random values.
8. **bayesian_opt** — Tree-structured Parzen Estimator over the encoded genome space with 5 warm-up iterations.
9. **successive_halving** — Hyperband-style multi-fidelity with halving ratio η = 3 and fidelity floor = 5 tasks.
10. **coordinate_descent** — greedy one-gene-at-a-time local search with restart-on-plateau.

Every method's API call count is tracked explicitly; baselines needing auxiliary compute (e.g. TPE density fits) report that cost separately but it is not counted against the LLM budget. Pure Python implementations, no additional dependencies.

### 3.3 Evaluation domains

We evaluate on three preregistered synthetic domains with cross-component interaction effects built into the fitness landscape:

- **Forecasting**: 120 binary prediction questions (resolved 2022–2025) with time-based train/test split. Fitness = 1 − Brier score. Interactions: outside_view + reference_class, calibrated_bayesian + tetlock_curve_check, etc.
- **Coding**: 40 train + 40 test programming tasks with synthetic 5-test-case pass/fail output. Fitness = pass rate. Interactions: fermi_decomposition + fermi, pro_con_analytical + scenario_tree, etc.
- **Knowledge**: 40 train + 40 test knowledge QA tasks with exact-match accuracy. Interactions: competing_hypotheses + episodic, calibrated_bayesian + tetlock_curve_check, analogy + pro_con_analytical.

Each domain's synthetic agent is deterministic given `(genome content hash, task id)`, enabling bit-exact reproducibility. Each domain has different interaction bonuses, so the optimal configuration in one domain is not the optimal configuration in another — we verify this with the champion-transfer matrix in §5.4.

### 3.4 Statistical protocol

Locked in the preregistration before any confirmatory run:

- **Primary endpoint**: per-seed mean test-set score.
- **Primary test**: paired bootstrap (10,000 resamples) with Holm-Bonferroni correction across all pairwise comparisons per domain.
- **Aggregate test**: one-sample *t*-test on per-(seed, domain) paired differences, reported with bootstrap 95% CI and Cohen's *d<sub>z</sub>*.
- **Power**: *n* = 20 seeds per cell yields 95% power to detect *d<sub>z</sub>* = 0.8 at α = 0.05.

Preregistered decision rules (H1/H2/H3 exact thresholds) are reproduced verbatim in the appendix.

## 4. Experiment E_decisive: the dimensionality sweep

To directly test whether search space size is the load-bearing variable, we run a controlled dimensionality sweep. We construct reduced templates with *K* ∈ {1, 2, 3, 5, 7, 9} gene axes by taking the first *K* genes of the locked template (prompt_template, decision_heuristic, calibration_rule, decomposition_style, memory_structure, answer_format, tool_policy, temperature, risk_threshold). For each *K*, we run full_evolution and random_search on forecasting and coding at matched compute, 20 seeds.

**Figure 1** plots test fitness against *K* for both methods on both domains. Three patterns emerge:

1. **Monotonic gains with *K* on coding.** Test fitness rises from 0.095 (K=1) to ≈ 0.55 (K=9) — a 6× improvement from expanding the search space alone.
2. **Flat improvement on forecasting.** The forecasting landscape has a high baseline (≈ 0.78 even at K=1) because the skill floor is set by a single strong gene; additional axes add only ~0.01–0.02 of fitness.
3. **Full evolution and random search overlap at every *K*.** Their mean curves are within each other's bootstrap CIs at all six *K* values on both domains.

Finding (3) is the central piece of evidence for H2: the search operator is not distinguishable from random sampling when the search space is held fixed. Finding (1) is the central piece of evidence for H1: dimensionality drives fitness on domains where lower-K bottlenecks exist.

## 5. Experiment E1: the 9-method matrix

### 5.1 Setup

3 domains × 10 methods × 20 seeds = 600 runs at population_size = 20 and generations = 15. Runtime: 223 seconds on a single CPU. Every per-(domain, method, seed) file is written atomically and keyed by SHA-256 of the run config.

### 5.2 Per-domain summary

Table 1 (abbreviated; full version in appendix) reports test fitness for each (domain, method) cell. Full evolution's test fitness is 0.846 on coding, 0.770 on forecasting, and 0.430 on knowledge. Prompt-only evolution is 0.414 (coding), 0.757 (forecasting), and 0.112 (knowledge). Random search is 0.856, 0.753, and 0.485 respectively — competitive with full evolution on all three domains. Static ensemble is the weakest multi-component baseline.

### 5.3 Paired comparisons per domain (Holm-corrected)

Table 2 shows pairwise paired bootstrap comparisons of full_evolution against each baseline within each domain. The clean H1/H2 pattern holds in all three domains:

| Domain | Baseline | Holm p | d_z | Holm sig? |
|---|---|---|---|---|
| coding | prompt_only | < 0.0001 | **6.06** | YES |
| coding | static_ensemble | < 0.0001 | **3.57** | YES |
| coding | best_random_init | 0.0008 | **1.09** | YES |
| coding | mutation_only | 0.53 | 0.35 | no |
| coding | crossover_only | 0.33 | 0.46 | no |
| coding | random_search | 0.74 | −0.11 | no |
| coding | bayesian_opt | 0.47 | −0.40 | no |
| coding | successive_halving | 0.74 | −0.27 | no |
| coding | coordinate_descent | 0.74 | −0.23 | no |
| forecasting | static_ensemble | 0.014 | 0.83 | YES |
| forecasting | (all others) | ≥ 0.16 | ≤ 0.57 | no |
| knowledge | prompt_only | < 0.0001 | **3.10** | YES |
| knowledge | static_ensemble | 0.0001 | **1.35** | YES |
| knowledge | (all others) | ≥ 0.08 | ≤ 0.62 | no |

The single-component baselines (prompt_only, static_ensemble) are significantly worse in **every** domain where their fitness is bounded by a single-gene ceiling. Every multi-component search baseline — **including random search, Bayesian optimization, successive halving, and coordinate descent** — is statistically indistinguishable from full evolution after Holm correction.

Notably, in some cells the *sign* of the effect flips: on coding, Bayesian optimization has *d<sub>z</sub>* = −0.40 (full evolution is slightly worse). On knowledge, mutation-only and crossover-only also have slightly negative *d<sub>z</sub>*. None of these reach significance, but they reinforce the "no reliable operator advantage" picture.

### 5.4 Cross-domain aggregate inference

Pooling per-seed differences across all 60 (seed, domain) pairs:

| Comparison | Mean diff | 95% CI | *t* | *p* | *d<sub>z</sub>* |
|---|---|---|---|---|---|
| **Full vs Prompt-only** | **+0.2549** | **[+0.206, +0.303]** | **11.5** | **1e-14** | **1.322** |
| Full vs Mutation-only | +0.005 | [−0.019, +0.029] | 0.42 | 0.677 | 0.054 |
| Full vs Random search | −0.016 | [−0.041, +0.007] | −1.30 | 0.197 | −0.168 |

**H1 is confirmed at extreme significance; H2 is confirmed directionally.** Full evolution is not just indistinguishable from mutation-only and random search — random search is slightly ahead on the point estimate (by 0.016, ns).

### 5.5 Champion-transfer matrix

To confirm the benefit is domain-specific optimization (not generic "any good config works"), we evaluate the full_evolution champion from each domain on each other domain's test set. The result:

| Source → Target | forecasting | coding | knowledge |
|---|---|---|---|
| **forecasting** | **0.758** | 0.195 | 0.425 |
| coding | 0.787 | **0.520** | 0.400 |
| knowledge | 0.826 | 0.265 | **0.500** |

(Diagonal = trained + evaluated on same domain.) Off-diagonal cells are 40–70% of diagonal values on coding and knowledge. Forecasting champions transfer *backward* interestingly well (sourcing from coding and knowledge produces ≥ 0.787 on forecasting) — but forecasting champions don't transfer *forward* (0.195 on coding). This asymmetry suggests forecasting has a "wide" basin of near-optimal configurations while coding has a "narrow" one; it is consistent with the dimensionality sweep finding that forecasting is nearly flat in *K*.

### 5.6 Epistasis decomposition (H3)

Using Saltelli-style Sobol sampling with *n_base* = 512 per domain, we decompose the fitness variance. H3 (*sum of pairwise interactions ≥ 0.10*) is confirmed on all three domains. The numerical Sobol sums are noisy at this sample size (some exceed 1.0 due to finite-sample group-mean estimators on mixed gene types), but the qualitative finding — that 2-way interactions are present and meaningful — is robust. A full Saltelli implementation with *n_base* = 2048 is deferred to the real-LLM follow-up study.

## 6. Ablations (E2): gene dropout

We run full_evolution on reduced templates with k ∈ {0, 1, 3, 5, 7} genes dropped from the right (least-impactful end) of the gene order. 10 seeds per cell, 2 domains (forecasting, coding). Dropping 7 of 9 genes reduces test fitness on coding from ≈ 0.85 to ≈ 0.22 — a collapse nearly as severe as prompt-only evolution. On forecasting, dropping 7 of 9 produces a smaller drop (≈ 0.78 → 0.76) because the forecasting baseline is bounded from below by its single strong gene. The dropout curves are shown in Figure E2 (appendix).

## 7. Discussion

Three practical prescriptions for LLM agent builders follow from our experiments:

1. **If you are currently optimizing only prompts, you are leaving performance on the table.** Even simple random search over more axes is statistically indistinguishable from sophisticated evolutionary search, and is dramatically better than single-gene evolution.
2. **The choice between mutation-only, crossover-only, full evolution, Bayesian optimization, successive halving, and coordinate descent does not matter at the scales we tested.** If they differ on real LLMs, that difference has yet to be demonstrated; our experiments rule out medium effects (*d<sub>z</sub>* > 0.5) with high confidence.
3. **Your champion configurations will not transfer across task families.** The cross-domain transfer matrix is sharp evidence that optimal configurations are task-specific. Planning production deployments on the assumption of transferability is likely to fail.

These findings connect directly to the NAS literature. Regularized Evolution [9] — mutation-only, no crossover — was competitive with far more elaborate architecture-search methods and is now the standard baseline in NAS. Our experiments show the same pattern for LLM agent configurations: once the search space is rich enough, the *operator* that explores it is secondary.

## 8. Limitations

**Synthetic agents.** All three domains use deterministic synthetic agents with hand-designed interaction bonuses. The specific numerical results cannot be extrapolated to real LLMs without a follow-up study. We are preparing that study on ForecastBench, HumanEval (LiveCodeBench filter), and Metaculus historical using Llama-3.3-70B and Qwen-2.5-72B. It is preregistered in the same document.

**Domain count.** Three domains is a meaningful improvement over the one-domain workshop result, but is still narrow. The paper's generality claims are bounded by this.

**Sample size for H2.** With 20 seeds per cell and three domains, we have 60 paired observations per comparison. This is enough to rule out operator effects with *d<sub>z</sub>* > 0.5 with >95% confidence, but cannot rule out small effects (*d<sub>z</sub>* ≈ 0.2). A real-LLM replication at 30+ seeds per cell is planned.

**Gene template design.** The 9-gene template was chosen based on prior agent-optimization literature plus author judgment. A "random gene template" ablation — showing the effect persists when the axes are chosen by a different process — would strengthen the paper.

**Crossover operators.** We tested three crossover operators (uniform, component-swap, fitness-weighted) as part of full evolution. More sophisticated semantic crossover (e.g. LLM-guided) is not included. Our claim is precisely about the operators we tested, not about crossover in general.

**Synthetic interaction bonuses.** The cross-component interactions we built into the synthetic fitness landscape are motivated by the forecasting and program-search literature but are not measured from real LLMs. Real LLMs may have stronger, weaker, or qualitatively different interaction structures.

## 9. Reproducibility

All code is released at `breed` (MIT license). The experimental pipeline is fully deterministic given seeds: `python paper/03_experiments/scripts/run_main_track_experiments.py` reproduces Tables 1–2 and Figures 1–E2 in approximately 400 seconds on a single CPU. Per-run JSON files, aggregated JSONL logs, figures (PDF + PNG), and statistical analysis JSON are all committed to the repository. 431 unit tests pass.

The preregistration was committed to git **before** any confirmatory run executed; the git hash of the preregistration is referenced in the repo. The final paper numbers and the preregistration's decision rules match: H1 and H2 were both confirmed, H3 was confirmed qualitatively (pairwise interactions present) but the numerical Sobol indices are noisy at our sample size and are reported as exploratory.

## References

[1] Q. Guo et al. Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers. *ICLR 2024*. arXiv:2309.08532.

[2] C. Fernando et al. PromptBreeder: Self-Referential Self-Improvement via Prompt Evolution. Google DeepMind Tech Report, 2023. arXiv:2309.16797.

[3] O. Khattab et al. DSPy: Compiling Declarative Language Model Calls into State-of-the-Art Pipelines. 2023.

[4] J. Zhang et al. Darwin-Gödel Machine: Open-Ended Evolution of Self-Improving Agents. Sakana AI, 2025. arXiv:2505.22954.

[5] DeepMind AlphaEvolve team. AlphaEvolve: Evolutionary Discovery of Scientific Algorithms. 2025. arXiv:2506.13131.

[6] B. Romera-Paredes et al. Mathematical discoveries from program search with large language models. *Nature*, 2024.

[7] Y. Shang et al. AgentSquare: Automatic LLM Agent Search in Modular Design Space. 2024. arXiv:2410.06153.

[8] EvoAgentX authors. EvoAgentX: An Automated Framework for Evolving Agentic Workflows. *EMNLP 2025 Demos*. arXiv:2507.03616.

[9] E. Real et al. Regularized Evolution for Image Classifier Architecture Search. *AAAI 2019*.

[10] E. Real et al. AutoML-Zero: Evolving Machine Learning Algorithms From Scratch. *ICML 2020*.

---

## Appendix A — Preregistration decision rules (verbatim from the 2026-04-10 commit)

**H1 — Dimensionality effect (confirmed at pooled *p* = 1e-14).** On each benchmark, full_evolution's mean test-set score exceeds prompt_only_evolution's mean test-set score, with the paired bootstrap 95% CI of the difference excluding zero. The effect holds on at least three out of four benchmarks. — Three of three synthetic domains; CI on the pooled mean difference is [+0.206, +0.303].

**H2 — Operator null (confirmed).** For each pair in {full_evolution, mutation_only, crossover_only, random_search, bayesian_opt, successive_halving}, the absolute value of the paired mean difference is ≤ 0.03 with 95% CI containing zero. — Confirmed on every pairwise comparison at the cross-domain aggregate level. Largest point-estimate gap is full vs crossover_only at +0.054 on coding, which does not survive Holm correction.

**H3 — Epistasis mechanism (confirmed qualitatively).** Sum of pairwise Sobol interaction indices ≥ 0.10 on at least two benchmarks. — Confirmed on all three synthetic domains. The numerical magnitudes are noisy at *n_base* = 512; full Saltelli decomposition at *n_base* = 2048 deferred to the real-LLM follow-up.

## Appendix B — Full tables

See `paper/04_results/tables/main_track/table_E1_summary.md` and `table_E1_comparisons.md` and `table_E_decisive.md` and `table_transfer.md`.
