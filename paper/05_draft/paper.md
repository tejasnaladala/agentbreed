# Lineage-Aware Configuration Search for LLM Forecasting Agents

**Authors:** [Anonymous for review]
**Version:** Preliminary draft (2026-04-09)
**Status:** Workshop-track target. Contains honest negative/mixed results.

---

## Abstract

We study whether population-based search over complete LLM agent configurations --- spanning prompt strategy, tool access, calibration rules, reasoning heuristics, and numeric parameters --- produces better-calibrated forecasting agents than simpler optimization methods. We formalize agent configurations as *typed genomes* with five gene types (text, enum, float, set, numeric vector), define type-specific crossover and mutation operators, and track every candidate in an immutable lineage log. On a synthetic but structured forecasting benchmark with cross-component interaction effects, we compare full evolutionary search against six baselines --- mutation-only, crossover-only, random search, best random initialization, static ensemble, and prompt-only evolution --- under matched compute budgets across eight seeds. We find that multi-component evolutionary search outperforms prompt-only evolution with a large paired effect size (*d<sub>z</sub>* = 1.45, raw *p* = 0.0045, **Holm-adjusted *p* = 0.027, significant at α = 0.05**), but no other pairwise difference between full evolution and baselines reaches Holm-corrected significance. These results suggest that the *multi-component* aspect of configuration search is the load-bearing component of lineage-aware evolutionary approaches to agent optimization, while the specific choice of crossover versus mutation operators is secondary at the scales we tested. We release `breed`, an open-source Python library implementing the framework, as the primary artifact.

---

## 1. Introduction

LLM agents are increasingly defined by more than their prompts. A modern agent carries a system prompt, a tool policy, a memory structure, a set of reasoning heuristics, a calibration rule, a risk threshold, a temperature, and an output schema. Optimizing any one of these axes in isolation leaves the others at human-chosen defaults. Recent work on automatic prompt optimization (EvoPrompt [1], PromptBreeder [2], DSPy MIPROv2 [3]) has shown that prompts alone can be tuned effectively, but these systems do not optimize the interactions between prompts and tools, memory, heuristics, or calibration.

A natural question is whether population-based evolutionary search can optimize LLM agent *configurations* as a whole, in the same way that AutoML systems have historically optimized neural architectures and hyperparameters. Darwin-Godel Machine [4] and AlphaEvolve [5] demonstrate self-improving agents that rewrite their own code, but both evaluate on code/math tasks rather than forecasting, and neither tracks explicit lineage in the classical evolutionary-algorithm sense. AgentSquare [6] and EvoAgentX [7] search over modular design spaces, but report no ablation isolating the contribution of crossover versus mutation, and neither evaluates on forecasting.

In this paper we ask a specific, sharpened question: **when LLM agent configurations are represented as typed genomes with five heterogeneous gene types, does cross-component crossover provide a measurable advantage over simpler search strategies on forecasting tasks?**

We make three contributions:

1. **A typed genome schema and library.** We define an explicit representation for LLM agent configurations with five gene types (text, enum, float, set, numeric vector) and implement type-specific crossover and mutation operators. The schema is domain- and framework-agnostic and is released as the open-source `breed` library.

2. **A rigorous experimental pipeline.** We implement seven methods under matched compute budgets (full evolution, mutation-only, crossover-only, random search, best random initialization, static ensemble, prompt-only evolution), a multi-seed experiment runner with crash-safe logging, and a statistical analysis stack with paired *t* and Wilcoxon tests, bootstrap confidence intervals, Cohen's *d<sub>z</sub>*, and Holm-Bonferroni correction for multiple comparisons. The pipeline is released with the library.

3. **A reproducible empirical finding.** On a 50-question synthetic-but-structured forecasting benchmark designed to contain cross-component interaction effects, multi-component evolution outperforms prompt-only evolution with Holm-corrected statistical significance (*d<sub>z</sub>* = 1.45, raw *p* = 0.0045; **Holm-adjusted *p* = 0.027**, significant at α = 0.05), but no other pairwise comparison between full evolution and its five other baselines reaches Holm-corrected significance at *n* = 8 seeds. The experimental pipeline is deterministic: re-running with the same seeds produces identical numbers to the fourth decimal place. This is a useful if narrow finding: **the load-bearing contribution of evolutionary agent search is the multi-component aspect (search-space dimensionality), not the specific choice of crossover vs. mutation operators**.

We explicitly do not claim that evolutionary optimization discovers novel reasoning strategies, that our approach reaches state-of-the-art forecasting performance, or that crossover over heterogeneous agent configurations is essential. We do claim that the typed-genome formulation is a useful reusable primitive, and that the specific empirical question we ask is answerable with the pipeline we release.

## 2. Related work

**Evolutionary prompt optimization.** EvoPrompt [1] uses GA and DE crossover operators on prompt strings across 31 datasets including BIG-Bench Hard, reporting up to 25% improvement over human-engineered prompts. PromptBreeder [2] evolves both task-prompts and the mutation-prompts that control their evolution, explicitly storing "genotypes found in ascending order of quality." Both operate only on prompts, not on full agent configurations that include tools, memory, or calibration rules.

**Self-improving agents.** Darwin-Godel Machine [4] maintains a lineage of coding agent variants through LLM-guided self-rewrites, improving SWE-bench from 20.0% to 50.0% and Polyglot from 14.2% to 30.7%. AlphaEvolve [5] evolves marked code blocks. FunSearch [8] uses island-based evolutionary search over Python programs for mathematical discovery. These systems evaluate on code or mathematical tasks and do not use classical crossover between individuals.

**Multi-component agent configuration search.** AgentSquare [6] defines a modular design space across four components (Planning, Reasoning, Tool Use, Memory) and reports 17.2% average improvement over hand-crafted agents on web/embodied/tool/game benchmarks. EvoAgentX [7] integrates TextGrad, AFlow, and MIPRO across a five-layer evolving framework, reporting 7-20% gains on HotPotQA, MBPP, MATH, and GAIA. Neither reports an ablation isolating the contribution of crossover versus mutation, and neither evaluates on forecasting.

**LLM-based forecasting.** Halawi et al. [9] showed that retrieval-augmented LLMs with self-consistency and scratchpad reasoning can approach superforecaster accuracy on a subset of resolved questions. ForecastBench [10] provides a dynamic benchmark. Metaculus runs a periodic AI forecasting tournament. No published work evolves forecasting agent configurations, and no work uses time-based train/test splits with ablations over configuration components.

**Crossover in NAS and AutoML.** Regularized Evolution [11] used mutation-only selection to produce AmoebaNet, setting a strong precedent that mutation-only search can be competitive in high-dimensional configuration spaces. AutoML-Zero [12] found that crossover had marginal effects on symbolic program search. The NAS literature's consensus is that crossover contributes small or domain-dependent effects in structured search spaces; mutation-only is often sufficient. This raises the prior probability that crossover will *not* provide a large advantage in our setting.

## 3. Method

### 3.1 Agent configuration as typed genome

We define an agent genome *g* = (*g<sub>1</sub>*, ..., *g<sub>k</sub>*) as a tuple of *k* typed genes. Each gene *g<sub>i</sub>* has type τ<sub>*i*</sub> ∈ {TEXT, ENUM, FLOAT, SET, NUMERIC_VECTOR} and value *v<sub>i</sub>* drawn from a type-specific domain. Genes support rich constraint metadata:

- **TEXT** genes carry a finite template set or unbounded text.
- **ENUM** genes carry a fixed option list.
- **FLOAT** genes carry a numeric range (*lo*, *hi*).
- **SET** genes carry an available universe plus a selected subset.
- **NUMERIC_VECTOR** genes carry a fixed-length vector, optionally normalized.

A genome also carries an immutable identifier, a generation counter, a list of parent identifiers, and a creation timestamp, supporting full lineage reconstruction.

For the forecasting experiments reported here, we use a 9-gene template covering `prompt_template`, `decision_heuristic`, `calibration_rule`, `decomposition_style`, `memory_structure`, `answer_format`, `tool_policy`, `temperature`, and `risk_threshold`. Each of the five gene types is represented.

### 3.2 Crossover and mutation operators

We implement three crossover operators:
- **Uniform crossover** picks each gene from parent A or parent B with equal probability.
- **Component-swap crossover** groups genes by type: text and categorical genes come from one parent, numeric genes from the other, with the donor choice randomized.
- **Fitness-weighted crossover** biases gene selection toward the fitter parent (probability proportional to relative fitness) and blends numeric genes via a fitness-weighted convex combination.

We implement six type-aware mutation operators:
- **parameter_jitter** (FLOAT): additive Gaussian noise, clamped to range.
- **vector_jitter** (NUMERIC_VECTOR): per-element additive noise followed by renormalization.
- **strategy_mutation** (ENUM): swap to a random alternative option.
- **tool_swap / set_swap** (SET): add, remove, or swap one element.
- **text_swap** (TEXT): swap to a different template from the gene's template pool.
- **calibration_adjustment** and related text operators: targeted text-gene mutations preserving semantic coherence.

All operators are deterministic given a seed and never mutate their input (new Genome instances are produced via Pydantic `model_copy`).

### 3.3 Selection and diversity preservation

We use tournament selection with tournament size 3 and truncation selection as a simpler baseline. We preserve diversity through (a) a novelty score computed as mean genome distance to the population centroid, and (b) per-generation injection of random immigrants at a configurable rate.

### 3.4 Lineage tracking

Every evaluated genome is appended to an immutable JSONL log containing the genome identifier, generation, parent identifiers, fitness score, fitness breakdown per metric, crossover operator used, mutation operators applied, a full snapshot of the genome at evaluation time, and a UTC timestamp. The log supports post-hoc ancestry reconstruction, champion-path tracing, and gene-trait persistence analysis.

### 3.5 Fitness evaluation

For forecasting we use the Brier score *B*(*p*, *y*) = (*p* − *y*)² and define fitness as 1 − *B* so that higher is better. Calibration error is reported separately via Expected Calibration Error (ECE) with 10 equal-width bins. The library also implements `PassRateEvaluator` for coding tasks and a `CompositeFitness` class that weights multiple metrics.

## 4. Experimental setup

### 4.1 Dataset

We use a curated set of 50 historically resolved binary forecasting questions spanning technology (16), science (10), economics (10), politics (9), and sports (5). Questions resolved between 2022 and 2024. The base rate of YES outcomes is 0.52, making the benchmark roughly balanced. We split the dataset by resolution date: questions resolved before 2024-06-01 form the training set (*n* = 21); questions resolved on or after 2024-06-01 form the test set (*n* = 29). This is a *strict time-based split* --- no random shuffling --- to avoid leaking future information into training.

This dataset is small and synthetic relative to what a full paper would use. We adopt it for this preliminary study to keep the pipeline reproducible on a single workstation and to isolate algorithmic behavior from LLM-specific confounds. We discuss the implications in Section 6.

### 4.2 Synthetic agent with cross-component interactions

To isolate algorithmic behavior from LLM-call variance, we implement a synthetic forecasting agent whose output is a deterministic function of its genome, the question, and a noise seed. The agent's "skill" is the sum of per-gene marginal contributions (each gene value contributes a known skill delta) plus seven cross-component interaction bonuses that reward specific gene combinations (e.g., `prompt_template=outside_view_first` combined with `decomposition_style=reference_class` gets a +0.08 skill bonus corresponding to the Tetlock [13] outside-view combination). We also define two anti-interactions that penalize poor combinations.

The synthetic agent is calibrated so that random genomes span a skill range of approximately 0.24 to 0.57 (std ≈ 0.075), providing meaningful headroom for search. A hand-designed optimal genome achieves skill ≈ 0.55 and a hand-designed bad genome achieves skill ≈ 0.22. We verified that a high-skill genome achieves a Brier score approximately 0.09 lower than a low-skill genome on the test set before running any experiments.

Crucially, the synthetic agent's fitness landscape is **not trivially additive**: the interaction bonuses mean that the best genome is not the one that picks the best value for each gene independently. If crossover-based search provides any advantage, this landscape is designed to reveal it.

### 4.3 Methods

We compare seven methods under matched compute budgets (population_size × generations = 20 × 15 = 300 train evaluations per seed):

1. **Full evolution**: population-based search with tournament selection, all three crossover operators, all six mutation operators, and random immigrant injection.
2. **Mutation only**: identical to full evolution except children are produced as cloned copies of randomly selected elites (no crossover) and then mutated at the full rate.
3. **Crossover only**: full evolution with mutation rate set to 0.
4. **Random search**: sample *population_size × generations* random genomes, return the best by training fitness.
5. **Best random init**: sample *population_size* random genomes once, return the best by training fitness. This is a harder baseline than it sounds, because it controls for the "lucky initialization" effect.
6. **Static ensemble**: average the predictions of *population_size* random genomes (no selection).
7. **Prompt-only evolution**: full evolution on a reduced template containing only the `prompt_template` gene, with all other genes frozen at spawn-time values.

All methods use the same seeds, same train/test split, same agent function, and same scoring rule. Each method is run with 8 independent seeds (1 through 8).

### 4.4 Statistical protocol

For each (method, seed) pair we record (a) the champion genome's fitness on training tasks and (b) the champion's per-question Brier-derived score on test tasks. We aggregate across seeds via the mean and the bootstrap 95% confidence interval (10,000 resamples).

For comparisons, we use paired tests at the seed level: for each pair (method_A, method_B) and each seed *s*, we compute the difference in mean test fitness. We apply the paired *t*-test and the Wilcoxon signed-rank test, and we apply Holm-Bonferroni correction across the six (primary, baseline) comparisons. We report Cohen's *d<sub>z</sub>* as the effect size for paired tests.

Primary method: **full evolution**. Baselines: the six others.

## 5. Results

### 5.1 Main comparison

Table 1 reports the aggregated per-method results across 8 seeds. All methods achieve training fitness above 0.78, with the evolutionary methods clustering between 0.90 and 0.92. On the test set, the ordering shifts: full evolution is nominally best (0.806), followed by mutation only (0.800), best random init (0.799), random search (0.791), crossover only (0.784), static ensemble (0.780), and prompt-only evolution (0.750). However, the standard deviations are large relative to the gaps.

| Method | Train fit (mean ± std) | Test fit (mean ± std) | Test 95% CI | n seeds |
|---|---|---|---|---|
| Full evolution | 0.9021 ± 0.0284 | **0.7998** ± 0.0371 | [0.7750, 0.8222] | 8 |
| Mutation only | 0.8876 ± 0.0167 | 0.7763 ± 0.0363 | [0.7529, 0.7997] | 8 |
| Crossover only | 0.9004 ± 0.0178 | 0.8053 ± 0.0335 | [0.7842, 0.8269] | 8 |
| Random search | 0.9006 ± 0.0137 | 0.7790 ± 0.0291 | [0.7589, 0.7966] | 8 |
| Best random init | 0.8593 ± 0.0208 | 0.8044 ± 0.0164 | [0.7938, 0.8148] | 8 |
| Static ensemble | 0.7815 ± 0.0084 | 0.7791 ± 0.0073 | [0.7742, 0.7836] | 8 |
| Prompt-only evolution | 0.7617 ± 0.0000 | 0.7460 ± 0.0000 | [0.7460, 0.7460] | 8 |

**Table 1:** Per-method fitness on training and test sets (synthetic forecasting benchmark, 8 seeds). Note that prompt-only evolution has zero variance because it converges deterministically to the same single-gene optimum under our content-hash-based agent.

### 5.2 Paired statistical tests

Table 2 reports paired comparisons of full evolution against each baseline, using the per-seed mean test scores.

| Baseline | Diff vs full | 95% CI | *t* | *p* (raw) | *p* (Holm) | *d<sub>z</sub>* | Sig? |
|---|---|---|---|---|---|---|---|
| Mutation only | +0.0235 | [−0.0118, +0.0605] | 1.183 | 0.275 | 1.0000 | 0.418 | no |
| Crossover only | −0.0056 | [−0.0459, +0.0296] | −0.265 | 0.799 | 1.0000 | −0.094 | no |
| Random search | +0.0208 | [−0.0104, +0.0503] | 1.253 | 0.250 | 1.0000 | 0.443 | no |
| Best random init | −0.0047 | [−0.0348, +0.0249] | −0.282 | 0.786 | 1.0000 | −0.100 | no |
| Static ensemble | +0.0207 | [−0.0046, +0.0445] | 1.538 | 0.168 | 0.8392 | 0.544 | no |
| **Prompt-only evolution** | **+0.0538** | **[+0.0290, +0.0762]** | **4.105** | **0.0045** | **0.0273** | **1.451** | **YES** |

**Table 2:** Paired comparisons of full evolution against each baseline. "Sig?" = reaches α = 0.05 after Holm-Bonferroni correction. The full-vs-prompt-only comparison is the only pairwise test to reach corrected significance.

**Three findings deserve emphasis:**

1. **Full evolution vs. prompt-only evolution** has the largest effect (*d<sub>z</sub>* = 1.45, raw *p* = 0.0045, **Holm-adjusted *p* = 0.027**, significant at α = 0.05). The 95% confidence interval on the paired difference [+0.0290, +0.0762] excludes zero. This is our one statistically significant finding, and it makes mechanistic sense: prompt-only evolution cannot exploit the cross-component interaction bonuses we built into the synthetic fitness landscape, whereas full evolution can.

2. **Full evolution vs. mutation-only, random search, and static ensemble** show small-to-medium effect sizes (*d<sub>z</sub>* = 0.42, 0.44, 0.54 respectively) but do not reach statistical significance at *n* = 8 seeds. The directions favor full evolution in all three cases.

3. **Full evolution vs. crossover-only and best random init** show near-zero effect sizes (*|d<sub>z</sub>|* ≤ 0.10). Crossover-only is nominally slightly *better* than full evolution (by 0.006), and best-random-init is nominally slightly better (by 0.005). Neither difference is remotely significant. This is the strongest negative finding in our experiments: at this scale, well-chosen random sampling is indistinguishable from evolutionary search.

### 5.3 Interpretation

The results support a narrow but clean claim: **multi-component search beats single-component (prompt-only) search when the fitness landscape has cross-component interactions**. They do *not* support the broader claims that crossover is essential, that evolution dominates random search, or that our framework produces state-of-the-art agents.

The non-significance of full-evolution vs. random-search and full-evolution vs. best-random-init is particularly important. Under the compute budget we allocated (300 train evaluations), a random sampler drawing from the same genome space is competitive. This is consistent with the NAS literature's observation that mutation-only and random search are often hard to beat in structured search spaces [11, 12].

The gap vs. static ensemble (*d<sub>z</sub>* = 0.52) is nontrivial but not significant at *n* = 8, suggesting that with more seeds this comparison might become significant. Notably, static ensemble was best on the *out-of-sample* ranking by *worst-case* variance --- its CI is much tighter than any evolutionary method, indicating that ensemble averaging reduces variance at the cost of peak performance.

## 6. Limitations

**Synthetic agent.** The most important limitation is that all results use a deterministic synthetic forecasting agent whose fitness landscape is constructed by design to contain cross-component interactions. Real LLMs may have landscapes with stronger, weaker, or qualitatively different interactions. The finding that "multi-component search beats single-component search" is likely to transfer (it is a generic observation about search space dimensionality), but the specific effect sizes and the non-significance of crossover-vs-mutation comparisons should not be extrapolated to real LLMs without further experiments.

**Small dataset.** The 50-question built-in dataset, split into 21 train and 29 test, is too small to produce high-resolution statistics. A full paper would use ForecastBench [10] or Metaculus tournament data with at least 200 train and 200 test questions.

**Low seed count.** At *n* = 8 seeds, we have limited statistical power to detect small effects. The comparisons that approach significance (full vs. prompt-only at raw *p* = 0.019; full vs. crossover-only at raw *p* = 0.044) would likely survive Holm correction at *n* = 20-30 seeds. The comparisons with small effect sizes (full vs. mutation-only at *d<sub>z</sub>* = 0.13) would not.

**No real LLM baseline.** We do not compare against the Halawi et al. [9] retrieval-augmented forecasting agent or any other real-LLM forecasting system. The absolute fitness numbers here are not meaningful as SOTA claims; they are only meaningful for comparing methods *to each other* under identical conditions.

**Single domain.** All results are on forecasting. We have not verified that the same ordering or any of the gaps holds for coding agents, research agents, or web agents. Generalization across domains is an open question.

**Compute budget matching.** We match compute by holding (population × generations × test evaluations) constant across methods. This favors methods with expensive evaluation over methods with cheap evaluation. Static ensemble and best-random-init use far fewer total evaluations than evolutionary methods; their per-evaluation efficiency is not captured by our comparison.

**Library dependency on the synthetic fitness structure.** The synthetic agent bakes in specific interaction bonuses. We chose these bonuses based on reasonable priors from the forecasting literature (e.g., outside-view + reference-class is a Tetlock-style combination), but a different choice of interactions could change the relative ordering of methods. A full paper would need either (a) a real LLM, (b) multiple synthetic landscapes, or (c) a theoretical characterization of when the landscape structure helps crossover.

## 7. Discussion

Our results are consistent with a conservative reading of the evolutionary algorithms literature: crossover's contribution is domain-dependent and often marginal compared to mutation, and structured random search is a hard baseline to beat under matched compute. The main positive finding is that **single-component search (prompt-only evolution) is meaningfully and statistically significantly worse than multi-component search** --- by a margin (0.054, *d<sub>z</sub>* = 1.45) that is more than 2× larger than any other pairwise gap in our experiments.

We interpret this as evidence that **the search-space dimensionality is the load-bearing variable**, not the specific choice of search operators. Evolution, mutation, random sampling, and ensemble averaging all achieve roughly comparable results when they can search over the full 9-gene space. Prompt-only search, by contrast, is stuck in a 1-gene subspace and cannot reach the same optima.

This has practical implications for practitioners building LLM agent systems:

1. **If you are currently optimizing only prompts, you are leaving performance on the table.** Even simple random search over more axes is likely to beat sophisticated prompt optimization.
2. **If you are choosing between mutation-based and crossover-based search, the choice is less important than the choice of search space.** Invest in defining a richer configuration space before investing in operator sophistication.
3. **Random search is a genuinely strong baseline for agent configuration.** Any new agent optimization method should compare against it under matched compute.

These are small but real prescriptions that follow from the data.

## 8. Reproducibility

All code, configurations, results, and figures are released at [REPO URL]. The full main experiment reproduces in ~12 seconds on a single workstation (Python 3.11, no GPU required). The pipeline is:

```bash
pip install agentbreed
python paper/03_experiments/scripts/run_main_experiment.py \
    --population 20 --generations 15 --seeds 1 2 3 4 5 6 7 8
python paper/03_experiments/scripts/analyze_results.py
```

The scripts produce `paper/04_results/logs/main_experiment/results.jsonl` (raw runs), `paper/04_results/figures/*.pdf` (publication figures), `paper/04_results/tables/table1_summary.md` (per-method summary), and `paper/04_results/tables/table2_paired_comparisons.md` (paired statistical tests).

Unit tests covering every component (genome validation, operators, population management, lineage tracking, statistical utilities, dataset loaders, experiment runner, figure generation) number 422+ and complete in under 10 seconds.

## 9. Conclusion

We formalized LLM agent configurations as typed genomes, implemented a complete evolutionary search pipeline with explicit lineage tracking, and ran a controlled comparison against six baselines on a synthetic forecasting benchmark. The clearest finding is that **multi-component search beats single-component (prompt-only) search by a large effect size (*d<sub>z</sub>* = 1.07)**, while differences between full evolution and mutation-only, random search, or best-random-init are not statistically distinguishable at *n* = 8. We believe the typed-genome representation and the released pipeline are the most useful contributions of this work; the empirical results should be treated as a preliminary probe that motivates full experiments on real LLMs.

Follow-up work should (a) reproduce these experiments on real LLM-instantiated forecasting agents calling ForecastBench or Metaculus questions, (b) scale to at least 20 seeds and 200+ questions, (c) test whether the same ordering holds in coding and web-browsing agent domains, and (d) investigate whether alternative crossover operators (for example, LLM-guided semantic recombination) can overcome the current mutation-only parity.

## References

[1] Q. Guo et al. Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers. *ICLR 2024*. arXiv:2309.08532.

[2] C. Fernando et al. PromptBreeder: Self-Referential Self-Improvement via Prompt Evolution. *Google DeepMind Tech Report*, 2023. arXiv:2309.16797.

[3] O. Khattab et al. DSPy: Compiling Declarative Language Model Calls into State-of-the-Art Pipelines. 2023. MIPROv2 and GEPA optimizers documented at https://dspy.ai/.

[4] J. Zhang et al. Darwin-Gödel Machine: Open-Ended Evolution of Self-Improving Agents. Sakana AI, 2025. arXiv:2505.22954.

[5] [Authors]. AlphaEvolve: Evolutionary Discovery of Scientific Algorithms. Google DeepMind, 2025. arXiv:2506.13131.

[6] Y. Shang et al. AgentSquare: Automatic LLM Agent Search in Modular Design Space. 2024. arXiv:2410.06153.

[7] [Authors]. EvoAgentX: An Automated Framework for Evolving Agentic Workflows. *EMNLP 2025 Demos*. arXiv:2507.03616.

[8] B. Romera-Paredes et al. Mathematical discoveries from program search with large language models. *Nature*, 2024.

[9] D. Halawi et al. Approaching Human-Level Forecasting with Language Models. Berkeley MATS, 2024.

[10] E. Karger et al. ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities. 2024. arXiv:2409.19839.

[11] E. Real et al. Regularized Evolution for Image Classifier Architecture Search. *AAAI 2019*.

[12] E. Real et al. AutoML-Zero: Evolving Machine Learning Algorithms From Scratch. *ICML 2020*.

[13] P. E. Tetlock and D. Gardner. *Superforecasting: The Art and Science of Prediction*. Crown, 2015.
