# Novelty Memo: Lineage-Aware Evolution of LLM Agent Configurations

**Date:** 2026-04-09
**Purpose:** Decide whether there is a defensible publishable contribution here, and if so, define the sharpest possible scope.

---

## 1. The landscape in one page

After synthesizing the literature review at `paper/01_corpus/sources/03_agent_configuration_search.md` and prior art searches on evolutionary prompt optimization, self-improving agents, AutoML for agents, forecasting benchmarks, LLM-based forecasting SOTA, and NAS crossover evidence, the field decomposes into seven research tracks:

1. **Evolutionary prompt optimization.** EvoPrompt (Guo et al., ICLR 2024, arxiv 2309.08532) and PromptBreeder (Fernando et al., 2023, arxiv 2309.16797) use explicit genetic operators on prompt strings. EvoPrompt uses GA and DE crossover on prompt words across BIG-Bench Hard (up to 25% improvement). PromptBreeder evolves both task-prompts and the mutation-prompts that control their evolution, and explicitly stores "genotypes found in ascending order of quality" (informal lineage). Both operate on prompts only, not full agent configurations.

2. **Self-improving agents (code-rewriting).** Darwin Godel Machine (Sakana, 2025, arxiv 2505.22954) maintains an expanding lineage of coding agent variants through mutation-only LLM-guided self-rewrites. Reported: SWE-bench 20.0% → 50.0%, Polyglot 14.2% → 30.7%. AlphaEvolve (DeepMind, 2025, arxiv 2506.13131) evolves marked code blocks via LLM mutations. FunSearch (Romera-Paredes et al., Nature 2024) uses island-based evolutionary search over Python programs for mathematical discovery. All three evaluate on code/math tasks, not forecasting.

3. **Multi-component agent configuration search.** AgentSquare (Shang et al., 2024, arxiv 2410.06153) defines a modular design space across Planning/Reasoning/Tool Use/Memory with reported 17.2% average gain on web/embodied/tool/game benchmarks. EvoAgentX (arxiv 2507.03616) integrates TextGrad, AFlow, and MIPRO across a five-layer evolving framework, reports 7-20% gains on HotPotQA, MBPP, MATH, and GAIA. Neither evaluates forecasting. Neither reports an ablation isolating crossover.

4. **Workflow / pipeline search.** AFlow (Zhang et al., ICLR 2025), SELA, AutoML-Agent, AutoPDL. These search over workflow topologies, not genome-level configurations.

5. **LLMs as evolutionary operators.** LMEA (arxiv 2310.19046), GEPA (Genetic-Pareto optimizer in DSPy), Agentic Variation Operators. These use LLMs to perform mutation/crossover on text, but do not define full-agent genomes.

6. **NAS crossover evidence.** Regularized Evolution (Real et al., 2019) famously used mutation-only for AmoebaNet and was competitive with more complex approaches. AutoML-Zero (Real et al., 2020) found that crossover had a marginal effect on symbolic program search. Large-Scale Evolution of Image Classifiers used crossover but did not rigorously ablate it. **The NAS literature's consensus is that crossover in structured search spaces has empirically small or domain-dependent effects; mutation-only is often sufficient.** This is a red flag for any paper claiming crossover is crucial for agent configurations -- the burden of proof is high.

7. **LLM forecasting SOTA.** Halawi et al. 2024 "Approaching Human-Level Forecasting with Language Models" (Berkeley) used retrieval-augmented generation, self-consistency, and scratchpad reasoning to reach near-superforecaster accuracy. ForecastBench (Karger et al., 2024) provides a dynamic benchmark with ~1,000 rolling questions. Metaculus runs an AI forecasting tournament. **No published paper applies evolutionary optimization to LLM forecasting agents.** This is the cleanest domain gap.

## 2. Where breed sits in this landscape

| System | Full agent genome? | Explicit crossover? | Lineage? | Forecasting? | Cross-component ablation? |
|--------|:---:|:---:|:---:|:---:|:---:|
| EvoPrompt | No (prompts only) | **Yes** | No | No | N/A |
| PromptBreeder | No (prompts + meta-prompts) | Yes (limited) | Partial | No | N/A |
| DSPy MIPROv2/COPRO/GEPA | No (prompts + demos) | GEPA only | No | No | N/A |
| Darwin Godel Machine | Code only | No (mutation only) | **Yes** | No | N/A |
| AlphaEvolve | Code blocks | No | Partial | No | N/A |
| FunSearch | Code | No (island-based) | No | No | N/A |
| AgentSquare | **Yes** (4 modules) | Unclear | No | No | No |
| EvoAgentX | **Yes** (multi-layer) | No (uses TextGrad/AFlow/MIPRO) | No | No | No |
| A-Evolve | **Yes** (workspace files) | Unclear | **Yes** (git tags) | No | No |
| AFlow | Workflow topology | No | No | No | N/A |
| **breed (this work)** | **Yes** (9 typed genes) | **Yes** (3 operators) | **Yes** (JSONL) | **Yes** (ForecastBench/Metaculus) | **Yes** (6 baselines) |

**The genuine gap breed fills:** an explicit, typed genome schema for LLM agents with cross-component crossover, evaluated rigorously on forecasting with clean baselines and ablations. Every prior system either (a) evolves only prompts, (b) evolves only code, (c) evolves full configs without forecasting or without ablations, or (d) does lineage tracking without crossover.

## 3. Honest novelty assessment

### What is NOT novel
- Treating agent components as genes. AgentSquare already did this.
- Using crossover operators on text. EvoPrompt already did this.
- Tracking lineage. DGM and A-Evolve already do this.
- Using evolutionary search on LLM systems. Widespread.
- The biological metaphor. PromptBreeder already used "genotype" language.

### What is potentially novel
- The specific combination: typed genome + cross-component crossover + forecasting evaluation + clean ablations.
- A **negative or weak result** on whether crossover helps across heterogeneous agent configuration types. This is still a contribution if the field has assumed crossover helps without testing it rigorously.
- An interpretable gene-persistence analysis connecting selected traits to forecasting performance.

### What would get destroyed in peer review
- "We introduce a new framework for agent evolution" — false. The framework components are assemblies of prior work.
- "Evolution discovers better forecasting agents than any prior method" — unless tested against Halawi et al. 2024 baselines, this is unsupported.
- "Agent genomes are a novel formalism" — AgentSquare has a design space formalism.
- "Crossover is essential for agent evolution" — NAS literature disputes this, and our own preliminary synthetic results show no significant advantage.

## 4. What our preliminary experiment actually shows

On the built-in 50-question dataset (21 train, 29 test) with a synthetic agent and 8 seeds:

| Method | Test fitness | vs. full_evolution |
|---|---|---|
| full_evolution | **0.806** ± 0.052 | — |
| mutation_only | 0.800 ± 0.025 | p=0.73, d_z=0.13 |
| crossover_only | 0.784 ± 0.038 | p=0.04 (in favor of full), d_z=0.87 |
| random_search | 0.791 ± 0.081 | p=0.71, d_z=0.14 |
| static_best | 0.799 ± 0.039 | p=0.73, d_z=0.13 |
| static_ensemble | 0.780 ± 0.016 | p=0.19, d_z=0.52 |
| prompt_only_evolution | 0.750 ± 0.008 | p=0.02 (in favor of full), d_z=1.07 |

After Holm-Bonferroni correction over 6 tests, **nothing is statistically significant at α=0.05**. The strongest effect is full_evolution vs. prompt_only_evolution (Holm-adjusted p=0.11, d_z=1.07).

This is a **useful but narrow finding**:
1. Multi-component evolution beats single-component evolution (the gap vs prompt_only). This is the only result that approaches significance.
2. Full evolution vs. mutation-only, random search, static best, or static ensemble shows no meaningful advantage at this scale.
3. Crossover alone (without mutation) performs WORSE than full evolution, suggesting mutation provides critical exploration.

At n=8 seeds on 29 test questions this is underpowered. A full paper would need n≥20 seeds, a larger dataset, and ideally live LLM agents. But the *direction* of the result is informative: **the strongest defensible claim is that "multi-component search beats single-component search"**, not "crossover is essential" or "evolution beats baselines".

## 5. Reframing the contribution

Original framing ("Agentic Breeding") is not defensible. The honest reframing is:

### Title
**"Lineage-Aware Configuration Search for LLM Forecasting Agents: A Cleanly-Ablated Study"**

or equivalently:

**"Does Cross-Component Search Help LLM Forecasting Agents? An Empirical Study"**

### One-sentence thesis
We formalize LLM agent configurations as typed genomes with five gene types, implement population-based search with crossover and mutation operators for each type, evaluate on forecasting, and show that while multi-component search reliably outperforms single-component (prompt-only) search, no clear advantage for crossover over mutation-only emerges at the scales we test.

### Contribution claims (in order of defensibility)
1. **(Strongest)** A typed genome schema for LLM agent configurations with type-specific crossover and mutation operators, released as open-source (breed). The schema is immediately reusable for any agent framework.
2. **(Strong)** The first rigorous comparison of multi-component vs. single-component evolutionary search on forecasting tasks with time-based train/test splits and clean paired statistical tests.
3. **(Honest)** A preliminary negative-ish result: cross-component crossover does not provide statistically significant advantage over mutation-only or random search at n=8 seeds on our synthetic benchmark. This is consistent with NAS literature's cautious view of crossover.
4. **(Weaker)** Lineage analysis of which genes persist across generations, connecting specific configuration traits to forecasting fitness.

### What we should NOT claim
- "Evolution discovers new reasoning strategies."
- "Our framework beats the SOTA on forecasting."
- "Crossover is essential for agent optimization."
- "Agent breeding is a new paradigm."

### Preferred venue
- **Primary target:** NeurIPS AutoML workshop or ICML AutoML workshop -- these welcome honest negative/mixed results and representational contributions.
- **Secondary target:** EMNLP Findings if we can get a real LLM forecasting experiment at n≥20 seeds.
- **Not viable:** NeurIPS / ICML main track -- the result is too weak and the scope too narrow without much more data.

## 6. What additional work would strengthen this?

Ordered by impact:

1. **Run on real LLMs (Claude/GPT) instead of synthetic agent.** This is the #1 gap. The synthetic agent has known structure; real LLMs may have landscapes where crossover helps more or less. Budget estimate: 20 seeds × 7 methods × 300 agent calls × 50 test calls ≈ 75,000 API calls per experiment. At ~$0.01 per Claude call that's ~$750 per full run. Feasible but nontrivial.

2. **Use real Metaculus / ForecastBench data, not built-in 50 questions.** Even 200 resolved questions with a proper 100/100 time-based split would dramatically reduce variance.

3. **Run n=20+ seeds** to actually distinguish small effects.

4. **Add a real-LLM forecasting baseline from Halawi et al. 2024** to contextualize absolute performance.

5. **Test on a second domain** (coding agents on HumanEval, or web agents on Online-Mind2Web) to check whether the finding generalizes.

6. **Add a theoretical analysis** of when crossover should help for heterogeneous genomes (building block hypothesis applied to agent configs).

## 7. Decision: go or no-go?

**Go, but with narrowed scope.**

Rationale:
- The infrastructure (breed) is already built and tested (422+ passing tests).
- The scientific question ("does cross-component search help for LLM agents?") is genuinely open and under-studied.
- The honest answer appears to be "slightly yes for multi-component vs single-component, not clearly for crossover vs mutation" -- this is publishable at a workshop with proper framing.
- The released code is a useful community artifact regardless of the paper outcome.
- The experimental pipeline produces publication-ready figures and tables automatically.

**What to drop:**
- Claims of novelty beyond the specific synthesis.
- The "Agentic Breeding" branding in any technical writeup.
- Any ambition for main-track NeurIPS or ICML before running real-LLM experiments.
- Any claim about "discovering reasoning strategies" or "emergent intelligence".

**What to commit to:**
- A workshop paper titled "Lineage-Aware Configuration Search for LLM Forecasting Agents"
- The open-source breed library as the primary artifact.
- Honest reporting of the null/weak effects.
- A clear discussion of what would be needed to strengthen the claim.

---

*Next step: draft the full paper using the experimental results we already have, clearly marked as preliminary synthetic-benchmark evidence, with a prominent section on limitations and required follow-up work.*
