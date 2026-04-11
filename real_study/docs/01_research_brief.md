# Phase 2 — Research Brief

## The root question

When practitioners optimize an LLM agent for a new task, they have many knobs: prompts, reasoning scaffolds, memory structures, tool policies, decoding hyperparameters. Published agent-optimization systems (AgentSquare, EvoAgentX, PromptBreeder, DSPy, GEPA) report gains from jointly searching several of these, but none rigorously isolates what the gain actually comes from. Is it the search operator (crossover, mutation, TPE, hyperband)? Is it the search space (which genes are searchable)? Is it the budget?

This paper tries to answer that.

## Three candidate framings

Below, each framing is laid out with its strongest form, its required experiments, its risks, and its fit to the two target venues (NeurIPS 2026 E&D and NeurIPS 2026 main track).

### Option 1 — The evaluation paper

**Strongest form.** "We build a rigorous, contamination-controlled evaluation harness for agent configuration search on real benchmarks, and use it to measure which parts of the search matter most. We find that [specific finding] dominates, while [specific finding] is unexpectedly weak." The novelty is in the careful measurement, not in any new algorithm.

**Required experiments:**
- Full 9-method × 3-benchmark × 2-model matrix at 15+ seeds each, budget-matched.
- Dimensionality sweep K ∈ {1, 2, 3, 5, 7, 9} on 2 benchmarks × 2 models.
- Transfer matrix (cross-model + cross-benchmark) at reduced seed count.
- Budget-scaling curves at `B ∈ {100, 300, 1000}` LLM calls.
- Honest limitations: decoding stochasticity, benchmark contamination, model-family diversity.

**Risks:**
- A reviewer says "this is just good benchmarking, not research." Mitigation: the equivalence-testing framework for H2 is novel in the agent-search literature, and the mixed-effects aggregation across benchmarks is rare.
- If the finding is "all methods tie," reviewers may call the paper boring. Mitigation: lean into the negative result — it is actionable for practitioners. Reference Regularized Evolution (Real et al. 2019) which was a famous negative-result paper.
- Fit-to-E&D is perfect; fit-to-main-track requires that the finding be surprising and actionable, not a routine benchmarking exercise.

**Compute:** ~25–30 GPU-days on H100 (main matrix + sweep + transfer). Within budget.

**Training / fine-tuning needed:** No.

**NeurIPS E&D fit:** **Very high.** This framing is exactly what E&D is for.

**NeurIPS main-track fit:** Moderate. Requires the finding to be either (a) a surprising negative result that overturns practice, or (b) a specific quantitative regime discovery ("dimensionality matters more than operators only above K=5") that nobody has shown before.

**Likely reviewer objections:** "What's the new algorithm?" (answer: none, and that is the point.) "Why not compare to SOTA agent frameworks?" (answer: SOTA agent frameworks conflate dimensionality and operator choice by design; our contribution is separating them.) "Are your benchmarks enough?" (answer: three benchmarks × two models × n=15 seeds is at or above the median of published work in this area.)

---

### Option 2 — The negative-result / simplification paper

**Strongest form.** "Fancy evolutionary operators are doing much less than practitioners think. We show on three real benchmarks and two open-weight models that, once the search space is rich enough, random search is statistically equivalent to full evolution, and the measured gains of multi-component search are entirely a search-space effect, not a search-operator effect. This has concrete practical implications: practitioners should spend their optimization budget expanding the set of genes, not building more complex operators."

**Required experiments:** Same as Option 1, **plus** an explicit equivalence-testing analysis that rules out meaningful operator effects with precise confidence bounds. Plus at least one "would simpler win?" case study where random search is shown to match full evolution on a hard benchmark cell.

**Risks:**
- **Highest risk: the finding may not replicate on real models.** Real LLMs have non-trivial landscape structure the synthetic pilot could not capture. If on real benchmarks full evolution significantly beats random search in some cells, this framing falls apart and we must retreat to Option 1.
- A reviewer says "your operator set is exhausted — what about LLM-guided crossover, learned mutation, etc.?" Mitigation: explicitly bound the claim to the operators tested; cite the missing families as future work.
- Equivalence-testing is still rare in the LLM community; some reviewers may not know TOST. Mitigation: explain it inline.

**Compute:** Identical to Option 1.

**Training / fine-tuning needed:** No.

**NeurIPS E&D fit:** High — E&D accepts rigorous negative results.

**NeurIPS main-track fit:** **High** — a clean, well-measured negative result on an active research topic is exactly the kind of paper that gets accepted to main track when the measurement is unimpeachable. Think of it as the LLM agent equivalent of Regularized Evolution.

**Likely reviewer objections:** "Maybe your operators are wrong." "Maybe your space is wrong." "Maybe 3 benchmarks is too narrow." All addressable with strong limitations + equivalence bounds.

---

### Option 3 — The learned-search paper

**Strongest form.** "We train a compact surrogate-fitness predictor on ~10k search trajectories, use it to rank candidate genomes, and show that a fitness-surrogate-guided search reaches the same final quality as full evolution at 1/4 the LLM evaluation budget. The surrogate is 200M parameters, trainable in a day on a single H100."

**Required experiments:** Everything in Option 1, plus:
- Collection of ~10k labeled (genome → fitness) pairs from Stage-B pilot runs.
- Training a small surrogate (ranker or regressor) with appropriate architecture.
- Running a surrogate-guided search loop (e.g. acquisition-function-style) at matched budget.
- Ablations: surrogate vs no-surrogate, size of training set, transfer of the surrogate across benchmarks.

**Risks:**
- **Scope explosion.** This is a second research project bolted onto the first. High chance of "too much engineering, not enough science."
- The learned component may not beat random search if the search space is already easy, which is exactly what Option 2 suggests. The learned-search branch is self-defeating if the operator null holds.
- Requires a training pipeline, data collection stage, and evaluation loop that does not exist in the pilot.

**Compute:** Option 1 + ~5 GPU-days for training + ~5 GPU-days for learned-search evaluation = ~35–40 GPU-days total. Tight against budget.

**Training / fine-tuning needed:** **Yes** (compact surrogate).

**NeurIPS E&D fit:** Low — this is a method paper, not an evaluation paper.

**NeurIPS main-track fit:** Moderate. If the surrogate works, it is a novel method. If it doesn't, the paper has no story.

**Likely reviewer objections:** "Your surrogate is too simple," "your training set is too small," "why not compare to BOHB / DEHB / SMAC?" These are answerable but expensive to address.

---

## Head-to-head comparison

| Criterion | Option 1 | Option 2 | Option 3 |
|---|---|---|---|
| Novelty (as framed) | Medium | High if result replicates | High if it works |
| Scientific defensibility | Highest | Very high | Lowest — depends on surrogate working |
| Compute cost | 25–30 GPU-days | 25–30 GPU-days | 35–40 GPU-days |
| Risk of null-story paper | Low | Medium | **High** |
| Fit to NeurIPS E&D | Perfect | Strong | Poor |
| Fit to NeurIPS main track | Moderate | **Strong if findings hold** | Moderate |
| Required training/FT | None | None | Surrogate (200M, ~1 day) |
| Time-to-draft from today | ~3 weeks | ~3 weeks | ~6 weeks |
| Risk of reviewer "not surprising" | Medium | Low if result is clean | Low |

---

## Decision

**Primary framing: Option 1 (evaluation paper)**, with the framing written flexibly enough that if the results replicate the pilot's operator-null finding on real data, the paper upgrades to Option 2's framing in the camera-ready revision.

**Backup framing: Option 2 (negative result)**, used if and only if the real-data results produce a clean operator-null pattern with equivalence bounds tight enough to rule out d_z > 0.2. If the operator null is noisy or only partial, we retreat to Option 1.

**Explicitly not pursued in v1: Option 3 (learned search).** It is valuable and we acknowledge that, but it is a second paper. The risks are too high to gate v1 on it. We will keep the harness surrogate-friendly so Option 3 can be executed as paper v2 without infrastructure rework.

### Why Option 1 as primary, not Option 2

Option 2 is the more interesting paper if its result is real. But we do not yet know if the result is real — the pilot evidence is from synthetic agents and has been thoroughly discredited. If we commit to Option 2 upfront and the real data says "actually full evolution does beat random search on 2 of 3 benchmarks," we end up rewriting the paper from scratch under deadline pressure. Option 1 makes the paper robust against any outcome of H2: if H2 holds, the paper is Option 2 in all but name; if H2 fails, the paper is still a valid E&D contribution.

### Why not Option 3

Three reasons:
1. **The surrogate depends on a strong evolution signal to be worth learning.** If Option 2's null result is correct (operators don't matter), then a surrogate is learning to match a fitness distribution that random search already explores efficiently. The surrogate's value depends on the operator ladder being meaningful, which is the very question Option 1/2 is testing.
2. **Scope creep is the single biggest risk to a NeurIPS deadline.** Every additional component doubles the chance of hitting a deadline wall.
3. **It can be paper v2.** Once we have ~10k labeled (genome, fitness) pairs from the real-study main matrix, training a surrogate in Q3 2026 is a low-risk follow-up paper, not a gating dependency.

---

## What the paper will look like (single-sentence summary)

> "Across three contamination-controlled real benchmarks (ForecastBench, LiveCodeBench v6, GPQA Diamond) and two open-weight models (Qwen3-32B, Llama-3.3-70B), we measure which components of agent configuration search contribute to final fitness at matched compute, and find that [search-space dimensionality / operator choice / budget / transfer] dominates — with preregistered mixed-effects aggregation, equivalence testing for null claims, and full per-run release."

The bracketed term is filled in after the real study runs. The paper commits to the question, the methodology, and the preregistered analysis, not to a specific answer.
