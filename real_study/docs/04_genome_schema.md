# Phase 5 — Agent Genome Schema

## Design principles

1. **Scientifically legible.** Every gene has a name, a type, a legal value set, and a category. No "magic numbers."
2. **Orthogonal categories.** Genes are partitioned into four categories (semantic / control-flow / tool-memory / compute-budget) so ablations can test category-specific hypotheses.
3. **Universal vs benchmark-specific.** Some genes have the same legal values across all benchmarks (e.g. temperature). Some have benchmark-specific value sets (e.g. output format). This is made explicit.
4. **Large enough to matter, small enough to reason about.** Nine genes — the same count as the pilot — so the dimensionality-sweep design carries over.
5. **Every legal value is executable.** No genes that look nice in a table but break at runtime.

## Genome structure

A genome `g` is a tuple `(g_1, ..., g_9)` of nine typed genes. Each gene has:

- **Name** (string, unique).
- **Type** (one of `ENUM`, `SET`, `FLOAT`, `BOOL`).
- **Category** (semantic / control_flow / tool_memory / compute_budget).
- **Scope** (universal / benchmark_specific).
- **Legal value set** (for ENUM/SET) or range (for FLOAT) or {true, false} for BOOL.
- **Default value** (used for "gene frozen" conditions in ablations).

## The 9 genes (v1)

### Semantic / reasoning genes (3)

#### Gene 1 — `system_prompt`
- **Type:** ENUM
- **Category:** semantic
- **Scope:** benchmark-specific
- **Legal values (per benchmark):**
  - ForecastBench: `["plain_forecaster", "calibrated_base_rates", "scenario_planner", "devils_advocate", "ensemble_of_views"]`
  - LiveCodeBench: `["plain_coder", "test_driven", "stepwise_algorithm", "edge_case_hunter", "explain_then_code"]`
  - GPQA Diamond: `["plain_scientist", "first_principles", "expert_in_field", "socratic_self_check", "compare_and_contrast"]`
- **Default:** `plain_*`
- **What it controls:** the role-framing and opening instructions of the system message.

#### Gene 2 — `decomposition_style`
- **Type:** ENUM
- **Category:** semantic
- **Scope:** universal
- **Legal values:** `["none", "chain_of_thought", "least_to_most", "tree_of_thought_depth_2", "self_ask"]`
- **Default:** `none`
- **What it controls:** whether and how the agent decomposes the problem before answering. `chain_of_thought` inserts "Let's think step by step" in the prompt; `tree_of_thought_depth_2` issues the LLM a 2-level branching instruction; etc.

#### Gene 3 — `self_critique`
- **Type:** ENUM
- **Category:** semantic
- **Scope:** universal
- **Legal values:** `["off", "single_pass_critique", "reflexion_1turn", "rebuttal_debate"]`
- **Default:** `off`
- **What it controls:** whether the agent self-checks its answer once inside the same call. `single_pass_critique` adds "Now critique your answer and fix any errors" within the same prompt; `reflexion_1turn` is a two-phase construct (the first phase writes a tentative answer, the second phase is gated inside the output format); `rebuttal_debate` instructs the agent to argue both sides. **All implemented within a single LLM call** (no multi-turn) for v1.

### Control-flow genes (2)

#### Gene 4 — `answer_format`
- **Type:** ENUM
- **Category:** control_flow
- **Scope:** benchmark-specific
- **Legal values:**
  - ForecastBench: `["bare_number", "json_probability", "markdown_reasoning_then_number", "percent_phrase"]`
  - LiveCodeBench: `["fenced_python", "markdown_code_then_explanation", "plain_function_only", "comment_then_code"]`
  - GPQA Diamond: `["bare_letter", "after_answer_marker", "reasoning_then_letter", "json_with_letter"]`
- **Default:** whichever is easiest for the default parser.
- **What it controls:** the expected output format that the parser looks for. Critically, this is the gene that couples to the benchmark scorer — if the format doesn't match, the parser falls back and the score suffers.

#### Gene 5 — `stopping_policy`
- **Type:** ENUM
- **Category:** control_flow
- **Scope:** universal
- **Legal values:** `["default_eos", "strict_short", "strict_medium", "strict_long"]`
- **Default:** `default_eos`
- **What it controls:** the `max_tokens` and `stop` parameters at the vLLM level:
  - `default_eos`: max_tokens = benchmark default, stop = []
  - `strict_short`: max_tokens = benchmark_default // 2, stop = ["\n\n"]
  - `strict_medium`: max_tokens = benchmark_default, stop = ["</answer>", "```\n\n"]
  - `strict_long`: max_tokens = 2 × benchmark_default, stop = []

### Tool / memory genes (2)

#### Gene 6 — `memory_structure`
- **Type:** ENUM
- **Category:** tool_memory
- **Scope:** universal
- **Legal values:** `["stateless", "running_scratchpad_in_prompt", "retrieved_similar_past_items"]`
- **Default:** `stateless`
- **What it controls:**
  - `stateless`: no memory across items within the same run.
  - `running_scratchpad_in_prompt`: a trimmed summary of previous answers is prepended to each new item's prompt (bounded to 512 tokens).
  - `retrieved_similar_past_items`: the agent retrieves the `k=3` most similar previous items (by a simple embedding similarity using `sentence-transformers/all-MiniLM-L6-v2`) and prepends their Q+A as few-shot context.

Note: retrieval and scratchpad update the prompt but do NOT add extra LLM calls — the added tokens are billed to the single call's input-token budget. This keeps budget-matching clean.

#### Gene 7 — `tool_policy`
- **Type:** SET (multi-hot over a universe)
- **Category:** tool_memory
- **Scope:** benchmark-specific
- **Universe (per benchmark):**
  - ForecastBench: `{base_rate_lookup, analogy_finder, scenario_tree}`
  - LiveCodeBench: `{run_example_tests_inline, type_check_inline, edge_case_generator}`
  - GPQA Diamond: `{formula_lookup, definition_lookup, unit_converter}`
- **Default:** `{}` (empty set)
- **What it controls:** a multi-hot bitmask over pseudo-tools. These are **in-prompt instructions** that ask the LLM to perform the tool's function before answering (not real external tools — no web search, no calculator execution). Keeps budget-matching clean while still exercising "tool-selection" as a searchable axis. An exploratory extension with **real external tools** is deferred to v2.

### Compute-budget genes (2)

#### Gene 8 — `temperature`
- **Type:** FLOAT
- **Category:** compute_budget
- **Scope:** universal
- **Range:** `[0.0, 1.2]`, discretized to 13 values `{0.0, 0.1, 0.2, ..., 1.2}` for the discrete-space baselines (random search, coordinate descent) and continuous for the evolutionary methods.
- **Default:** `0.0`
- **What it controls:** vLLM sampling temperature. Passed through directly.

#### Gene 9 — `prompt_token_budget`
- **Type:** ENUM (effective FLOAT discretized)
- **Category:** compute_budget
- **Scope:** universal
- **Legal values:** `["tight_512", "medium_1024", "large_2048", "xlarge_4096"]`
- **Default:** `medium_1024`
- **What it controls:** the maximum input tokens the prompt is allowed to use. Sets the cap for in-prompt content — e.g. how much scratchpad or retrieval can be prepended. If the constructed prompt exceeds the budget, content is truncated from the least-critical end (retrieved items first, scratchpad second, system prompt last).

## Category summary

| Category | Genes | Count |
|---|---|---|
| Semantic | `system_prompt`, `decomposition_style`, `self_critique` | 3 |
| Control flow | `answer_format`, `stopping_policy` | 2 |
| Tool / memory | `memory_structure`, `tool_policy` | 2 |
| Compute budget | `temperature`, `prompt_token_budget` | 2 |

## Legal search space size

On a single benchmark:
- `system_prompt`: 5 values
- `decomposition_style`: 5
- `self_critique`: 4
- `answer_format`: 4
- `stopping_policy`: 4
- `memory_structure`: 3
- `tool_policy`: 2^3 = 8
- `temperature` (discretized): 13
- `prompt_token_budget`: 4

Product: 5 × 5 × 4 × 4 × 4 × 3 × 8 × 13 × 4 ≈ **1.99 × 10^6 discrete configurations.** Much larger than the pilot (~10^5) and large enough that exhaustive search is infeasible (as it should be).

## Benchmark-specific vs universal genes

Five genes (`decomposition_style`, `self_critique`, `stopping_policy`, `memory_structure`, `temperature`, `prompt_token_budget` — actually six) are **universal**: same legal value set across all three benchmarks. Three genes (`system_prompt`, `answer_format`, `tool_policy`) are **benchmark-specific**: the legal values depend on the target benchmark.

This matters for cross-benchmark transfer (secondary hypothesis S2 in Phase 9): a champion on one benchmark can only transfer its **universal** gene values directly. For benchmark-specific genes, we map them onto the target benchmark's legal set using a documented mapping or the default if no mapping exists.

**Mapping rule for transfer:**
- For universal genes: copy verbatim.
- For benchmark-specific genes: use the default value on the target benchmark.

## Implementation contract

The schema lives in `real_study/harness/genome/schema.py`. The validator (`real_study/harness/genome/validator.py`) enforces:

1. Every gene value is in its legal set.
2. Benchmark-specific gene values match the declared benchmark.
3. On transfer, the mapping rule is applied and logged.
4. The content hash of a genome is stable under tuple-ordering.

## Ablation handles built into the schema

The schema supports these ablations without code changes:
- **Category ablations:** freeze all genes in one category at defaults, vary the others. Supports "does the semantic category alone suffice?" etc.
- **Dimensionality sweep:** take the first K genes of a canonical order (the sweep order is fixed below), freeze the rest at defaults.
- **Universal-only:** freeze all benchmark-specific genes at defaults, so the transfer matrix becomes a direct cross-benchmark comparison.

**Canonical sweep order for the dimensionality experiment (fixed before runs):**
1. `system_prompt`
2. `decomposition_style`
3. `answer_format`
4. `self_critique`
5. `memory_structure`
6. `tool_policy`
7. `stopping_policy`
8. `temperature`
9. `prompt_token_budget`

This order is chosen to place the most impactful genes first so the sweep at K=3 already gives reasonable performance, and the diminishing returns at K=9 are visible but not zero.

## What we are NOT putting in the genome (v1)

- **Real external tools** (web search, code execution, calculator). No execution sandbox exists in the current codebase, and these tools have too many failure modes. Belongs in a future version only after OS-level isolation is implemented and verified.
- **Multi-turn dialogue with the LLM.** One call per item is the v1 rule.
- **Fine-tuning or LoRA adapters as genes.** Out of scope.
- **Prompt ensembles / self-consistency voting.** A gene that evolves "number of voters" would break budget matching. Out of scope for v1.
- **Continuous-free-text prompt genes.** We use ENUM over a fixed set of 5 prompt templates per benchmark. Evolving free-text prompts via LLM mutation is interesting but belongs in the learned-search follow-up, not v1.
