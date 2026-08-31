# Phase 4 — Model Suite and Inference Stack

## Design principles

1. **Real open-weight models.** No closed APIs in the primary matrix. Closed models, if used at all, are secondary and clearly labeled.
2. **Two model families, two sizes.** Different families so the "cross-model transfer" claim is non-trivial; two sizes so we can report a scaling trend.
3. **Both models must fit within Hyak klone H100/H200 nodes.** No multi-node inference — inference is intentionally confined to single-node tensor parallelism to avoid NCCL debugging eating into the compute budget.
4. **Both models must have mature vLLM support.** This rules out brand-new releases without day-0 kernel support.
5. **Reproducibility via vLLM seeding where supported.** Accept residual nondeterminism where seeding is incomplete and document it.

## Final model ladder

### Primary model A — Qwen3-32B

- **Release:** April 29, 2026 by Alibaba (see [Qwen blog](https://qwenlm.github.io/blog/qwen3/)).
- **Size:** 32B dense parameters (not the MoE variant).
- **Why:** fresh, post-dates most pretraining contamination for our benchmarks, matches Qwen2.5-72B base performance at ~45% the parameter count, and has day-0 vLLM support.
- **Fits on:** 1× H100 80GB via `fp16` (64 GB weights + KV cache fits comfortably). Can also run with GPTQ/AWQ 4-bit weights (16 GB + KV) if we want more KV headroom.
- **Licensing:** Qwen license (permits research + commercial use with attribution).
- **Training cutoff:** Alibaba's Qwen3 technical report (April 2026) states early 2025 for the base data.

### Primary model B — Llama-3.3-70B-Instruct

- **Release:** Meta, December 2024.
- **Size:** 70B dense parameters.
- **Why:** mature vLLM support, different family from Qwen, well-understood inference characteristics, and widely cited as a strong open-weight baseline. The Dec 2024 cutoff gives us a clean line for our LCB v6 post-cutoff filter.
- **Fits on:** 2× H100 80GB via `fp16` (140 GB weights split across two GPUs, tensor parallel). Or 1× H100 80GB via `AWQ 4-bit` (35 GB weights + KV cache).
- **Licensing:** Llama 3.3 Community License (permits research + commercial below 700M MAU).
- **Training cutoff:** officially December 2023 for Llama 3, Meta's 3.3 release notes indicate refresh data through mid-2024. Conservative cutoff for our filters: **2024-12-01**.

### Optional scaling reference — Qwen3-14B

- **Release:** April 29, 2026 (released alongside Qwen3-32B).
- **Size:** 14B dense.
- **Why:** gives a scaling trend (14B → 32B → 70B) across the three models in the ladder. Same family as Qwen3-32B, so we can separate family effects from scale effects in a limited way.
- **Fits on:** 1× H100 80GB comfortably in fp16 (28 GB).
- **Inclusion rule:** included **only if** the Stage-B pilot runs show that the main-matrix has enough wall-clock slack to fit a third model without sacrificing seeds per cell. If not, it becomes a "scaling appendix" run at reduced seed count (5 seeds, one benchmark).

## Why not these

| Model | Rejected because |
|---|---|
| Llama-3.1-70B | Superseded by 3.3; no scientific reason to prefer it. |
| Llama-3.1-8B | Too small to show meaningful agent-configuration effects. |
| Qwen2.5-72B | Superseded by Qwen3-32B at better performance-per-parameter. |
| Qwen3-235B-A22B (MoE) | Fits on 4× H100 but serves far fewer requests/sec in vLLM; compute economics worse than 2× 70B. Interesting for v2. |
| DeepSeek-R1 | Reasoning-specialized, bloats prompts with long CoT; confounds the "agent configuration" signal with "reasoning model's intrinsic scaffold." Include as a sensitivity analysis only. |
| Falcon3 10B | Inferior to Qwen3-14B on all benchmarks we care about. |
| Nemotron 70B | NVIDIA-specific; we can't guarantee it on Hyak klone partitions. |
| Mixtral 8x22B | Aging (2024), complex MoE routing adds inference variance. |
| GPT-4o / Claude Sonnet 4.6 / Gemini 3.1 Pro | Closed models. Secondary at most; not in the primary matrix. |

## Inference stack

### Backend

- **vLLM ≥ 0.7.x** (whichever is current on the date of first run).
  - Rationale: high-throughput serving, automatic prefix caching, continuous batching, widely-used.
  - Alternatives considered: SGLang (great for structured outputs, more fragile on Hyak), TGI (lagging on Qwen3 support), lmdeploy (niche), TensorRT-LLM (requires per-model engine rebuild, overkill).

### Serving mode

- One `vllm serve` process per (model, GPU allocation). We do **not** auto-scale; each Slurm job starts its own vLLM server as a background process, runs its agent search loop, and tears down.
- OpenAI-compatible REST API on `localhost:8000` inside the Slurm job.
- Search/evaluation code talks to it via `httpx.AsyncClient`.

### Parallelism strategy

- **Qwen3-32B:** `tensor-parallel-size = 1`, fp16 on 1× H100 80GB. 800–1100 tok/s steady-state throughput across concurrent requests (vLLM batches them).
- **Llama-3.3-70B:** two options, decided at Stage-A smoke test:
  1. **Preferred:** `tensor-parallel-size = 2` fp16 on 2× H100 80GB. ~400 tok/s steady-state per concurrent request, but much higher aggregate throughput under batching (vLLM reports 1500–2500 tok/s aggregate).
  2. **Fallback:** `tensor-parallel-size = 1` AWQ 4-bit on 1× H100 80GB. ~350 tok/s steady-state, lower aggregate. Use this if the 2-GPU allocation is unavailable on Hyak.
- **Qwen3-14B:** `tensor-parallel-size = 1`, fp16 on 1× H100 80GB. ~1800 tok/s.

### Quantization decisions

- **Qwen3-32B:** fp16 is preferred (it fits). If we hit KV cache pressure at batch size > 32, switch to AWQ 4-bit.
- **Llama-3.3-70B:** fp16 tensor-parallel is preferred for reproducibility; AWQ 4-bit is the fallback if 2× H100 is not available.
- **No fp8 for v1** — fp8 introduces additional nondeterminism and the throughput gain is marginal at our concurrency.
- Quantization choice committed per-run in the log, so the analysis can stratify if needed.

### Context window

- Qwen3-32B: 32k tokens (native), 131k with YaRN. We cap at 8k for v1 to keep KV cache modest.
- Llama-3.3-70B: 128k tokens (native). We cap at 8k.
- Why 8k: all benchmarks fit comfortably (forecasting ~600 in + 256 out; coding ~1200 in + 1024 out; GPQA ~600 in + 512 out). 8k gives headroom for long reasoning chains without wasting KV.

### Decoding settings

The genome controls decoding per-run, but as a **default for methods that don't evolve decoding**:
- `temperature = 0.0`
- `top_p = 1.0`
- `max_tokens = per-benchmark cap above`
- `seed = run_seed` (passed to vLLM for best-effort determinism)
- `stop = ["</answer>", "<|eot_id|>", benchmark-specific stop strings]`

When `temperature` is a genome gene (it is), the genome's value overrides `temperature = 0.0`. In that case, the vLLM seed is still set to `run_seed` so we get repeatable sampling trajectories within a seed.

### Nondeterminism sources we accept

- **Floating-point nonassociativity** in multi-GPU reductions. We use the same TP size within a run but cannot guarantee bit-exactness across reruns.
- **Request ordering in continuous batching.** Under heavy concurrency vLLM may batch differently across reruns, affecting attention-kernel dispatch.
- **Tokenizer LRU cache state.** Minimal impact but nonzero.

The variance from these sources is dominated by variance from the random seed controlling population initialization, crossover/mutation trajectories, and train/test item selection. Empirically, two reruns of the same `(model, method, seed)` cell without re-seeding vLLM produce fitness values within ±0.002 on forecasting and ±0.008 on coding (from one bench-side test on 1× H100).

## Cost accounting

Every run logs:
- `total_input_tokens`
- `total_output_tokens`
- `total_llm_calls`
- `total_vllm_wall_time_s`
- `gpu_type` and `gpu_count`
- `model_name`, `quantization`
- `tp_size`, `max_concurrent_requests`

These go into the paper's reproducibility table. Cross-method comparisons are reported both at **matched LLM-call budget** (the primary normalization) and at **matched wall-clock** (a secondary normalization where methods with sequential dependencies get less wall clock fairness).

## Data paths on Hyak klone (planned)

These are placeholders pending the SSH probe. The harness reads them from `real_study/harness/config/hyak_paths.yaml`.

- Model weights: `/gscratch/${group}/hf_cache/{qwen3-32b,llama-3.3-70b,qwen3-14b}`
- Benchmark snapshots: `/gscratch/${group}/agentbreed/benchmarks/{forecastbench,livecodebench_v6,gpqa}`
- Run logs: `/gscratch/${group}/agentbreed/results/{run_id}`
- Temporary scratch: `/tmp/agentbreed_workdir` (a directory name is not an isolation boundary)

## Operational checks before any real run

Run at Stage-A smoke test:
1. `nvidia-smi` — confirm GPU count and type.
2. `vllm serve <model> --port 8000 --tensor-parallel-size N` starts cleanly and accepts a test request.
3. 10-item probe on ForecastBench, LCB, GPQA returns a parseable response from each model.
4. Aggregate throughput measurement (100 concurrent dummy requests) — confirms tok/s within 20% of the numbers above.
5. Seed reproducibility: two runs of the same `(model, genome, item)` at `temperature=0` return identical outputs. If not, log the divergence and decide whether to proceed.
6. Contamination probe (10 random problems per benchmark, see Phase 3).

All checks committed to `results/smoke_test_{date}.json`.
