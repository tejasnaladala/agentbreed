"""Concurrent throughput probe for the vLLM server.

Issues N concurrent identical requests to the vLLM server, measures total
wall-clock and aggregate tokens/second. Used at Stage A2 and A3 to verify
the model serves at the throughput the compute plan assumes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from real_study.harness.models.vllm_client import VLLMClient, VLLMConfig  # noqa: E402


async def _one_call(client: VLLMClient, i: int, max_tokens: int) -> dict:
    result = await client.generate(
        system="You are a concise assistant.",
        user=f"Count from 1 to {10 + (i % 5)} in plain digits separated by commas. No other text.",
        max_tokens=max_tokens,
        temperature=0.0,
        seed=i,
    )
    return {
        "i": i,
        "ok": result.ok,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "wall_time_ms": result.wall_time_ms,
        "error": result.error,
    }


async def _run(args) -> dict:
    cfg = VLLMConfig(
        model=args.model,
        base_url=args.endpoint,
        max_concurrent_requests=args.concurrent,
        timeout_s=120.0,
    )
    client = VLLMClient(cfg)
    try:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[
            _one_call(client, i, args.max_tokens) for i in range(args.n_calls)
        ])
        total_wall = time.perf_counter() - t0
    finally:
        await client.close()

    ok_results = [r for r in results if r["ok"]]
    total_out = sum(r["output_tokens"] for r in ok_results)
    total_in = sum(r["input_tokens"] for r in ok_results)
    per_call_wall = [r["wall_time_ms"] / 1000.0 for r in ok_results]

    report = {
        "model": args.model,
        "endpoint": args.endpoint,
        "concurrent": args.concurrent,
        "n_calls_requested": args.n_calls,
        "n_calls_ok": len(ok_results),
        "n_calls_failed": args.n_calls - len(ok_results),
        "max_tokens": args.max_tokens,
        "total_wall_s": round(total_wall, 3),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "aggregate_output_tokens_per_s": round(total_out / total_wall, 1) if total_wall > 0 else 0,
        "per_call_wall_s_mean": round(statistics.mean(per_call_wall), 3) if per_call_wall else None,
        "per_call_wall_s_p50": round(statistics.median(per_call_wall), 3) if per_call_wall else None,
        "per_call_wall_s_p95": round(sorted(per_call_wall)[int(0.95 * len(per_call_wall))], 3)
        if len(per_call_wall) >= 20 else None,
    }

    print(f"[throughput] n={report['n_calls_ok']}/{report['n_calls_requested']}, "
          f"wall={report['total_wall_s']:.1f}s, "
          f"agg_tok_per_s={report['aggregate_output_tokens_per_s']}, "
          f"p50 per-call={report['per_call_wall_s_p50']}s")

    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--concurrent", type=int, default=16)
    p.add_argument("--n-calls", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    report = asyncio.run(_run(args))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[throughput] wrote {out_path}")


if __name__ == "__main__":
    main()
