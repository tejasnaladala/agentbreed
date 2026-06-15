"""vLLM OpenAI-compatible REST client.

Connects to a locally-running `vllm serve` process on localhost:8000 (by default)
and issues generation calls through the OpenAI chat-completions endpoint. Uses
httpx.AsyncClient for concurrency-safe request handling.

This client intentionally does NOT start vLLM itself — that's done by the Slurm
job script (see harness/slurm/). The client assumes vLLM is already up and ready
when the Python process starts.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .base import ModelCallResult, ModelClient


@dataclass
class VLLMConfig:
    model: str                       # e.g. "Qwen/Qwen3-32B"
    base_url: str = "http://localhost:8000/v1"
    max_concurrent_requests: int = 16
    timeout_s: float = 120.0
    # Optional retry policy
    max_retries: int = 1
    retry_backoff_s: float = 2.0


class VLLMClient(ModelClient):
    """Async httpx client for a local vLLM OpenAI-compatible server."""

    def __init__(self, config: VLLMConfig):
        self._config = config
        self.model_name = config.model
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        # Lazy import so this module doesn't hard-require httpx at import time
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "httpx is required for VLLMClient. Install via `pip install httpx`."
            ) from e

    async def _client(self):
        import httpx
        if not hasattr(self, "_http") or self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=httpx.Timeout(self._config.timeout_s),
            )
        return self._http

    async def generate(
        self,
        *,
        system: Optional[str],
        user: str,
        max_tokens: int,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> ModelCallResult:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        body: Dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": 1.0,
        }
        if seed is not None:
            body["seed"] = int(seed)
        if stop:
            body["stop"] = list(stop)

        async with self._semaphore:
            for attempt in range(self._config.max_retries + 1):
                t0 = time.perf_counter()
                try:
                    client = await self._client()
                    resp = await client.post("/chat/completions", json=body)
                    elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    if resp.status_code != 200:
                        if attempt < self._config.max_retries:
                            await asyncio.sleep(self._config.retry_backoff_s)
                            continue
                        return ModelCallResult(
                            text="",
                            input_tokens=0,
                            output_tokens=0,
                            wall_time_ms=elapsed_ms,
                            model=self._config.model,
                            seed=seed,
                            temperature=temperature,
                            finish_reason="error",
                            error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                        )
                    payload = resp.json()
                    choice = payload["choices"][0]
                    text = choice["message"]["content"] or ""
                    usage = payload.get("usage", {})
                    return ModelCallResult(
                        text=text,
                        input_tokens=int(usage.get("prompt_tokens", 0)),
                        output_tokens=int(usage.get("completion_tokens", 0)),
                        wall_time_ms=elapsed_ms,
                        model=self._config.model,
                        seed=seed,
                        temperature=temperature,
                        finish_reason=choice.get("finish_reason", ""),
                        error=None,
                    )
                except Exception as e:  # network, timeout, parse
                    elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    if attempt < self._config.max_retries:
                        await asyncio.sleep(self._config.retry_backoff_s)
                        continue
                    return ModelCallResult(
                        text="",
                        input_tokens=0,
                        output_tokens=0,
                        wall_time_ms=elapsed_ms,
                        model=self._config.model,
                        seed=seed,
                        temperature=temperature,
                        finish_reason="exception",
                        error=f"{type(e).__name__}: {e}",
                    )

    async def close(self) -> None:
        if hasattr(self, "_http") and self._http is not None:
            await self._http.aclose()
            self._http = None
