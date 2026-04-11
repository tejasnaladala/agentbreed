"""LLM model clients for the real study.

The primary client is `VLLMClient`, which talks to a `vllm serve` process via
its OpenAI-compatible REST API on localhost. The client is designed for:
- high concurrency (async httpx)
- deterministic seeding where supported
- clean per-call logging (input_tokens, output_tokens, wall_time_ms)
- graceful retry on transient errors
"""

from .base import ModelClient, ModelCallResult
from .vllm_client import VLLMClient, VLLMConfig

__all__ = ["ModelClient", "ModelCallResult", "VLLMClient", "VLLMConfig"]
