"""Adapters translate genomes into framework-specific agent configs."""

from breed.adapters.base import Adapter, AgentResult
from breed.adapters.callable_adapter import CallableAdapter

__all__ = ["Adapter", "AgentResult", "CallableAdapter"]

# Optional SDK adapters -- available when their SDK is installed
try:
    from breed.adapters.anthropic_adapter import AnthropicMessagesAdapter

    __all__.append("AnthropicMessagesAdapter")
except ImportError:
    pass

try:
    from breed.adapters.openai_adapter import OpenAIAdapter

    __all__.append("OpenAIAdapter")
except ImportError:
    pass
