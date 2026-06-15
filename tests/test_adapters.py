"""Tests for the adapter system: base, callable, anthropic, openai."""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from breed.adapters.base import AgentResult
from breed.adapters.callable_adapter import CallableAdapter
from breed.genome import Gene, GeneType, Genome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_genome(**gene_overrides: Any) -> Genome:
    """Build a minimal Genome with the three text genes and temperature."""
    genes: dict[str, Gene] = {
        "prompt_template": Gene(
            name="prompt_template",
            type=GeneType.TEXT,
            value="You are a helpful assistant.",
        ),
        "decision_heuristic": Gene(
            name="decision_heuristic",
            type=GeneType.TEXT,
            value="Use expected-value reasoning.",
        ),
        "calibration_rule": Gene(
            name="calibration_rule",
            type=GeneType.TEXT,
            value="Move 10% toward the base rate.",
        ),
        "temperature": Gene(
            name="temperature",
            type=GeneType.FLOAT,
            value=0.5,
            range=(0.0, 1.5),
        ),
        "tool_policy": Gene(
            name="tool_policy",
            type=GeneType.SET,
            value=["web_search", "calculator"],
            available=["web_search", "web_fetch", "calculator", "code_exec", "read", "write"],
        ),
    }
    genes.update(gene_overrides)
    return Genome(genes=genes)


# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------

class TestAgentResult:
    """Tests for the AgentResult dataclass."""

    def test_success_true_when_no_error(self) -> None:
        result = AgentResult(output="hello")
        assert result.success is True

    def test_success_false_when_error(self) -> None:
        result = AgentResult(output="", error="something broke")
        assert result.success is False

    def test_defaults(self) -> None:
        result = AgentResult(output="ok")
        assert result.metadata == {}
        assert result.cost_tokens == 0
        assert result.duration_ms == 0
        assert result.error is None

    def test_frozen(self) -> None:
        result = AgentResult(output="ok")
        with pytest.raises(AttributeError):
            result.output = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Adapter.build_system_prompt
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:
    """Tests for the base Adapter.build_system_prompt method."""

    def test_combines_three_text_genes(self) -> None:
        genome = _make_genome()
        # Use CallableAdapter as a concrete subclass to access build_system_prompt.
        adapter = CallableAdapter(fn=AsyncMock(return_value=""))
        prompt = adapter.build_system_prompt(genome)

        assert "You are a helpful assistant." in prompt
        assert "Use expected-value reasoning." in prompt
        assert "Move 10% toward the base rate." in prompt
        # Parts separated by double newlines.
        assert prompt.count("\n\n") == 2

    def test_missing_genes_skipped(self) -> None:
        genome = Genome(
            genes={
                "prompt_template": Gene(
                    name="prompt_template",
                    type=GeneType.TEXT,
                    value="Only this.",
                ),
            }
        )
        adapter = CallableAdapter(fn=AsyncMock(return_value=""))
        prompt = adapter.build_system_prompt(genome)
        assert prompt == "Only this."

    def test_empty_genome(self) -> None:
        genome = Genome(genes={})
        adapter = CallableAdapter(fn=AsyncMock(return_value=""))
        prompt = adapter.build_system_prompt(genome)
        assert prompt == ""


# ---------------------------------------------------------------------------
# CallableAdapter
# ---------------------------------------------------------------------------

class TestCallableAdapter:
    """Tests for the CallableAdapter."""

    def test_calls_fn_and_returns_result(self) -> None:
        async def my_fn(genome_dict: dict[str, Any], task: str) -> str:
            return f"answer for {task}"

        adapter = CallableAdapter(my_fn)
        genome = _make_genome()
        result = asyncio.get_event_loop().run_until_complete(
            adapter.run(genome, "What is 2+2?")
        )

        assert result.success is True
        assert result.output == "answer for What is 2+2?"
        assert result.error is None
        assert result.duration_ms >= 0

    def test_receives_genome_dict(self) -> None:
        captured: dict[str, Any] = {}

        async def capture_fn(genome_dict: dict[str, Any], task: str) -> str:
            captured.update(genome_dict)
            return "ok"

        adapter = CallableAdapter(capture_fn)
        genome = _make_genome()
        asyncio.get_event_loop().run_until_complete(adapter.run(genome, "task"))

        assert "genome_id" in captured
        assert "genes" in captured
        assert "prompt_template" in captured["genes"]

    def test_exception_wrapped_in_error(self) -> None:
        async def failing_fn(genome_dict: dict[str, Any], task: str) -> str:
            raise RuntimeError("boom")

        adapter = CallableAdapter(failing_fn)
        genome = _make_genome()
        result = asyncio.get_event_loop().run_until_complete(
            adapter.run(genome, "task")
        )

        assert result.success is False
        assert "RuntimeError" in result.error  # type: ignore[operator]
        assert "boom" in result.error  # type: ignore[operator]
        assert result.output == ""


# ---------------------------------------------------------------------------
# AnthropicMessagesAdapter -- import guard
# ---------------------------------------------------------------------------

class TestAnthropicImportGuard:
    """Verify that AnthropicMessagesAdapter raises clearly when SDK missing."""

    def test_raises_import_error_when_sdk_missing(self) -> None:
        # Temporarily hide the anthropic module and reload the adapter module.
        saved = sys.modules.get("anthropic")
        sys.modules["anthropic"] = None  # type: ignore[assignment]
        try:
            import breed.adapters.anthropic_adapter as mod

            # Force re-evaluation of the guard.
            mod._anthropic_available = False
            mod._anthropic_import_error = (
                "The 'anthropic' package is required for AnthropicMessagesAdapter. "
                "Install it with: pip install anthropic"
            )

            with pytest.raises(ImportError, match="pip install anthropic"):
                mod.AnthropicMessagesAdapter()
        finally:
            # Restore original state.
            if saved is not None:
                sys.modules["anthropic"] = saved
            else:
                sys.modules.pop("anthropic", None)

    def test_class_is_importable_without_sdk(self) -> None:
        """The class itself should be importable even without the SDK."""
        from breed.adapters.anthropic_adapter import AnthropicMessagesAdapter

        assert AnthropicMessagesAdapter is not None


# ---------------------------------------------------------------------------
# AnthropicMessagesAdapter -- mocked API call
# ---------------------------------------------------------------------------

class TestAnthropicAdapterRun:
    """Test AnthropicMessagesAdapter.run with a mocked Anthropic client."""

    def test_successful_run(self) -> None:
        import breed.adapters.anthropic_adapter as mod

        # Ensure the guard thinks SDK is available.
        original_available = mod._anthropic_available
        mod._anthropic_available = True

        try:
            # Build a mock response matching the Anthropic SDK shape.
            mock_text_block = MagicMock()
            mock_text_block.text = "The answer is 4"

            mock_response = MagicMock()
            mock_response.content = [mock_text_block]
            mock_response.model = "claude-sonnet-4-6"
            mock_response.stop_reason = "end_turn"
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50

            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)

            # Patch __init__ to skip real client creation.
            with patch.object(mod.AnthropicMessagesAdapter, "__init__", lambda self, **kw: None):
                adapter = mod.AnthropicMessagesAdapter()
                adapter._model = "claude-sonnet-4-6"
                adapter._client = mock_client

            genome = _make_genome()
            result = asyncio.get_event_loop().run_until_complete(
                adapter.run(genome, "What is 2+2?")
            )

            assert result.success is True
            assert result.output == "The answer is 4"
            assert result.cost_tokens == 150
            assert result.metadata["model"] == "claude-sonnet-4-6"
        finally:
            mod._anthropic_available = original_available

    def test_api_error_wrapped(self) -> None:
        import breed.adapters.anthropic_adapter as mod

        original_available = mod._anthropic_available
        mod._anthropic_available = True

        try:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(
                side_effect=RuntimeError("API timeout")
            )

            with patch.object(mod.AnthropicMessagesAdapter, "__init__", lambda self, **kw: None):
                adapter = mod.AnthropicMessagesAdapter()
                adapter._model = "claude-sonnet-4-6"
                adapter._client = mock_client

            genome = _make_genome()
            result = asyncio.get_event_loop().run_until_complete(
                adapter.run(genome, "task")
            )

            assert result.success is False
            assert "RuntimeError" in result.error  # type: ignore[operator]
            assert "API timeout" in result.error  # type: ignore[operator]
        finally:
            mod._anthropic_available = original_available


# ---------------------------------------------------------------------------
# OpenAIAdapter -- import guard
# ---------------------------------------------------------------------------

class TestOpenAIImportGuard:
    """Verify that OpenAIAdapter raises clearly when SDK missing."""

    def test_raises_import_error_when_sdk_missing(self) -> None:
        saved = sys.modules.get("openai")
        sys.modules["openai"] = None  # type: ignore[assignment]
        try:
            import breed.adapters.openai_adapter as mod

            mod._openai_available = False
            mod._openai_import_error = (
                "The 'openai' package is required for OpenAIAdapter. "
                "Install it with: pip install openai"
            )

            with pytest.raises(ImportError, match="pip install openai"):
                mod.OpenAIAdapter()
        finally:
            if saved is not None:
                sys.modules["openai"] = saved
            else:
                sys.modules.pop("openai", None)

    def test_class_is_importable_without_sdk(self) -> None:
        from breed.adapters.openai_adapter import OpenAIAdapter

        assert OpenAIAdapter is not None


# ---------------------------------------------------------------------------
# OpenAIAdapter -- mocked API call
# ---------------------------------------------------------------------------

class TestOpenAIAdapterRun:
    """Test OpenAIAdapter.run with a mocked OpenAI client."""

    def test_successful_run(self) -> None:
        import breed.adapters.openai_adapter as mod

        original_available = mod._openai_available
        mod._openai_available = True

        try:
            mock_message = MagicMock()
            mock_message.content = "The answer is 4"

            mock_choice = MagicMock()
            mock_choice.message = mock_message
            mock_choice.finish_reason = "stop"

            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 80
            mock_usage.completion_tokens = 20

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.model = "gpt-4o"
            mock_response.usage = mock_usage

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            with patch.object(mod.OpenAIAdapter, "__init__", lambda self, **kw: None):
                adapter = mod.OpenAIAdapter()
                adapter._model = "gpt-4o"
                adapter._client = mock_client

            genome = _make_genome()
            result = asyncio.get_event_loop().run_until_complete(
                adapter.run(genome, "What is 2+2?")
            )

            assert result.success is True
            assert result.output == "The answer is 4"
            assert result.cost_tokens == 100
            assert result.metadata["model"] == "gpt-4o"
        finally:
            mod._openai_available = original_available

    def test_api_error_wrapped(self) -> None:
        import breed.adapters.openai_adapter as mod

        original_available = mod._openai_available
        mod._openai_available = True

        try:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=RuntimeError("rate limited")
            )

            with patch.object(mod.OpenAIAdapter, "__init__", lambda self, **kw: None):
                adapter = mod.OpenAIAdapter()
                adapter._model = "gpt-4o"
                adapter._client = mock_client

            genome = _make_genome()
            result = asyncio.get_event_loop().run_until_complete(
                adapter.run(genome, "task")
            )

            assert result.success is False
            assert "RuntimeError" in result.error  # type: ignore[operator]
            assert "rate limited" in result.error  # type: ignore[operator]
        finally:
            mod._openai_available = original_available
