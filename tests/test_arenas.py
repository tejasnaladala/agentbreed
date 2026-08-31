"""Tests for the arenas system: forecasting, coding, and custom arenas."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from breed.adapters.base import Adapter, AgentResult
from breed.arenas.base import EvalResult, Task
from breed.arenas.coding import CodingArena, extract_code
from breed.arenas.custom import CustomArena
from breed.arenas.forecasting import ForecastingArena, parse_probability
from breed.genome import Genome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_genome() -> Genome:
    """Create a minimal genome for testing."""
    return Genome(genome_id="test-genome-001", generation=0)


def _make_mock_adapter(output: str, cost_tokens: int = 10) -> Adapter:
    """Create a mock adapter that always returns the given output."""
    adapter = AsyncMock(spec=Adapter)
    adapter.run.return_value = AgentResult(
        output=output,
        cost_tokens=cost_tokens,
    )
    return adapter


def _make_error_adapter() -> Adapter:
    """Create a mock adapter that always returns an error."""
    adapter = AsyncMock(spec=Adapter)
    adapter.run.return_value = AgentResult(
        output="",
        error="Simulated failure",
        cost_tokens=0,
    )
    return adapter


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------


class TestTaskDataclass:
    """Tests for the Task dataclass."""

    def test_task_has_correct_fields(self) -> None:
        task = Task(task_id="t1", prompt="What is 2+2?", expected=4)
        assert task.task_id == "t1"
        assert task.prompt == "What is 2+2?"
        assert task.expected == 4
        assert task.metadata == {}

    def test_task_with_metadata(self) -> None:
        task = Task(
            task_id="t2",
            prompt="test",
            expected=None,
            metadata={"domain": "math"},
        )
        assert task.metadata == {"domain": "math"}

    def test_task_is_frozen(self) -> None:
        task = Task(task_id="t1", prompt="test")
        with pytest.raises(AttributeError):
            task.task_id = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ForecastingArena
# ---------------------------------------------------------------------------


class TestForecastingArenaGenerateTasks:
    """Tests for ForecastingArena.generate_tasks."""

    async def test_generates_correct_count(self) -> None:
        arena = ForecastingArena()
        tasks = await arena.generate_tasks(count=5, seed=42)
        assert len(tasks) == 5

    async def test_generates_more_than_pool_with_replacement(self) -> None:
        arena = ForecastingArena()
        tasks = await arena.generate_tasks(count=50, seed=42)
        assert len(tasks) == 50

    async def test_tasks_have_expected_outcomes(self) -> None:
        arena = ForecastingArena()
        tasks = await arena.generate_tasks(count=10, seed=42)
        for task in tasks:
            assert task.expected in (0.0, 1.0)

    async def test_seed_produces_deterministic_results(self) -> None:
        arena = ForecastingArena()
        tasks_a = await arena.generate_tasks(count=5, seed=123)
        tasks_b = await arena.generate_tasks(count=5, seed=123)
        for a, b in zip(tasks_a, tasks_b):
            assert a.prompt == b.prompt
            assert a.expected == b.expected

    async def test_custom_questions(self) -> None:
        custom_qs = [
            {"prompt": "Will it rain?", "outcome": 1.0},
            {"prompt": "Will it snow?", "outcome": 0.0},
        ]
        arena = ForecastingArena(questions=custom_qs)
        tasks = await arena.generate_tasks(count=2, seed=0)
        assert len(tasks) == 2
        prompts = {t.prompt for t in tasks}
        assert prompts <= {"Will it rain?", "Will it snow?"}


class TestParseProbability:
    """Tests for parse_probability across various formats."""

    def test_plain_float(self) -> None:
        assert parse_probability("0.75") == 0.75

    def test_percentage(self) -> None:
        assert parse_probability("75%") == 0.75

    def test_percentage_with_space(self) -> None:
        assert parse_probability("75 %") == 0.75

    def test_text_with_percentage(self) -> None:
        result = parse_probability("I estimate 75% probability")
        assert result == 0.75

    def test_json_object(self) -> None:
        result = parse_probability('{"probability": 0.82}')
        assert result == 0.82

    def test_json_with_alt_key(self) -> None:
        result = parse_probability('{"prob": 0.6}')
        assert result == 0.6

    def test_json_plain_number(self) -> None:
        result = parse_probability("0.9")
        assert result == 0.9

    def test_empty_string_fallback(self) -> None:
        assert parse_probability("") == 0.5

    def test_unparseable_fallback(self) -> None:
        assert parse_probability("I have no idea whatsoever") == 0.5

    def test_clamps_above_one(self) -> None:
        # Percentage over 100 should clamp
        result = parse_probability("150%")
        assert result == 1.0

    def test_zero_probability(self) -> None:
        assert parse_probability("0.0") == 0.0

    def test_one_probability(self) -> None:
        assert parse_probability("1.0") == 1.0


class TestForecastingArenaEvaluate:
    """Tests for ForecastingArena.evaluate with a mock adapter."""

    async def test_evaluate_produces_valid_result(self) -> None:
        arena = ForecastingArena()
        genome = _make_genome()
        adapter = _make_mock_adapter("0.80")
        tasks = await arena.generate_tasks(count=5, seed=42)

        result = await arena.evaluate(genome, adapter, tasks)

        assert isinstance(result, EvalResult)
        assert result.genome_id == "test-genome-001"
        assert 0.0 <= result.fitness_score <= 1.0
        assert "brier_fitness" in result.fitness_breakdown
        assert "calibration_error" in result.fitness_breakdown
        assert len(result.task_results) == 5
        assert result.cost_tokens == 50  # 5 tasks * 10 tokens each

    async def test_evaluate_perfect_predictions(self) -> None:
        """An agent that outputs perfect probabilities should score well."""
        questions = [
            {"prompt": "Q1", "outcome": 1.0},
            {"prompt": "Q2", "outcome": 0.0},
        ]
        arena = ForecastingArena(questions=questions)
        tasks = await arena.generate_tasks(count=2, seed=0)

        # Build adapter that returns the correct probability for each task
        # by inspecting the task prompt to determine the right answer.
        outcome_by_prompt = {"Q1": 1.0, "Q2": 0.0}

        async def _perfect_agent(_genome: Any, prompt: str) -> AgentResult:
            prob = outcome_by_prompt.get(prompt, 0.5)
            return AgentResult(output=str(prob), cost_tokens=5)

        adapter = AsyncMock(spec=Adapter)
        adapter.run.side_effect = _perfect_agent

        genome = _make_genome()
        result = await arena.evaluate(genome, adapter, tasks)

        # Perfect predictions: Brier score = 0, fitness = 1.0
        assert result.fitness_score == 1.0

    async def test_evaluate_with_adapter_errors(self) -> None:
        """On adapter error, probability defaults to 0.5."""
        arena = ForecastingArena()
        genome = _make_genome()
        adapter = _make_error_adapter()
        tasks = await arena.generate_tasks(count=3, seed=42)

        result = await arena.evaluate(genome, adapter, tasks)

        assert isinstance(result, EvalResult)
        assert 0.0 <= result.fitness_score <= 1.0


# ---------------------------------------------------------------------------
# CodingArena
# ---------------------------------------------------------------------------


class TestCodingArenaGenerateTasks:
    """Tests for CodingArena.generate_tasks."""

    async def test_generates_correct_count(self) -> None:
        arena = CodingArena()
        tasks = await arena.generate_tasks(count=5, seed=42)
        assert len(tasks) == 5

    async def test_tasks_have_function_name_and_tests(self) -> None:
        arena = CodingArena()
        tasks = await arena.generate_tasks(count=3, seed=42)
        for task in tasks:
            assert "function_name" in task.expected
            assert "tests" in task.expected
            assert len(task.expected["tests"]) > 0

    async def test_seed_determinism(self) -> None:
        arena = CodingArena()
        a = await arena.generate_tasks(count=5, seed=99)
        b = await arena.generate_tasks(count=5, seed=99)
        for ta, tb in zip(a, b):
            assert ta.prompt == tb.prompt


class TestExtractCode:
    """Tests for extract_code helper."""

    def test_python_fenced_block(self) -> None:
        text = "Here is my solution:\n```python\ndef foo():\n    return 42\n```\nDone."
        assert "def foo():" in extract_code(text)

    def test_generic_fenced_block(self) -> None:
        text = "```\ndef bar():\n    pass\n```"
        assert "def bar():" in extract_code(text)

    def test_raw_code(self) -> None:
        text = "def baz():\n    return 1"
        assert "def baz():" in extract_code(text)

    def test_empty_input(self) -> None:
        assert extract_code("") == ""


# ---------------------------------------------------------------------------
# CustomArena
# ---------------------------------------------------------------------------


class TestCustomArena:
    """Tests for CustomArena with user-provided scorer."""

    async def test_uses_provided_scorer(self) -> None:
        """Scorer that checks exact match should work correctly."""

        def exact_match_scorer(output: str, expected: Any) -> float:
            return 1.0 if output.strip() == str(expected) else 0.0

        tasks = [
            Task(task_id="c1", prompt="What is 2+2?", expected="4"),
            Task(task_id="c2", prompt="What is 3+3?", expected="6"),
        ]
        arena = CustomArena(tasks=tasks, scorer=exact_match_scorer)

        genome = _make_genome()
        adapter = _make_mock_adapter("4")  # correct for first, wrong for second

        # generate_tasks returns the stored tasks
        eval_tasks = await arena.generate_tasks(count=0)
        result = await arena.evaluate(genome, adapter, eval_tasks)

        # "4" matches expected "4" for task 1, but not "6" for task 2 -> 0.5
        assert result.fitness_score == 0.5

    async def test_scorer_receives_correct_arguments(self) -> None:
        """Verify scorer is called with (agent_output, expected)."""
        calls: list[tuple[str, Any]] = []

        def tracking_scorer(output: str, expected: Any) -> float:
            calls.append((output, expected))
            return 0.8

        tasks = [Task(task_id="t1", prompt="test", expected="answer")]
        arena = CustomArena(tasks=tasks, scorer=tracking_scorer)
        genome = _make_genome()
        adapter = _make_mock_adapter("agent says hello")

        eval_tasks = await arena.generate_tasks(count=0)
        result = await arena.evaluate(genome, adapter, eval_tasks)

        assert len(calls) == 1
        assert calls[0] == ("agent says hello", "answer")
        assert result.fitness_score == 0.8

    async def test_generate_tasks_returns_stored_tasks(self) -> None:
        tasks = [
            Task(task_id="x1", prompt="p1"),
            Task(task_id="x2", prompt="p2"),
        ]
        arena = CustomArena(tasks=tasks, scorer=lambda o, e: 0.0)
        result = await arena.generate_tasks(count=99)
        assert len(result) == 2
        assert result[0].task_id == "x1"

    async def test_scorer_output_clamped(self) -> None:
        """Scorer returning out-of-range values should be clamped to [0, 1]."""

        def bad_scorer(output: str, expected: Any) -> float:
            return 2.5  # out of range

        tasks = [Task(task_id="t1", prompt="test", expected=None)]
        arena = CustomArena(tasks=tasks, scorer=bad_scorer)
        genome = _make_genome()
        adapter = _make_mock_adapter("anything")

        eval_tasks = await arena.generate_tasks(count=0)
        result = await arena.evaluate(genome, adapter, eval_tasks)

        assert result.fitness_score <= 1.0

    async def test_adapter_error_scores_zero(self) -> None:
        def scorer(output: str, expected: Any) -> float:
            return 1.0

        tasks = [Task(task_id="t1", prompt="test", expected=None)]
        arena = CustomArena(tasks=tasks, scorer=scorer)
        genome = _make_genome()
        adapter = _make_error_adapter()

        eval_tasks = await arena.generate_tasks(count=0)
        result = await arena.evaluate(genome, adapter, eval_tasks)

        assert result.fitness_score == 0.0


# ---------------------------------------------------------------------------
# EvalResult dataclass
# ---------------------------------------------------------------------------


class TestEvalResult:
    """Tests for the EvalResult dataclass."""

    def test_eval_result_fields(self) -> None:
        result = EvalResult(
            genome_id="g1",
            fitness_score=0.85,
            fitness_breakdown={"brier": 0.85},
            cost_tokens=100,
        )
        assert result.genome_id == "g1"
        assert result.fitness_score == 0.85
        assert result.fitness_breakdown == {"brier": 0.85}
        assert result.cost_tokens == 100
        assert result.task_results == []

    def test_eval_result_is_frozen(self) -> None:
        result = EvalResult(genome_id="g1", fitness_score=0.5)
        with pytest.raises(AttributeError):
            result.fitness_score = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# evaluate_population (base class default)
# ---------------------------------------------------------------------------


class TestEvaluatePopulation:
    """Tests for Arena.evaluate_population default implementation."""

    async def test_evaluates_all_genomes(self) -> None:
        arena = ForecastingArena()
        genomes = [_make_genome() for _ in range(3)]
        adapter = _make_mock_adapter("0.5")
        tasks = await arena.generate_tasks(count=5, seed=42)

        results = await arena.evaluate_population(genomes, adapter, tasks)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, EvalResult)
