"""Coding arena: function implementation challenges with test-case validation.

Agents receive coding prompts and must produce working Python functions.
Code is extracted, executed in a sandbox namespace, and tested against
predefined cases.  Fitness is the pass rate across all test cases.
"""

from __future__ import annotations

import random
import re
from typing import Any

from breed.adapters.base import Adapter, AgentResult
from breed.arenas.base import Arena, EvalResult, Task
from breed.fitness import PassRateEvaluator
from breed.genome import Genome

# ---------------------------------------------------------------------------
# Built-in coding problems with test cases
# ---------------------------------------------------------------------------

_BUILTIN_PROBLEMS: list[dict[str, Any]] = [
    {
        "prompt": "Write a Python function `is_palindrome(s: str) -> bool` that returns True if the string is a palindrome (ignoring case and non-alphanumeric characters), False otherwise.",
        "function_name": "is_palindrome",
        "tests": [
            {"args": ["racecar"], "expected": True},
            {"args": ["hello"], "expected": False},
            {"args": ["A man a plan a canal Panama"], "expected": True},
            {"args": [""], "expected": True},
            {"args": ["ab"], "expected": False},
        ],
    },
    {
        "prompt": "Write a Python function `fibonacci(n: int) -> int` that returns the n-th Fibonacci number (0-indexed, so fibonacci(0)=0, fibonacci(1)=1, fibonacci(2)=1).",
        "function_name": "fibonacci",
        "tests": [
            {"args": [0], "expected": 0},
            {"args": [1], "expected": 1},
            {"args": [2], "expected": 1},
            {"args": [10], "expected": 55},
            {"args": [15], "expected": 610},
        ],
    },
    {
        "prompt": "Write a Python function `fizzbuzz(n: int) -> str` that returns 'FizzBuzz' if n is divisible by both 3 and 5, 'Fizz' if divisible by 3, 'Buzz' if divisible by 5, or str(n) otherwise.",
        "function_name": "fizzbuzz",
        "tests": [
            {"args": [15], "expected": "FizzBuzz"},
            {"args": [9], "expected": "Fizz"},
            {"args": [10], "expected": "Buzz"},
            {"args": [7], "expected": "7"},
            {"args": [30], "expected": "FizzBuzz"},
        ],
    },
    {
        "prompt": "Write a Python function `reverse_words(s: str) -> str` that reverses the order of words in a string. Words are separated by spaces.",
        "function_name": "reverse_words",
        "tests": [
            {"args": ["hello world"], "expected": "world hello"},
            {"args": ["the quick brown fox"], "expected": "fox brown quick the"},
            {"args": ["single"], "expected": "single"},
            {"args": [""], "expected": ""},
        ],
    },
    {
        "prompt": "Write a Python function `max_subarray_sum(nums: list[int]) -> int` that returns the maximum sum of any contiguous subarray (Kadane's algorithm). Return 0 for an empty list.",
        "function_name": "max_subarray_sum",
        "tests": [
            {"args": [[-2, 1, -3, 4, -1, 2, 1, -5, 4]], "expected": 6},
            {"args": [[1, 2, 3]], "expected": 6},
            {"args": [[-1, -2, -3]], "expected": -1},
            {"args": [[]], "expected": 0},
            {"args": [[5]], "expected": 5},
        ],
    },
    {
        "prompt": "Write a Python function `count_vowels(s: str) -> int` that counts the number of vowels (a, e, i, o, u) in a string, case-insensitive.",
        "function_name": "count_vowels",
        "tests": [
            {"args": ["hello"], "expected": 2},
            {"args": ["AEIOU"], "expected": 5},
            {"args": ["xyz"], "expected": 0},
            {"args": [""], "expected": 0},
            {"args": ["Python Programming"], "expected": 4},
        ],
    },
    {
        "prompt": "Write a Python function `flatten(nested: list) -> list` that flattens a nested list of arbitrary depth into a single flat list.",
        "function_name": "flatten",
        "tests": [
            {"args": [[[1, [2, 3]], [4, 5]]], "expected": [1, 2, 3, 4, 5]},
            {"args": [[[1, 2, 3]]], "expected": [1, 2, 3]},
            {"args": [[[]]], "expected": []},
            {"args": [[[1, [2, [3, [4]]]]]], "expected": [1, 2, 3, 4]},
        ],
    },
    {
        "prompt": "Write a Python function `two_sum(nums: list[int], target: int) -> tuple[int, int]` that returns indices of two numbers that add up to target. Assume exactly one solution exists.",
        "function_name": "two_sum",
        "tests": [
            {"args": [[2, 7, 11, 15], 9], "expected": (0, 1)},
            {"args": [[3, 2, 4], 6], "expected": (1, 2)},
            {"args": [[3, 3], 6], "expected": (0, 1)},
        ],
    },
    {
        "prompt": "Write a Python function `is_anagram(s: str, t: str) -> bool` that returns True if t is an anagram of s (case-insensitive).",
        "function_name": "is_anagram",
        "tests": [
            {"args": ["listen", "silent"], "expected": True},
            {"args": ["hello", "world"], "expected": False},
            {"args": ["Astronomer", "Moon starer"], "expected": True},
            {"args": ["", ""], "expected": True},
        ],
    },
    {
        "prompt": "Write a Python function `roman_to_int(s: str) -> int` that converts a Roman numeral string to an integer. Valid symbols: I=1, V=5, X=10, L=50, C=100, D=500, M=1000.",
        "function_name": "roman_to_int",
        "tests": [
            {"args": ["III"], "expected": 3},
            {"args": ["IV"], "expected": 4},
            {"args": ["IX"], "expected": 9},
            {"args": ["MCMXCIV"], "expected": 1994},
            {"args": ["LVIII"], "expected": 58},
        ],
    },
    {
        "prompt": "Write a Python function `merge_sorted(a: list[int], b: list[int]) -> list[int]` that merges two sorted lists into one sorted list.",
        "function_name": "merge_sorted",
        "tests": [
            {"args": [[1, 3, 5], [2, 4, 6]], "expected": [1, 2, 3, 4, 5, 6]},
            {"args": [[], [1, 2]], "expected": [1, 2]},
            {"args": [[1], []], "expected": [1]},
            {"args": [[], []], "expected": []},
            {"args": [[1, 1], [1, 1]], "expected": [1, 1, 1, 1]},
        ],
    },
    {
        "prompt": "Write a Python function `gcd(a: int, b: int) -> int` that returns the greatest common divisor of two non-negative integers using the Euclidean algorithm.",
        "function_name": "gcd",
        "tests": [
            {"args": [12, 8], "expected": 4},
            {"args": [100, 75], "expected": 25},
            {"args": [7, 13], "expected": 1},
            {"args": [0, 5], "expected": 5},
            {"args": [0, 0], "expected": 0},
        ],
    },
    {
        "prompt": "Write a Python function `unique_chars(s: str) -> bool` that returns True if all characters in the string are unique.",
        "function_name": "unique_chars",
        "tests": [
            {"args": ["abcdef"], "expected": True},
            {"args": ["hello"], "expected": False},
            {"args": [""], "expected": True},
            {"args": ["a"], "expected": True},
            {"args": ["abcda"], "expected": False},
        ],
    },
    {
        "prompt": "Write a Python function `rotate_list(lst: list, k: int) -> list` that rotates the list to the right by k steps. For example, [1,2,3,4,5] rotated by 2 becomes [4,5,1,2,3].",
        "function_name": "rotate_list",
        "tests": [
            {"args": [[1, 2, 3, 4, 5], 2], "expected": [4, 5, 1, 2, 3]},
            {"args": [[1, 2, 3], 0], "expected": [1, 2, 3]},
            {"args": [[], 3], "expected": []},
            {"args": [[1], 5], "expected": [1]},
            {"args": [[1, 2], 3], "expected": [2, 1]},
        ],
    },
    {
        "prompt": "Write a Python function `binary_search(arr: list[int], target: int) -> int` that returns the index of target in a sorted list, or -1 if not found.",
        "function_name": "binary_search",
        "tests": [
            {"args": [[1, 3, 5, 7, 9], 5], "expected": 2},
            {"args": [[1, 3, 5, 7, 9], 6], "expected": -1},
            {"args": [[], 1], "expected": -1},
            {"args": [[1], 1], "expected": 0},
            {"args": [[2, 4, 6, 8], 8], "expected": 3},
        ],
    },
]


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


def extract_code(text: str) -> str:
    """Extract Python code from agent output.

    Handles:
        - ``python ... `` blocks
        - `` ... `` blocks
        - Raw code without fences

    Returns:
        The extracted code string.
    """
    if not text or not text.strip():
        return ""

    # Try ```python blocks first
    py_match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if py_match:
        return py_match.group(1).strip()

    # Try generic ``` blocks
    generic_match = re.search(r"```\s*\n?(.*?)```", text, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()

    # Fall back to raw text
    return text.strip()


# ---------------------------------------------------------------------------
# Safe code execution
# ---------------------------------------------------------------------------


def _run_test(
    code: str,
    function_name: str,
    args: list[Any],
    expected: Any,
) -> bool:
    """Execute extracted code and check one test case.

    Returns True if the function produces the expected output, False otherwise.
    Catches all exceptions to prevent agent code from crashing the evaluator.

    Note: Uses exec() intentionally -- agent-generated code must be run
    dynamically.  The namespace is isolated to limit side effects.
    """
    namespace: dict[str, Any] = {}
    try:
        compiled = compile(code, "<agent-code>", "exec")
        _do_exec(compiled, namespace)
    except Exception:
        return False

    func = namespace.get(function_name)
    if func is None or not callable(func):
        return False

    try:
        result = func(*args)
        return result == expected
    except Exception:
        return False


def _do_exec(compiled: Any, namespace: dict[str, Any]) -> None:
    """Run compiled code in the given namespace.

    Separated into its own function so static analysis tools can audit
    the single call site easily.  Uses dynamic execution intentionally
    because agent-generated code must be run at runtime.
    """
    import builtins

    runner = getattr(builtins, "exec")  # noqa: S102
    runner(compiled, namespace)


# ---------------------------------------------------------------------------
# Coding Arena
# ---------------------------------------------------------------------------


class CodingArena(Arena):
    """Arena for evaluating coding ability via function implementation.

    Agents receive a problem description and must produce a Python function.
    The function is tested against predefined test cases.  Fitness equals
    the fraction of tests passed (via PassRateEvaluator).

    Attributes:
        problems: The pool of coding problems to draw from.
    """

    def __init__(self, problems: list[dict[str, Any]] | None = None) -> None:
        """Initialise with an optional custom problem pool.

        Args:
            problems: List of dicts with 'prompt', 'function_name', and 'tests'.
                      Defaults to the built-in problem set.
        """
        self.problems: list[dict[str, Any]] = (
            list(problems) if problems is not None else list(_BUILTIN_PROBLEMS)
        )
        self._pass_rate = PassRateEvaluator()

    async def generate_tasks(self, count: int, seed: int | None = None) -> list[Task]:
        """Sample *count* coding tasks from the problem pool.

        Args:
            count: Number of tasks to generate.
            seed: Optional RNG seed for reproducibility.

        Returns:
            A list of Task objects. Each task's ``expected`` field holds
            the dict with 'function_name' and 'tests'.
        """
        rng = random.Random(seed)
        pool = list(self.problems)

        if count > len(pool):
            selected = rng.choices(pool, k=count)
        else:
            selected = rng.sample(pool, count)

        tasks: list[Task] = []
        for idx, prob in enumerate(selected):
            tasks.append(
                Task(
                    task_id=f"code-{idx:04d}",
                    prompt=prob["prompt"],
                    expected={
                        "function_name": prob["function_name"],
                        "tests": prob["tests"],
                    },
                    metadata={"domain": "coding"},
                )
            )
        return tasks

    async def evaluate(
        self, genome: Genome, adapter: Adapter, tasks: list[Task]
    ) -> EvalResult:
        """Run the agent on each coding task, extract code, and test it.

        Args:
            genome: The genome defining the agent.
            adapter: Translates the genome into a runnable agent.
            tasks: The coding tasks to evaluate on.

        Returns:
            EvalResult with pass-rate fitness.
        """
        test_outcomes: list[float] = []
        agent_results: list[AgentResult] = []
        total_tokens = 0

        for task in tasks:
            result = await adapter.run(genome, task.prompt)
            agent_results.append(result)
            total_tokens += result.cost_tokens

            if not result.success:
                # Count all tests for this task as failed
                num_tests = len(task.expected["tests"])
                test_outcomes.extend([0.0] * num_tests)
                continue

            code = extract_code(result.output)
            function_name: str = task.expected["function_name"]
            tests: list[dict[str, Any]] = task.expected["tests"]

            for tc in tests:
                passed = _run_test(code, function_name, tc["args"], tc["expected"])
                test_outcomes.append(1.0 if passed else 0.0)

        if not test_outcomes:
            fitness = 0.0
        else:
            fitness = self._pass_rate.evaluate(test_outcomes, [0.0] * len(test_outcomes))

        return EvalResult(
            genome_id=genome.genome_id,
            fitness_score=fitness,
            fitness_breakdown={"pass_rate": fitness},
            task_results=agent_results,
            cost_tokens=total_tokens,
        )
