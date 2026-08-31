"""Coding tasks with execution disabled pending a real OS sandbox.

Task generation and code extraction remain available, but this module does not
execute model-generated Python. Evaluation fails closed until process,
filesystem, environment, network, and resource isolation are implemented.
"""

from __future__ import annotations

import random
import re
from typing import Any

from breed.adapters.base import Adapter
from breed.arenas.base import Arena, EvalResult, Task
from breed.genome import Genome


_DISABLED_MESSAGE = (
    "CodingArena is disabled: evaluating model-generated Python requires an "
    "OS-level sandbox with process, filesystem, environment, network, and "
    "resource isolation."
)


class CodingArenaDisabledError(RuntimeError):
    """Raised whenever code evaluation is attempted while containment is active."""

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
# Disabled code execution
# ---------------------------------------------------------------------------


def _run_test(
    code: str,
    function_name: str,
    args: list[Any],
    expected: Any,
) -> bool:
    """Reject a code-evaluation attempt without inspecting or executing it."""
    raise CodingArenaDisabledError(_DISABLED_MESSAGE)


# ---------------------------------------------------------------------------
# Coding Arena
# ---------------------------------------------------------------------------


class CodingArena(Arena):
    """Coding task provider whose evaluation path is currently disabled.

    Problems can still be sampled for inspection, but ``evaluate`` fails before
    calling an adapter or handling generated code. It will remain disabled until
    code execution is moved into a real OS-level sandbox.

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
        """Reject evaluation before requesting or executing generated code.

        Args:
            genome: The genome defining the agent.
            adapter: Translates the genome into a runnable agent.
            tasks: The coding tasks to evaluate on.

        Raises:
            CodingArenaDisabledError: Always, while OS-level isolation is absent.
        """
        raise CodingArenaDisabledError(_DISABLED_MESSAGE)
