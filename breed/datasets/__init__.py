"""Dataset loaders for benchmark forecasting questions.

Provides typed access to public forecasting datasets with explicit
provenance, licenses, and time-based train/test splits suitable for
research evaluation.
"""

from breed.datasets.base import (
    DatasetMetadata,
    DatasetSplit,
    ForecastingQuestion,
    QuestionDataset,
)
from breed.datasets.builtin import load_builtin_questions
from breed.datasets.loader import load_jsonl, save_jsonl, time_based_split

__all__ = [
    "DatasetMetadata",
    "DatasetSplit",
    "ForecastingQuestion",
    "QuestionDataset",
    "load_builtin_questions",
    "load_jsonl",
    "save_jsonl",
    "time_based_split",
]
