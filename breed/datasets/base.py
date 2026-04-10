"""Base dataset types for forecasting questions.

Frozen dataclasses with full provenance tracking. Every question carries
its source identifier, resolution date, license, and category for
reproducible research evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterator


class DatasetSplit(str, Enum):
    """Standard splits for time-based forecasting evaluation."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    EXTERNAL = "external"


@dataclass(frozen=True)
class DatasetMetadata:
    """Provenance and licensing information for a dataset."""

    name: str
    source: str
    url: str
    license: str
    version: str
    download_date: str
    n_questions: int
    description: str
    citations: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "url": self.url,
            "license": self.license,
            "version": self.version,
            "download_date": self.download_date,
            "n_questions": self.n_questions,
            "description": self.description,
            "citations": list(self.citations),
        }


@dataclass(frozen=True)
class ForecastingQuestion:
    """A single resolved binary forecasting question.

    Attributes
    ----------
    question_id:
        Stable identifier (source-prefixed, e.g. 'metaculus-12345').
    prompt:
        The forecasting question text.
    outcome:
        Resolved outcome in {0.0, 1.0}.
    category:
        Topic category (e.g. 'tech', 'politics', 'science').
    asked_date:
        ISO-8601 date the question was first asked.
    resolved_date:
        ISO-8601 date the question resolved.
    source:
        Source identifier (e.g. 'metaculus', 'forecastbench', 'builtin').
    metadata:
        Source-specific extra fields (community prediction, etc).
    """

    question_id: str
    prompt: str
    outcome: float
    category: str
    asked_date: str
    resolved_date: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "prompt": self.prompt,
            "outcome": self.outcome,
            "category": self.category,
            "asked_date": self.asked_date,
            "resolved_date": self.resolved_date,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ForecastingQuestion":
        return cls(
            question_id=data["question_id"],
            prompt=data["prompt"],
            outcome=float(data["outcome"]),
            category=data.get("category", "unknown"),
            asked_date=data.get("asked_date", ""),
            resolved_date=data.get("resolved_date", ""),
            source=data.get("source", "unknown"),
            metadata=dict(data.get("metadata", {})),
        )

    @property
    def resolved_datetime(self) -> datetime | None:
        """Parse the resolved_date as a datetime, or None if invalid."""
        if not self.resolved_date:
            return None
        try:
            return datetime.fromisoformat(self.resolved_date.replace("Z", "+00:00"))
        except ValueError:
            return None


@dataclass(frozen=True)
class QuestionDataset:
    """A collection of forecasting questions with metadata.

    Iterable, indexable, and sliceable. Designed to be passed directly
    into the experiment runner.
    """

    metadata: DatasetMetadata
    questions: tuple[ForecastingQuestion, ...]

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self) -> Iterator[ForecastingQuestion]:
        return iter(self.questions)

    def __getitem__(self, idx: int | slice) -> ForecastingQuestion | tuple[ForecastingQuestion, ...]:
        return self.questions[idx]

    def filter_by_category(self, *categories: str) -> "QuestionDataset":
        """Return a new dataset containing only questions in the given categories."""
        cats = set(categories)
        filtered = tuple(q for q in self.questions if q.category in cats)
        return QuestionDataset(metadata=self.metadata, questions=filtered)

    def filter_resolved_before(self, iso_date: str) -> "QuestionDataset":
        """Return a new dataset of questions resolved strictly before iso_date."""
        cutoff = datetime.fromisoformat(iso_date)
        filtered = tuple(
            q for q in self.questions
            if q.resolved_datetime is not None and q.resolved_datetime < cutoff
        )
        return QuestionDataset(metadata=self.metadata, questions=filtered)

    def filter_resolved_after(self, iso_date: str) -> "QuestionDataset":
        """Return a new dataset of questions resolved on or after iso_date."""
        cutoff = datetime.fromisoformat(iso_date)
        filtered = tuple(
            q for q in self.questions
            if q.resolved_datetime is not None and q.resolved_datetime >= cutoff
        )
        return QuestionDataset(metadata=self.metadata, questions=filtered)

    def categories(self) -> tuple[str, ...]:
        """Return all unique categories present in the dataset."""
        return tuple(sorted({q.category for q in self.questions}))

    def category_counts(self) -> dict[str, int]:
        """Return a category -> question count mapping."""
        counts: dict[str, int] = {}
        for q in self.questions:
            counts[q.category] = counts.get(q.category, 0) + 1
        return counts

    def base_rate(self) -> float:
        """Mean outcome (fraction of questions resolved YES)."""
        if not self.questions:
            return 0.0
        return sum(q.outcome for q in self.questions) / len(self.questions)
