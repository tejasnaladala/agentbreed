"""JSONL loaders, savers, and time-based split utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from breed.datasets.base import (
    DatasetMetadata,
    ForecastingQuestion,
    QuestionDataset,
)


def load_jsonl(path: str | Path, metadata: DatasetMetadata) -> QuestionDataset:
    """Load a QuestionDataset from a JSONL file.

    Parameters
    ----------
    path:
        Path to a .jsonl file with one ForecastingQuestion dict per line.
    metadata:
        Dataset metadata to attach.
    """
    path = Path(path)
    questions: list[ForecastingQuestion] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(ForecastingQuestion.from_dict(json.loads(line)))
    return QuestionDataset(metadata=metadata, questions=tuple(questions))


def save_jsonl(dataset: QuestionDataset, path: str | Path) -> None:
    """Write a QuestionDataset to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in dataset.questions:
            f.write(json.dumps(q.to_dict()) + "\n")


def time_based_split(
    dataset: QuestionDataset,
    train_cutoff: str,
    test_cutoff: str | None = None,
) -> tuple[QuestionDataset, QuestionDataset]:
    """Split a dataset by resolution date.

    Train set: questions resolved strictly before ``train_cutoff``.
    Test set: questions resolved on or after ``train_cutoff``, and
              strictly before ``test_cutoff`` if provided.

    This is the only valid split for forecasting -- random splits
    leak future information into training.

    Parameters
    ----------
    dataset:
        The full dataset to split.
    train_cutoff:
        ISO-8601 date string. Questions resolved before this go to train.
    test_cutoff:
        Optional upper bound for test set. If None, test = everything
        from train_cutoff onward.

    Returns
    -------
    (train, test):
        Two new QuestionDataset instances sharing the same metadata.
    """
    train = dataset.filter_resolved_before(train_cutoff)
    if test_cutoff is None:
        test = dataset.filter_resolved_after(train_cutoff)
    else:
        cutoff_dt = datetime.fromisoformat(test_cutoff)
        test = tuple(
            q for q in dataset.questions
            if q.resolved_datetime is not None
            and datetime.fromisoformat(train_cutoff) <= q.resolved_datetime < cutoff_dt
        )
        test = QuestionDataset(metadata=dataset.metadata, questions=test)
    return train, test


def stratified_subsample(
    dataset: QuestionDataset,
    n_per_category: int,
    seed: int | None = None,
) -> QuestionDataset:
    """Subsample n_per_category questions from each category.

    Useful for cost-controlled experiments where evaluating the full
    dataset is too expensive.
    """
    import random as _random

    rng = _random.Random(seed)
    by_category: dict[str, list[ForecastingQuestion]] = {}
    for q in dataset.questions:
        by_category.setdefault(q.category, []).append(q)

    sampled: list[ForecastingQuestion] = []
    for cat, qs in sorted(by_category.items()):
        n = min(n_per_category, len(qs))
        sampled.extend(rng.sample(qs, n))

    return QuestionDataset(metadata=dataset.metadata, questions=tuple(sampled))
