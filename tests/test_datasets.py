"""Tests for the breed.datasets module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from breed.datasets import (
    DatasetMetadata,
    DatasetSplit,
    ForecastingQuestion,
    QuestionDataset,
    load_builtin_questions,
    load_jsonl,
    save_jsonl,
    time_based_split,
)
from breed.datasets.loader import stratified_subsample


class TestForecastingQuestion:
    def test_creation(self) -> None:
        q = ForecastingQuestion(
            question_id="test-001",
            prompt="Will X happen?",
            outcome=1.0,
            category="tech",
            asked_date="2024-01-01",
            resolved_date="2024-06-01",
            source="test",
        )
        assert q.question_id == "test-001"
        assert q.outcome == 1.0

    def test_dict_roundtrip(self) -> None:
        q = ForecastingQuestion(
            question_id="test-001",
            prompt="Will X happen?",
            outcome=0.0,
            category="science",
            asked_date="2023-01-01",
            resolved_date="2024-12-31",
            source="test",
            metadata={"community_pred": 0.3},
        )
        restored = ForecastingQuestion.from_dict(q.to_dict())
        assert restored == q

    def test_resolved_datetime(self) -> None:
        q = ForecastingQuestion(
            question_id="test-001",
            prompt="X?",
            outcome=1.0,
            category="tech",
            asked_date="",
            resolved_date="2024-06-15",
            source="test",
        )
        dt = q.resolved_datetime
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 6
        assert dt.day == 15

    def test_resolved_datetime_invalid(self) -> None:
        q = ForecastingQuestion(
            question_id="test-001",
            prompt="X?",
            outcome=1.0,
            category="tech",
            asked_date="",
            resolved_date="not-a-date",
            source="test",
        )
        assert q.resolved_datetime is None


class TestQuestionDataset:
    def test_builtin_loads(self) -> None:
        ds = load_builtin_questions()
        # The expanded built-in set has exactly 120 questions.
        assert len(ds) == 120
        assert all(isinstance(q, ForecastingQuestion) for q in ds)
        # Metadata n_questions must stay in sync with the actual pool.
        assert ds.metadata.n_questions == 120

    def test_builtin_outcomes_binary(self) -> None:
        ds = load_builtin_questions()
        for q in ds:
            assert q.outcome in (0.0, 1.0), f"{q.question_id} has non-binary outcome {q.outcome}"

    def test_builtin_balanced(self) -> None:
        ds = load_builtin_questions()
        base_rate = ds.base_rate()
        # Expansion targets 55-65% YES with a small tolerance band.
        assert 0.5 <= base_rate <= 0.7, f"base rate {base_rate} outside target band"

    def test_builtin_categories_balanced(self) -> None:
        ds = load_builtin_questions()
        counts = ds.category_counts()
        # The expansion mandates these per-category counts.
        expected = {
            "tech": 25,
            "science": 20,
            "politics": 15,
            "economics": 25,
            "sports": 15,
            "culture": 10,
            "finance": 10,
        }
        assert counts == expected

    def test_categories(self) -> None:
        ds = load_builtin_questions()
        cats = ds.categories()
        assert len(cats) >= 4
        assert "tech" in cats
        assert "science" in cats

    def test_category_counts(self) -> None:
        ds = load_builtin_questions()
        counts = ds.category_counts()
        assert sum(counts.values()) == len(ds)

    def test_filter_by_category(self) -> None:
        ds = load_builtin_questions()
        tech_only = ds.filter_by_category("tech")
        assert all(q.category == "tech" for q in tech_only)
        assert len(tech_only) > 0
        assert len(tech_only) < len(ds)

    def test_iter_and_index(self) -> None:
        ds = load_builtin_questions()
        first = ds[0]
        assert isinstance(first, ForecastingQuestion)
        all_qs = list(ds)
        assert len(all_qs) == len(ds)


class TestTimeBasedSplit:
    def test_basic_split(self) -> None:
        ds = load_builtin_questions()
        train, test = time_based_split(ds, train_cutoff="2024-01-01")
        # All train questions resolved before 2024
        for q in train:
            dt = q.resolved_datetime
            assert dt is not None
            assert dt.year < 2024
        # All test questions resolved on or after 2024
        for q in test:
            dt = q.resolved_datetime
            assert dt is not None
            assert dt.year >= 2024

    def test_split_no_overlap(self) -> None:
        ds = load_builtin_questions()
        train, test = time_based_split(ds, train_cutoff="2024-01-01")
        train_ids = {q.question_id for q in train}
        test_ids = {q.question_id for q in test}
        assert not (train_ids & test_ids)

    def test_split_total_lossless(self) -> None:
        ds = load_builtin_questions()
        train, test = time_based_split(ds, train_cutoff="2024-01-01")
        # Some questions may not have valid dates -> excluded
        valid = [q for q in ds if q.resolved_datetime is not None]
        assert len(train) + len(test) == len(valid)


class TestJsonlRoundtrip:
    def test_save_and_load(self, tmp_path: Path) -> None:
        ds = load_builtin_questions()
        path = tmp_path / "test.jsonl"
        save_jsonl(ds, path)
        assert path.exists()

        restored = load_jsonl(path, ds.metadata)
        assert len(restored) == len(ds)
        for orig, new in zip(ds, restored):
            assert orig.question_id == new.question_id
            assert orig.outcome == new.outcome


class TestStratifiedSubsample:
    def test_per_category_count(self) -> None:
        ds = load_builtin_questions()
        sub = stratified_subsample(ds, n_per_category=2, seed=42)
        counts = sub.category_counts()
        # Each category should have at most 2 (some may have fewer if pool is small)
        for cat, n in counts.items():
            assert n <= 2

    def test_deterministic(self) -> None:
        ds = load_builtin_questions()
        sub1 = stratified_subsample(ds, n_per_category=2, seed=42)
        sub2 = stratified_subsample(ds, n_per_category=2, seed=42)
        ids1 = [q.question_id for q in sub1]
        ids2 = [q.question_id for q in sub2]
        assert ids1 == ids2


class TestDatasetMetadata:
    def test_to_dict(self) -> None:
        m = DatasetMetadata(
            name="test",
            source="test",
            url="https://example.com",
            license="MIT",
            version="0.1",
            download_date="2026-04-09",
            n_questions=10,
            description="test dataset",
        )
        d = m.to_dict()
        assert d["name"] == "test"
        assert d["n_questions"] == 10
