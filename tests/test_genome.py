"""Tests for breed.genome -- Gene, Genome, serialization, and distance."""

from __future__ import annotations

import pytest

from breed.genome import (
    DEFAULT_GENE_TEMPLATES,
    Gene,
    GeneType,
    Genome,
    create_random_genome,
)


# ── Gene creation ────────────────────────────────────────────────────────

class TestGeneCreation:
    """Verify that each GeneType can be instantiated with valid data."""

    def test_text_gene(self) -> None:
        gene = Gene(name="my_text", type=GeneType.TEXT, value="hello world")
        assert gene.type == GeneType.TEXT
        assert gene.value == "hello world"

    def test_enum_gene(self) -> None:
        gene = Gene(
            name="my_enum",
            type=GeneType.ENUM,
            value="a",
            options=["a", "b", "c"],
        )
        assert gene.type == GeneType.ENUM
        assert gene.value == "a"

    def test_float_gene(self) -> None:
        gene = Gene(
            name="my_float",
            type=GeneType.FLOAT,
            value=0.5,
            range=(0.0, 1.0),
        )
        assert gene.type == GeneType.FLOAT
        assert gene.value == 0.5

    def test_set_gene(self) -> None:
        gene = Gene(
            name="my_set",
            type=GeneType.SET,
            value=["a", "c"],
            available=["a", "b", "c", "d"],
        )
        assert gene.type == GeneType.SET
        assert set(gene.value) == {"a", "c"}

    def test_numeric_vector_gene(self) -> None:
        gene = Gene(
            name="my_vec",
            type=GeneType.NUMERIC_VECTOR,
            value=[0.4, 0.35, 0.25],
        )
        assert gene.type == GeneType.NUMERIC_VECTOR
        assert len(gene.value) == 3
        assert abs(sum(gene.value) - 1.0) < 1e-9


# ── Gene validation ─────────────────────────────────────────────────────

class TestGeneValidation:
    """Verify that invalid gene values are rejected."""

    def test_enum_rejects_invalid_value(self) -> None:
        with pytest.raises(ValueError, match="not in options"):
            Gene(
                name="bad_enum",
                type=GeneType.ENUM,
                value="z",
                options=["a", "b", "c"],
            )

    def test_enum_requires_options(self) -> None:
        with pytest.raises(ValueError, match="requires 'options'"):
            Gene(name="no_opts", type=GeneType.ENUM, value="a")

    def test_float_rejects_out_of_range_low(self) -> None:
        with pytest.raises(ValueError, match="outside range"):
            Gene(
                name="bad_float",
                type=GeneType.FLOAT,
                value=-1.0,
                range=(0.0, 1.0),
            )

    def test_float_rejects_out_of_range_high(self) -> None:
        with pytest.raises(ValueError, match="outside range"):
            Gene(
                name="bad_float",
                type=GeneType.FLOAT,
                value=2.0,
                range=(0.0, 1.0),
            )

    def test_float_requires_range(self) -> None:
        with pytest.raises(ValueError, match="requires 'range'"):
            Gene(name="no_range", type=GeneType.FLOAT, value=0.5)

    def test_set_rejects_invalid_items(self) -> None:
        with pytest.raises(ValueError, match="not in available"):
            Gene(
                name="bad_set",
                type=GeneType.SET,
                value=["a", "x"],
                available=["a", "b", "c"],
            )

    def test_set_requires_available(self) -> None:
        with pytest.raises(ValueError, match="requires 'available'"):
            Gene(name="no_avail", type=GeneType.SET, value=["a"])

    def test_numeric_vector_rejects_non_numeric(self) -> None:
        with pytest.raises(ValueError, match="list of numbers"):
            Gene(name="bad_vec", type=GeneType.NUMERIC_VECTOR, value=["x", "y"])


# ── Genome creation ─────────────────────────────────────────────────────

class TestGenomeCreation:
    """Verify Genome construction and defaults."""

    def test_create_random_genome_has_all_genes(self) -> None:
        genome = create_random_genome(seed=42)
        assert set(genome.genes.keys()) == set(DEFAULT_GENE_TEMPLATES.keys())

    def test_create_random_genome_generation_zero(self) -> None:
        genome = create_random_genome(seed=42)
        assert genome.generation == 0

    def test_create_random_genome_no_parents(self) -> None:
        genome = create_random_genome(seed=42)
        assert genome.parents == []

    def test_create_random_genome_has_id(self) -> None:
        genome = create_random_genome(seed=42)
        assert len(genome.genome_id) == 32  # UUID4 hex

    def test_create_random_genome_deterministic(self) -> None:
        g1 = create_random_genome(seed=123)
        g2 = create_random_genome(seed=123)
        assert g1.genes["temperature"].value == g2.genes["temperature"].value
        assert g1.genes["prompt_template"].value == g2.genes["prompt_template"].value

    def test_create_random_genome_different_seeds_differ(self) -> None:
        g1 = create_random_genome(seed=1)
        g2 = create_random_genome(seed=999)
        # At least one gene should differ
        any_different = any(
            g1.genes[name].value != g2.genes[name].value
            for name in g1.genes
        )
        assert any_different


# ── Serialization roundtrip ─────────────────────────────────────────────

class TestSerialization:
    """Verify dict and YAML serialization roundtrips."""

    def test_dict_roundtrip(self) -> None:
        genome = create_random_genome(seed=42)
        data = genome.to_dict()
        restored = Genome.from_dict(data)
        assert restored.genome_id == genome.genome_id
        assert restored.generation == genome.generation
        assert set(restored.genes.keys()) == set(genome.genes.keys())
        for name in genome.genes:
            assert restored.genes[name].value == genome.genes[name].value

    def test_yaml_roundtrip(self) -> None:
        genome = create_random_genome(seed=42)
        yaml_str = genome.to_yaml()
        assert isinstance(yaml_str, str)
        assert len(yaml_str) > 0
        restored = Genome.from_yaml(yaml_str)
        assert restored.genome_id == genome.genome_id
        assert set(restored.genes.keys()) == set(genome.genes.keys())
        for name in genome.genes:
            assert restored.genes[name].value == genome.genes[name].value

    def test_dict_contains_expected_top_level_keys(self) -> None:
        genome = create_random_genome(seed=42)
        data = genome.to_dict()
        expected_keys = {
            "genome_id", "generation", "parents", "created_at",
            "genes", "fitness_history", "metadata",
        }
        assert expected_keys == set(data.keys())


# ── Distance metric ─────────────────────────────────────────────────────

class TestDistance:
    """Verify the normalised distance metric."""

    def test_distance_to_self_is_zero(self) -> None:
        genome = create_random_genome(seed=42)
        assert genome.distance(genome) == pytest.approx(0.0)

    def test_distance_is_symmetric(self) -> None:
        g1 = create_random_genome(seed=1)
        g2 = create_random_genome(seed=2)
        assert g1.distance(g2) == pytest.approx(g2.distance(g1))

    def test_distance_in_valid_range(self) -> None:
        g1 = create_random_genome(seed=10)
        g2 = create_random_genome(seed=20)
        d = g1.distance(g2)
        assert 0.0 <= d <= 1.0

    def test_identical_genomes_have_zero_distance(self) -> None:
        g1 = create_random_genome(seed=42)
        g2 = Genome.from_dict(g1.to_dict())
        assert g1.distance(g2) == pytest.approx(0.0)

    def test_different_genomes_have_nonzero_distance(self) -> None:
        g1 = create_random_genome(seed=1)
        g2 = create_random_genome(seed=999)
        assert g1.distance(g2) > 0.0

    def test_distance_no_shared_genes(self) -> None:
        g1 = Genome(genes={
            "a": Gene(name="a", type=GeneType.TEXT, value="hello"),
        })
        g2 = Genome(genes={
            "b": Gene(name="b", type=GeneType.TEXT, value="world"),
        })
        assert g1.distance(g2) == 1.0
