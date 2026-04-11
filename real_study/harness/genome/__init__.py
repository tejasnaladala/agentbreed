"""Agent genome schema and validator.

Implements the 9-gene schema from docs/04_genome_schema.md. Every genome value
is type-checked against its legal set before any LLM call.
"""

from .schema import (
    BENCHMARKS,
    GENES,
    GENE_NAMES,
    CANONICAL_SWEEP_ORDER,
    GeneSpec,
    Genome,
    GeneCategory,
    GeneType,
    GeneScope,
    legal_values_for,
    default_genome_for,
    random_genome_for,
)
from .validator import validate_genome, genome_content_hash

__all__ = [
    "BENCHMARKS",
    "GENES",
    "GENE_NAMES",
    "CANONICAL_SWEEP_ORDER",
    "GeneSpec",
    "Genome",
    "GeneCategory",
    "GeneType",
    "GeneScope",
    "legal_values_for",
    "default_genome_for",
    "random_genome_for",
    "validate_genome",
    "genome_content_hash",
]
