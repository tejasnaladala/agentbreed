"""Genome validator and content-hash.

Every genome produced by any search method is passed through validate_genome
before an LLM call is issued. If validation fails, the genome is replaced with
the default and the failure is logged — the run does not crash.

The content hash is content-addressable and stable under tuple ordering,
so two genomes with the same values produce the same hash regardless of how
they were constructed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .schema import (
    BENCHMARKS,
    GENES,
    GENE_NAMES,
    GeneScope,
    GeneType,
    Genome,
    legal_values_for,
)


class GenomeValidationError(ValueError):
    """Raised when a genome has a value not in its legal set."""


def validate_genome(genome: Genome, *, strict: bool = True) -> None:
    """Validate a Genome against the frozen schema.

    Raises GenomeValidationError if strict=True and any gene value is invalid.
    If strict=False, returns None and logs to stderr (so search methods can
    recover gracefully).
    """
    if genome.benchmark not in BENCHMARKS:
        raise GenomeValidationError(
            f"unknown benchmark {genome.benchmark!r}; legal: {BENCHMARKS}"
        )

    names_in_genome = tuple(n for n, _ in genome.values)
    if names_in_genome != GENE_NAMES:
        raise GenomeValidationError(
            f"genome gene order/names do not match schema. "
            f"Expected {GENE_NAMES}, got {names_in_genome}"
        )

    for gene in GENES:
        value = genome.get(gene.name)
        legal = legal_values_for(gene.name, genome.benchmark)

        if gene.type == GeneType.ENUM:
            if value not in legal:
                if strict:
                    raise GenomeValidationError(
                        f"gene {gene.name!r}: value {value!r} not in legal set "
                        f"{legal!r} for benchmark {genome.benchmark!r}"
                    )
        elif gene.type == GeneType.SET:
            if not isinstance(value, frozenset):
                if strict:
                    raise GenomeValidationError(
                        f"gene {gene.name!r}: SET gene must be frozenset, got {type(value).__name__}"
                    )
                continue
            illegal = value - set(legal)
            if illegal:
                if strict:
                    raise GenomeValidationError(
                        f"gene {gene.name!r}: set contains illegal elements {illegal}"
                    )
        elif gene.type == GeneType.FLOAT:
            low, high = legal
            if not isinstance(value, (int, float)):
                if strict:
                    raise GenomeValidationError(
                        f"gene {gene.name!r}: FLOAT gene must be numeric, got {type(value).__name__}"
                    )
                continue
            if not (low <= float(value) <= high):
                if strict:
                    raise GenomeValidationError(
                        f"gene {gene.name!r}: value {value} out of range [{low}, {high}]"
                    )
        elif gene.type == GeneType.BOOL:
            if not isinstance(value, bool):
                if strict:
                    raise GenomeValidationError(
                        f"gene {gene.name!r}: BOOL gene must be bool, got {type(value).__name__}"
                    )


def _canonicalize(value: Any) -> Any:
    """Canonicalize a value for stable hashing."""
    if isinstance(value, frozenset):
        return ["__frozenset__", *sorted(_canonicalize(x) for x in value)]
    if isinstance(value, (set, list, tuple)):
        return [_canonicalize(x) for x in value]
    if isinstance(value, dict):
        return {k: _canonicalize(v) for k, v in sorted(value.items())}
    if isinstance(value, float):
        # Round to avoid nonassociativity drift.
        return round(value, 6)
    return value


def genome_content_hash(genome: Genome) -> str:
    """Stable SHA-256 of the (benchmark, sorted_gene_values) tuple.

    Two genomes with the same benchmark and same gene values produce the same hash,
    independent of how they were constructed or what random seed made them.
    """
    payload = {
        "benchmark": genome.benchmark,
        "genes": [(name, _canonicalize(value)) for name, value in genome.values],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
