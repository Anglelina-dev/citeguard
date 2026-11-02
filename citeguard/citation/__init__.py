"""Citation utilities."""

from .formatter import CitationFormatter
from .normalizer import (
    author_coverage,
    author_lists_comparable,
    first_authors_match,
    names_match,
    normalize_text,
    sequence_similarity,
    tokenize_text,
    year_matches,
)
from .proposer import CandidateCitation, CitationProposer

__all__ = [
    "CandidateCitation",
    "CitationFormatter",
    "CitationProposer",
    "author_coverage",
    "author_lists_comparable",
    "first_authors_match",
    "names_match",
    "normalize_text",
    "sequence_similarity",
    "tokenize_text",
    "year_matches",
]
