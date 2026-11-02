"""Normalization helpers used across retrieval and verification."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, List, Optional, Set


STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "with",
}


# CJK Unified Ideographs main block (covers common Chinese); extend later if needed.
_CJK_PATTERN = "一-鿿"


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace; keep latin and CJK."""

    text = text.lower()
    text = re.sub(rf"[^a-z0-9{_CJK_PATTERN}\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _cjk_bigrams(run: str) -> List[str]:
    """Character bigrams for a run of CJK characters (unigram if length 1)."""

    if len(run) == 1:
        return [run]
    return [run[index : index + 2] for index in range(len(run) - 1)]


def tokenize_text(text: str) -> List[str]:
    """Tokenize text: latin words (minus stopwords) + CJK character bigrams."""

    tokens: List[str] = []
    for chunk in normalize_text(text).split():
        for segment in re.findall(rf"[{_CJK_PATTERN}]+|[a-z0-9]+", chunk):
            if re.match(rf"[{_CJK_PATTERN}]", segment):
                tokens.extend(_cjk_bigrams(segment))
            elif segment not in STOPWORDS:
                tokens.append(segment)
    return tokens


def sequence_similarity(left: str, right: str) -> float:
    """Character-level string similarity."""

    if not left or not right:
        return 0.0
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def year_matches(candidate_year: Optional[int], canonical_year: Optional[int]) -> bool:
    """Year equality with a narrow tolerance for preprint versus venue metadata."""

    if candidate_year is None or canonical_year is None:
        return False
    return abs(candidate_year - canonical_year) <= 1


_SURNAME_PARTICLES = {"van", "von", "de", "del", "della", "di", "da", "le", "la", "st", "bin", "al"}
_ET_AL_RE = re.compile(r",?\s+et\s+al\.?$", re.IGNORECASE)
_INITIALS_RE = re.compile(r"[A-Za-z]+")


def _usable_author_names(names: Iterable[str]) -> List[str]:
    usable = []
    for name in names:
        cleaned = _ET_AL_RE.sub("", str(name or "")).strip(" ,;")
        if cleaned and cleaned.lower() not in {"et al", "et al."}:
            usable.append(cleaned)
    return usable


def _is_cjk_name(name: str) -> bool:
    compact = re.sub(r"\s+", "", name or "")
    return bool(compact) and bool(re.fullmatch(rf"[{_CJK_PATTERN}·]+", compact))


def _name_script(names: Iterable[str]) -> str:
    cjk = latin = 0
    for name in names:
        if re.search(rf"[{_CJK_PATTERN}]", name or ""):
            cjk += 1
        if re.search(r"[A-Za-z]", name or ""):
            latin += 1
    if cjk and not latin:
        return "cjk"
    if latin and not cjk:
        return "latin"
    return "mixed"


def author_lists_comparable(candidate_authors: Iterable[str], canonical_authors: Iterable[str]) -> bool:
    """False when one side is Chinese and the other is Latin, so a mismatch is not evidence."""

    candidate = _usable_author_names(candidate_authors)
    canonical = _usable_author_names(canonical_authors)
    if not candidate or not canonical:
        return True
    return {_name_script(candidate), _name_script(canonical)} != {"cjk", "latin"}


def _surname_and_given(name: str) -> tuple:
    cleaned = _ET_AL_RE.sub("", re.sub(r"\s+", " ", name or "")).strip(" .;,")
    if not cleaned:
        return "", ""
    if _is_cjk_name(cleaned):
        compact = re.sub(r"[\s·]+", "", cleaned)
        return compact[0], compact[1:]
    if "," in cleaned:
        family, _, given = cleaned.partition(",")
        return family.strip(), given.strip()
    parts = cleaned.split(" ")
    if len(parts) >= 2 and parts[-2].lower().rstrip(".") in _SURNAME_PARTICLES:
        return " ".join(parts[-2:]), " ".join(parts[:-2])
    if len(parts) >= 2:
        return parts[-1], " ".join(parts[:-1])
    return parts[0], ""


def _initials(given: str) -> str:
    return "".join(token[0].lower() for token in _INITIALS_RE.findall(given or ""))


def names_match(left: str, right: str) -> bool:
    """Match surnames plus compatible initials, including ``Holman, L.`` vs ``Lynette Holman``."""

    if _is_cjk_name(left) and _is_cjk_name(right):
        ln = re.sub(r"[\s·]+", "", left)
        rn = re.sub(r"[\s·]+", "", right)
        if ln == rn:
            return True
        if len(ln) == 1 and rn.startswith(ln):
            return True
        return len(rn) == 1 and ln.startswith(rn)
    left_surname, left_given = _surname_and_given(left)
    right_surname, right_given = _surname_and_given(right)
    if not left_surname or not right_surname:
        return False
    if normalize_text(left_surname) != normalize_text(right_surname):
        return False
    left_initials = _initials(left_given)
    right_initials = _initials(right_given)
    if not left_initials or not right_initials:
        return True
    shorter, longer = (
        (left_initials, right_initials)
        if len(left_initials) <= len(right_initials)
        else (right_initials, left_initials)
    )
    return longer.startswith(shorter)


def author_coverage(candidate_authors: Iterable[str], canonical_authors: Iterable[str]) -> float:
    """Estimate how much of the candidate author list matches the canonical list.

    Comparison is by surname and initials, so ``Lewis, S. C.`` matches
    ``Seth C. Lewis`` and does not require the full display name.
    """

    candidate = _usable_author_names(candidate_authors)
    canonical = _usable_author_names(canonical_authors)
    if not candidate or not canonical:
        return 0.0
    if not author_lists_comparable(candidate, canonical):
        return 0.0
    matched = sum(1 for name in candidate if any(names_match(name, other) for other in canonical))
    return matched / len(candidate)


def first_authors_match(candidate_authors: Iterable[str], canonical_authors: Iterable[str]) -> bool:
    """True when the cited first author is the canonical first author, or either side is absent."""

    candidate = _usable_author_names(candidate_authors)
    canonical = _usable_author_names(canonical_authors)
    if not candidate or not canonical:
        return True
    if not author_lists_comparable(candidate, canonical):
        return True
    return names_match(candidate[0], canonical[0])
