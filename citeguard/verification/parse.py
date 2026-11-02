"""Parse free-text or structured citation input into a candidate record."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from citeguard.graph import CitationRecord
from citeguard.retrieval.scholarly_clients.utils import (
    normalize_arxiv_id,
    normalize_doi,
    stable_record_id,
)

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)
ARXIV_NEW_ID_PATTERN = r"\d{4}\.\d{4,5}"
ARXIV_OLD_ID_PATTERN = r"[a-z][a-z-]*(?:\.[a-z]{2})?/\d{7}"
ARXIV_ID_PATTERN = rf"(?:{ARXIV_NEW_ID_PATTERN}|{ARXIV_OLD_ID_PATTERN})(?:v\d+)?"
ARXIV_LABELLED_RE = re.compile(rf"arxiv:\s*({ARXIV_ID_PATTERN})", re.IGNORECASE)
ARXIV_URL_RE = re.compile(
    rf"https?://arxiv\.org/(?:abs|pdf|html)/({ARXIV_ID_PATTERN})(?:\.pdf)?",
    re.IGNORECASE,
)
ARXIV_BARE_URL_RE = re.compile(
    rf"\barxiv\.org/(?:abs|pdf|html)/({ARXIV_ID_PATTERN})(?:\.pdf)?",
    re.IGNORECASE,
)
ARXIV_BARE_RE = re.compile(rf"\b({ARXIV_ID_PATTERN})\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

# GB/T 7714 (Chinese national standard) document-type markers such as
# [J] journal, [M] monograph, [C] proceedings, [D] thesis, [R] report,
# [S] standard, [P] patent, [N] newspaper, plus /OL-style carrier suffixes.
# Fullwidth brackets （［］） show up in Word/PDF exports of the same standard.
GBT7714_TYPE_RE = re.compile(
    r"[\[［](?P<type>EB|DB|CP|CM|DS|[JMCDRSPNAZG])(?:[/／](?:OL|CD|DK|MT))?[\]］]"
)
_GBT7714_AUTHOR_SPLIT_RE = re.compile(r"[,，;；、]+")
_GBT7714_ET_AL = {"等", "et al", "et al."}
_AUTHOR_DATE_APA_RE = re.compile(
    r"\((?:n\.d\.|(?:19|20)\d{2}[a-z]?)\)\.\s*",
    re.IGNORECASE,
)
_AUTHOR_DATE_CHICAGO_RE = re.compile(
    r"\.\s*(?:n\.d\.|(?:19|20)\d{2}[a-z]?\.)\s*",
    re.IGNORECASE,
)
_ZH_AUTHOR_YEAR_RE = re.compile(
    r"^(?P<authors>[\u4e00-\u9fff·][\u4e00-\u9fff·、，,\s]{0,80}?)"
    r"[(（](?P<year>(?:19|20)\d{2})[)）]\s*[:：.．。]\s*"
    r"(?P<title>[^，,。]+)(?:[，,]\s*[《〈](?P<venue>[^》〉]+)[》〉])?"
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def extract_doi(text: str) -> str:
    match = DOI_RE.search(text or "")
    return normalize_doi(match.group(0)) if match else ""


def extract_arxiv_id(text: str) -> str:
    match = ARXIV_LABELLED_RE.search(text or "")
    if match:
        return normalize_arxiv_id(match.group(1))
    match = ARXIV_URL_RE.search(text or "")
    if match:
        return normalize_arxiv_id(match.group(1))
    match = ARXIV_BARE_URL_RE.search(text or "")
    if match:
        return normalize_arxiv_id(match.group(1))
    match = ARXIV_BARE_RE.search(text or "")
    return normalize_arxiv_id(match.group(1)) if match else ""


def extract_year(text: str) -> Optional[int]:
    """Return the first publication-like year, ignoring years embedded in DOIs and URLs."""

    if not text:
        return None
    masked = _URL_RE.sub(" ", DOI_RE.sub(" ", text))
    match = YEAR_RE.search(masked)
    return int(match.group(0)) if match else None


def parse_gbt7714_reference(text: str) -> Optional[Dict[str, Any]]:
    """Parse one GB/T 7714 style reference (the Chinese national standard).

    Handles Chinese and English variants such as
    ``作者1, 作者2. 标题[J]. 期刊名, 年, 卷(期): 页.`` and
    ``CHEN X, LI Y. Some title[C]//Proceedings. City: Publisher, 2020.``.
    Returns None when the text does not look like a GB/T 7714 reference, so
    other free-text parsing stays untouched.
    """

    cleaned = (text or "").strip()
    match = GBT7714_TYPE_RE.search(cleaned)
    if not match:
        return None
    head = cleaned[: match.start()].strip()
    tail = cleaned[match.end() :].strip()
    type_code = match.group("type")

    authors_part = ""
    title_part = ""
    separator = ""
    for sep in (". ", "．", "。", "."):
        authors_part, separator, title_part = head.partition(sep)
        if separator:
            break
    if not separator:
        return None
    title = title_part.strip().rstrip(".．。").strip()
    authors_raw = re.sub(r"[(（](?:19|20)\d{2}[)）]", "", authors_part).strip()
    if not title or not authors_raw:
        return None

    authors = [
        author.strip(" .．。")
        for author in _GBT7714_AUTHOR_SPLIT_RE.split(authors_raw)
        if author.strip(" .．。") and author.strip(" .．。").lower() not in _GBT7714_ET_AL
    ]
    if not authors:
        return None

    venue = ""
    extra: Dict[str, Any] = {"type_code": type_code}
    tail = tail.lstrip(".．。").strip()
    if tail.startswith("//"):
        tail = tail[2:].strip()
    if type_code in {"J", "N", "C"} and tail:
        pieces = re.split(r"[,，]", tail, maxsplit=1)
        venue = pieces[0].strip().rstrip(".").strip()
        if type_code == "C":
            venue = venue.split(". ", 1)[0].strip()
        if len(pieces) > 1 and pieces[1].strip():
            extra["publication_info"] = pieces[1].strip().rstrip(".")
    elif tail:
        extra["publication_info"] = tail.rstrip(".")

    return {"title": title, "authors": authors, "venue": venue, "gbt7714": extra}


def parse_chinese_author_year_reference(text: str) -> Optional[Dict[str, Any]]:
    """Parse ``作者、作者(年):标题，《期刊》`` references used by many Chinese journals.

    This is distinct from GB/T 7714 ``[J]`` markers. Returns None when the line
    does not match, so other parsers still run.
    """

    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    match = _ZH_AUTHOR_YEAR_RE.match(cleaned)
    if not match:
        return None
    authors = [
        author.strip()
        for author in re.split(r"[、，,]", match.group("authors"))
        if author.strip() and author.strip() not in {"等"}
    ]
    title = match.group("title").strip().rstrip("。.")
    if not authors or len(title) < 2:
        return None
    return {
        "title": title,
        "authors": authors,
        "venue": (match.group("venue") or "").strip(),
        "year": int(match.group("year")),
        "reference_format": "zh_authoryear",
    }


def _split_author_date_authors(authors_raw: str) -> List[str]:
    """Split APA (``Family, I., and Family, J.``) and Chicago author lists."""

    text = re.sub(r"\s+", " ", authors_raw or "").strip(" ,;")
    text = re.sub(r",?\s+\band\b\s+", " | ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*&\s*", " | ", text)
    text = re.sub(r",?\s+et\s+al\.?$", "", text, flags=re.IGNORECASE).strip(" ,|")
    authors: List[str] = []
    for chunk in [part.strip(" ,") for part in text.split("|") if part.strip(" ,")]:
        authors.extend(_split_comma_author_chunk(chunk))
    return [author for author in authors if author and author.lower() not in {"et al", "et al."}]


def _is_initials_token(token: str) -> bool:
    return bool(re.fullmatch(r"(?:[A-Z]\.\s*)+", token.strip()))


def _is_single_name_token(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][\w'’\-]*\.?", token.strip()))


def _split_comma_author_chunk(chunk: str) -> List[str]:
    tokens = [token.strip() for token in chunk.split(",") if token.strip()]
    if len(tokens) >= 4 and _is_initials_token(tokens[1]) and _is_single_name_token(tokens[2]) and (
        _is_initials_token(tokens[3]) or _is_single_name_token(tokens[3])
    ):
        # APA repeats ``Family, I.`` for every author.
        if _is_initials_token(tokens[1]) and _is_initials_token(tokens[3]):
            paired: List[str] = []
            index = 0
            while index + 1 < len(tokens) and _is_single_name_token(tokens[index]) and _is_initials_token(tokens[index + 1]):
                paired.append(f"{tokens[index]}, {tokens[index + 1].strip()}")
                index += 2
            if index < len(tokens):
                paired.append(", ".join(tokens[index:]))
            return paired
    if len(tokens) >= 2 and (_is_initials_token(tokens[1]) or _is_single_name_token(tokens[1])):
        first = f"{tokens[0]}, {tokens[1].strip()}"
        return [first, *[token for token in tokens[2:] if token]]
    return [chunk]


def _venue_from_tail(tail: str) -> str:
    cleaned = _URL_RE.sub(" ", DOI_RE.sub(" ", tail or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;，,")
    if not cleaned:
        return ""
    volume = re.match(r"(.+?)\s+\d+\s*(?:\(|:)", cleaned)
    if volume:
        return volume.group(1).strip(" .")
    return cleaned.split(".")[0].strip(" .")[:160]


def _year_anchor_outside_identifier(text: str, pattern: re.Pattern) -> Optional[re.Match]:
    spans = [(match.start(), match.end()) for match in DOI_RE.finditer(text)]
    spans.extend((match.start(), match.end()) for match in _URL_RE.finditer(text))
    for match in pattern.finditer(text):
        if any(start <= match.start() < end for start, end in spans):
            continue
        return match
    return None


def parse_author_date_reference(text: str) -> Optional[Dict[str, Any]]:
    """Parse APA ``Author (Year). Title.`` and Chicago ``Author. Year. "Title."`` references."""

    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned or not re.search(r"[A-Za-z]", cleaned):
        return None
    apa = _year_anchor_outside_identifier(cleaned, _AUTHOR_DATE_APA_RE)
    chicago = _year_anchor_outside_identifier(cleaned, _AUTHOR_DATE_CHICAGO_RE)
    match = None
    style = ""
    if apa and (chicago is None or apa.start() <= chicago.start()):
        match, style = apa, "apa"
    elif chicago:
        match, style = chicago, "chicago"
    if match is None or match.start() > 800 or match.start() < 2:
        return None
    year_token = re.search(r"(?:19|20)\d{2}|n\.d\.", match.group(0), re.IGNORECASE)
    if year_token is None:
        return None
    authors = _split_author_date_authors(cleaned[: match.start()])
    if not authors:
        return None
    rest = cleaned[match.end() :]
    title = ""
    tail = rest
    if rest[:1] in {'"', "“", "‘", "'"}:
        opener = rest[0]
        closer = {'"': '"', "“": "”", "‘": "’", "'": "'"}.get(opener, opener)
        end = rest.find(closer, 1)
        if end > 1:
            title = rest[1:end].strip()
            tail = rest[end + 1 :]
    if not title:
        title, separator, tail = rest.partition(". ")
        if not separator:
            title, tail = rest, ""
    title = title.strip(" .“”\"'")
    if len(title) < 3:
        return None
    year = None if year_token.group(0).lower() == "n.d." else int(year_token.group(0)[:4])
    return {
        "title": title,
        "authors": authors,
        "venue": _venue_from_tail(tail),
        "year": year,
        "reference_format": style,
    }


def parse_structured_reference(text: str) -> Optional[Dict[str, Any]]:
    """Parse GB/T 7714, Chinese author-year, then APA/Chicago. None if unstructured."""

    parsed = parse_gbt7714_reference(text)
    if parsed:
        parsed = dict(parsed)
        parsed["reference_format"] = "gbt7714"
        parsed.setdefault("year", extract_year(text))
        return parsed
    parsed = parse_chinese_author_year_reference(text)
    if parsed:
        return parsed
    return parse_author_date_reference(text)


def parse_citation(
    raw_text: str = "",
    title: str = "",
    authors: Optional[List[str]] = None,
    year: Optional[int] = None,
    venue: str = "",
    abstract: str = "",
    doi: str = "",
    arxiv_id: str = "",
    evidence_chunks: Optional[List[Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> CitationRecord:
    """Build a candidate CitationRecord from whatever the caller provides.

    `title` is treated as an explicit, comparable title only when passed in.
    When only `raw_text` is given it is used as the search query (stored in
    metadata['raw_text']) and NOT treated as an explicit title.
    """

    authors = list(authors or [])
    doi = normalize_doi(doi) or extract_doi(raw_text)
    arxiv_id = normalize_arxiv_id(arxiv_id) or extract_arxiv_id(raw_text)
    if year is None:
        year = extract_year(raw_text)

    record_metadata = dict(metadata or {})
    if raw_text and not title:
        parsed_ref = parse_structured_reference(raw_text)
        if parsed_ref:
            title = parsed_ref["title"]
            if not authors:
                authors = list(parsed_ref["authors"])
            if not venue:
                venue = parsed_ref.get("venue") or ""
            if year is None and parsed_ref.get("year") is not None:
                year = parsed_ref["year"]
            record_metadata.setdefault("reference_format", parsed_ref.get("reference_format", ""))
            if parsed_ref.get("gbt7714"):
                record_metadata.setdefault("gbt7714", parsed_ref["gbt7714"])

    title_explicit = bool(title)
    search_title = title or raw_text
    seed = doi or arxiv_id or search_title or "citation"
    record_metadata.update({"raw_text": raw_text, "title_explicit": title_explicit})
    if evidence_chunks:
        record_metadata["evidence_chunks"] = list(evidence_chunks)

    return CitationRecord(
        citation_id=stable_record_id("input", seed),
        title=search_title,
        authors=authors,
        year=year,
        venue=venue,
        abstract=abstract,
        doi=doi,
        arxiv_id=arxiv_id,
        source="input",
        metadata=record_metadata,
    )
