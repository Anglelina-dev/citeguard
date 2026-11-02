"""Regression tests for bugs found while auditing real papers in this fork."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from citeguard.citation import author_coverage, first_authors_match, names_match
from citeguard.graph import CitationRecord
from citeguard.mcp.server import mcp_dependency_message
from citeguard.retrieval.scholarly_clients import InMemoryMetadataSource
from citeguard.verification.extract import extract_citation_candidates, load_citation_candidates
from citeguard.verification.models import Verdict
from citeguard.verification.parse import extract_year, parse_citation, parse_structured_reference
from citeguard.verification.verify import verify_citation


HOLMAN = (
    "Holman, L., and G. P. Perreault. 2023. “Diffusion of Innovations in Digital Journalism: "
    "Technology, Roles, and Gender in Modern Newsrooms.” Journalism 24 (5): 938–957. "
    "https://doi.org/10.1080/1461670X.2023.2192295."
)
LEWIS = (
    "Lewis, S. C. 2019. “Artificial Intelligence and Journalism.” Journalism & Mass Communication "
    "Quarterly 96 (3): 673–695. https://doi.org/10.1177/1077699019859901."
)
GITLIN = (
    "Gitlin, T. (1980). The Whole World Is Watching: Mass Media in the Making and (Un)making of "
    "the New Left. Berkeley: University of California Press."
)


class AuthorMatchingTests(unittest.TestCase):
    def test_initials_match_full_given_names(self):
        self.assertTrue(names_match("Holman, L.", "Lynette Holman"))
        self.assertTrue(names_match("Lewis, S. C.", "Seth C. Lewis"))
        self.assertFalse(names_match("Lewis, M.", "Seth C. Lewis"))

    def test_first_author_must_be_the_canonical_first_author(self):
        cited = ["Lewis, S. C."]
        canonical = ["Meredith Broussard", "Nicholas Diakopoulos", "Seth C. Lewis"]
        self.assertGreater(author_coverage(cited, canonical), 0.5)
        self.assertFalse(first_authors_match(cited, canonical))

    def test_chinese_and_latin_names_are_not_treated_as_a_mismatch_signal(self):
        self.assertEqual(author_coverage(["程萧潇"], ["Xiaoxiao Cheng"]), 0.0)
        self.assertTrue(first_authors_match(["程萧潇"], ["Xiaoxiao Cheng"]))


class ReferenceParsingTests(unittest.TestCase):
    def test_chicago_reference_extracts_title_authors_and_ignores_doi_year(self):
        record = parse_citation(raw_text=HOLMAN)
        self.assertEqual(
            record.title,
            "Diffusion of Innovations in Digital Journalism: Technology, Roles, and Gender in Modern Newsrooms",
        )
        self.assertEqual(record.authors, ["Holman, L.", "G. P. Perreault"])
        self.assertEqual(record.year, 2023)
        self.assertEqual(record.doi, "10.1080/1461670x.2023.2192295")
        self.assertTrue(record.metadata["title_explicit"])

    def test_wrong_first_author_reference_keeps_the_cited_name(self):
        record = parse_citation(raw_text=LEWIS)
        self.assertEqual(record.title, "Artificial Intelligence and Journalism")
        self.assertEqual(record.authors[0], "Lewis, S. C")
        self.assertEqual(record.year, 2019)

    def test_apa_book_reference_parses_title_and_author(self):
        record = parse_citation(raw_text=GITLIN)
        self.assertEqual(record.authors, ["Gitlin, T."])
        self.assertIn("Whole World Is Watching", record.title)
        self.assertEqual(record.year, 1980)

    def test_fullwidth_gbt7714_punctuation_parses(self):
        parsed = parse_structured_reference(
            "张钹，朱军，苏航．迈向第三代人工智能［J］．中国科学：信息科学，2020，50（9）：1281-1302．"
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["title"], "迈向第三代人工智能")
        self.assertEqual(parsed["authors"], ["张钹", "朱军", "苏航"])
        self.assertEqual(parsed["venue"], "中国科学：信息科学")
        self.assertEqual(parsed["reference_format"], "gbt7714")

    def test_chinese_author_year_reference_parses_without_type_marker(self):
        record = parse_citation(
            raw_text="程萧潇、吴栎骞(2024):生成式人工智能在内容分析中的应用及测量效度评估，《全球传媒学刊》,第11卷第2期，51-78页。"
        )
        self.assertEqual(record.title, "生成式人工智能在内容分析中的应用及测量效度评估")
        self.assertEqual(record.authors, ["程萧潇", "吴栎骞"])
        self.assertEqual(record.venue, "全球传媒学刊")
        self.assertEqual(record.year, 2024)
        self.assertEqual(record.metadata["reference_format"], "zh_authoryear")

    def test_year_inside_a_doi_is_not_the_publication_year(self):
        self.assertIsNone(extract_year("https://doi.org/10.1080/1461670X.2023.2192295"))
        self.assertEqual(
            extract_year("Holman, L. 2022. Title. https://doi.org/10.1080/1461670X.2023.2192295."),
            2022,
        )

    def test_unstructured_free_text_stays_unstructured(self):
        record = parse_citation(raw_text="Asai et al., OpenScholar, arXiv:2411.14199, 2024")
        self.assertFalse(record.metadata["title_explicit"])
        self.assertNotIn("reference_format", record.metadata)


class ReferenceExtractionTests(unittest.TestCase):
    def test_hanging_indent_author_date_entries_stay_separate(self):
        text = "\n".join(
            [
                "References",
                "Abebe, R. 2019. Designing Algorithms for Social Good. Cornell",
                "   University Press.",
                "Holman, L., and G. P. Perreault. 2023. “Diffusion of Innovations in Digital Journalism.”",
                "   Journalism 24 (5): 938–957. https://doi.org/10.1080/1461670X.2023.2192295.",
                "18       R. ARAFAT AND G. COOPER",
                "Lewis, S. C. 2019. “Artificial Intelligence and Journalism.” Journalism Quarterly.",
            ]
        )
        candidates = extract_citation_candidates(text, source_format="text")
        titles = [item.get("title", "") for item in candidates]
        self.assertEqual(len(candidates), 3)
        self.assertTrue(any(title.startswith("Designing Algorithms") for title in titles))
        self.assertTrue(any("Diffusion of Innovations" in title for title in titles))
        self.assertTrue(any(title == "Artificial Intelligence and Journalism" for title in titles))
        self.assertFalse(any("ARAFAT" in item["raw_text"] for item in candidates))

    def test_two_column_reference_lines_are_reflowed(self):
        text = "\n".join(
            [
                "References",
                "Ariyanto, A. M., Hornsey, J., and Gallois, C. (2007). Group allegiances.             Giner-Sorolla, R., and Chaiken, S. (1994). Hostile media judgments.",
                "Group Process. Intergroup Relat. 10, 266–279. doi: 10.1177/1368430207074733                J. Exp. Soc. Psychol. 30, 165–180. doi: 10.1006/jesp.1994.1008",
                "Arpan, L. M., and Raney, A. A. (2003). News source effects here.             Gunther, A. C., and Schmitt, K. (2004). Mapping hostile media boundaries.",
                "Journal. Mass Commun. Quart. 80, 265–281. doi: 10.1177/107769900308000203                J. Commun. 54, 55–70. doi: 10.1111/j.1460-2466.2004.tb02613.x",
            ]
        )
        candidates = extract_citation_candidates(text, source_format="text")
        raw = " ".join(item["raw_text"] for item in candidates)
        self.assertGreaterEqual(len(candidates), 2)
        self.assertIn("10.1177/1368430207074733", raw)
        self.assertIn("10.1006/jesp.1994.1008", raw)
        self.assertTrue(any("Ariyanto" in item["raw_text"] and "Giner-Sorolla" not in item["raw_text"] for item in candidates))

    def test_glued_chinese_reference_paragraph_splits(self):
        text = (
            "参考文献(References)： "
            "程萧潇、吴栎骞(2024):生成式人工智能在内容分析中的应用及测量效度评估，《全球传媒学刊》,第11卷第2期，51-78页。 "
            "何向向(2023):基于智能内容生成的新闻传播生态重塑、风险与应对，《传媒》,第10期，32-34页。"
        )
        candidates = extract_citation_candidates(text, source_format="text")
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["authors"], ["程萧潇", "吴栎骞"])
        self.assertEqual(candidates[1]["authors"], ["何向向"])

    def test_html_bibliography_is_not_left_as_one_blob(self):
        html_text = (
            "<html><body><p>引言</p><p>参考文献(References)： "
            "程萧潇、吴栎骞(2024):生成式人工智能在内容分析中的应用及测量效度评估，《全球传媒学刊》,第11卷第2期，51-78页。 "
            "何向向(2023):基于智能内容生成的新闻传播生态重塑，《传媒》,第10期，32-34页。"
            "</p></body></html>"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.html"
            path.write_text(html_text, encoding="utf-8")
            candidates = load_citation_candidates(str(path))
        self.assertEqual(len(candidates), 2)

    def test_initials_inside_an_author_list_do_not_start_a_new_reference(self):
        text = "\n".join(
            [
                "References",
                "Broussard, M., N. Diakopoulos, A. L. Guzman, R. Abebe, M. Dupagne, and C. H. Chuan. 2019. "
                "“Artificial Intelligence and Journalism.” Journalism Quarterly. https://doi.org/10.1177/1077699019859901.",
                "Lewis, S. C. 2019. “Artificial Intelligence and Journalism.” Journalism Quarterly.",
            ]
        )
        candidates = extract_citation_candidates(text, source_format="text")
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["authors"][0], "Broussard, M.")

    def test_line_wrapped_doi_is_rejoined(self):
        text = "\n".join(
            [
                "References",
                "Holman, L., and G. P. Perreault. 2023. “Diffusion of Innovations in Digital Journalism.”",
                "   Journalism 24 (5): 938–957. https://doi.org/10.1080/",
                "   1461670X.2023.2192295.",
            ]
        )
        candidates = extract_citation_candidates(text, source_format="text")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["doi"], "10.1080/1461670x.2023.2192295")

    def test_pdf_bytes_raise_oserror_instead_of_unicode_decode_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "messy.pdf"
            path.write_bytes(b"%PDF-1.4\n\xff\xfe not utf-8 \x00\x01")
            with self.assertRaises(OSError):
                load_citation_candidates(str(path))


class MetadataVerdictTests(unittest.TestCase):
    def test_wrong_doi_is_reported_against_the_titled_paper(self):
        wrong = CitationRecord(
            citation_id="doi-target",
            title="Exploring News Innovations, Journalistic Routines, and Role Perceptions",
            authors=["Rana Arafat", "Colin Porlezza"],
            year=2023,
            doi="10.1080/1461670x.2023.2192295",
            source="crossref",
        )
        right = CitationRecord(
            citation_id="titled-paper",
            title="Diffusion of Innovations in Digital Journalism: Technology, Roles, and Gender in Modern Newsrooms",
            authors=["Lynette Holman", "Gregory P. Perreault"],
            year=2022,
            doi="10.1177/14648849211073441",
            source="crossref",
        )

        class Crossref(InMemoryMetadataSource):
            name = "crossref"

            def lookup_identifier(self, candidate):
                if candidate.doi == "10.1080/1461670x.2023.2192295":
                    return wrong
                return None

        result = verify_citation(parse_citation(raw_text=HOLMAN), Crossref([right, wrong]))
        self.assertEqual(result.verdict, Verdict.METADATA_MISMATCH)
        self.assertEqual(result.canonical_record.citation_id, "titled-paper")
        doi_diff = next(diff for diff in result.field_diffs if diff.field == "doi")
        self.assertFalse(doi_diff.matches)
        self.assertEqual(result.canonical_record.doi, "10.1177/14648849211073441")
        self.assertIn("Diffusion of Innovations", result.suggested_citation)

    def test_wrong_first_author_is_a_metadata_mismatch(self):
        paper = CitationRecord(
            citation_id="jmcq",
            title="Artificial Intelligence and Journalism",
            authors=["Meredith Broussard", "Nicholas Diakopoulos", "Seth C. Lewis"],
            year=2019,
            doi="10.1177/1077699019859901",
            source="crossref",
        )

        class Crossref(InMemoryMetadataSource):
            name = "crossref"

            def lookup_identifier(self, candidate):
                return paper

        result = verify_citation(parse_citation(raw_text=LEWIS), Crossref([paper]))
        self.assertEqual(result.verdict, Verdict.METADATA_MISMATCH)
        author_diff = next(diff for diff in result.field_diffs if diff.field == "authors")
        self.assertFalse(author_diff.matches)
        self.assertIn("Broussard", result.suggested_citation)

    def test_initials_do_not_false_flag_the_real_first_author(self):
        paper = CitationRecord(
            citation_id="holman",
            title="Diffusion of Innovations in Digital Journalism: Technology, Roles, and Gender in Modern Newsrooms",
            authors=["Lynette Holman", "Gregory P. Perreault"],
            year=2022,
            doi="10.1177/14648849211073441",
            source="crossref",
        )
        citation = parse_citation(
            raw_text=HOLMAN.replace("10.1080/1461670X.2023.2192295", "10.1177/14648849211073441")
        )

        class Crossref(InMemoryMetadataSource):
            name = "crossref"

            def lookup_identifier(self, candidate):
                return paper

        result = verify_citation(citation, Crossref([paper]))
        self.assertEqual(result.verdict, Verdict.VERIFIED)

    def test_book_review_is_not_suggested_as_the_book(self):
        review = CitationRecord(
            citation_id="review",
            title="The Whole World Is Watching: Mass Media in the Making and (Un)making of the New Left",
            authors=["Raymond A. Schroth"],
            year=1980,
            doi="10.1017/s0084255900042169",
            source="crossref",
            metadata={"type": "book-review"},
        )
        book = CitationRecord(
            citation_id="book",
            title="The Whole World Is Watching: Mass Media in the Making and (Un)making of the New Left",
            authors=["Todd Gitlin"],
            year=1980,
            source="openalex",
            metadata={"type": "book"},
        )
        result = verify_citation(parse_citation(raw_text=GITLIN), InMemoryMetadataSource([review, book]))
        self.assertEqual(result.verdict, Verdict.VERIFIED)
        self.assertEqual(result.canonical_record.citation_id, "book")

        review_only = verify_citation(parse_citation(raw_text=GITLIN), InMemoryMetadataSource([review]))
        self.assertEqual(review_only.verdict, Verdict.AMBIGUOUS)
        self.assertEqual(review_only.suggested_citation, "")
        self.assertIn("review", review_only.explanation.lower())


class McpDependencyTests(unittest.TestCase):
    def test_mcp2_rename_is_reported_as_an_incompatible_pin(self):
        message = mcp_dependency_message(
            ModuleNotFoundError("FastMCP was renamed to MCPServer in mcp 2.x")
        )
        self.assertIn("mcp>=1.2,<2", message)


if __name__ == "__main__":
    unittest.main()
