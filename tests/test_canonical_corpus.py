import json
import unittest
from collections import defaultdict
from pathlib import Path

from scripts.build_canonical_corpus import select_canonical_records
from scripts.policy_metadata import (
    CANONICAL_SOURCE_RULES,
    DOCUMENT_METADATA,
    POLICY_RELATIONSHIPS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "extracted" / "documents.jsonl"
CORRECTED_PATH = PROJECT_ROOT / "data" / "extracted" / "documents_corrected.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class CanonicalCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_records = read_jsonl(RAW_PATH)
        cls.corrected_records = read_jsonl(CORRECTED_PATH)
        cls.canonical_records = select_canonical_records(
            cls.raw_records,
            cls.corrected_records,
            project_root=PROJECT_ROOT,
        )
        cls.by_document: dict[str, list[dict]] = defaultdict(list)
        for record in cls.canonical_records:
            cls.by_document[record["source_pdf"]].append(record)

    def test_selects_one_canonical_record_per_source_page(self) -> None:
        expected_page_keys = {
            (record["source_pdf"], record["page_number"])
            for record in self.raw_records
        }
        selected_page_keys = [
            (record["source_pdf"], record["page_number"])
            for record in self.canonical_records
        ]

        self.assertEqual(set(self.by_document), set(DOCUMENT_METADATA))
        self.assertEqual(len(selected_page_keys), len(set(selected_page_keys)))
        self.assertEqual(set(selected_page_keys), expected_page_keys)
        self.assertEqual(len(self.canonical_records), len(self.raw_records))

    def test_routes_special_sources_without_competing_duplicates(self) -> None:
        for filename in (
            "10_Branch_Working_Hours_Notice_ADM-2026-01.pdf",
            "16_Locker_Suvidha_Niyam_LOC-2016-07.pdf",
        ):
            records = self.by_document[filename]
            self.assertEqual(
                {record["extraction_method"] for record in records},
                {"manually_verified_transcription"},
            )
            self.assertEqual(
                {
                    record["canonical_source"]["text_source_path"]
                    for record in records
                },
                {CANONICAL_SOURCE_RULES[filename]["text_source_path"]},
            )

        for filename in (
            "12_Bachat_Khata_Byaj_Dar_Suchna_SB-2026-03.pdf",
            "13_Varishth_Nagrik_Suvidhayen_SC-2026-01.pdf",
            "14_Digital_Dhokhadhadi_Se_Bachav_CA-2025-09.pdf",
            "15_Grih_Rin_Niti_HL-2025-08.pdf",
        ):
            for record in self.by_document[filename]:
                self.assertEqual(
                    record["canonical_source"]["record_store"],
                    "documents_corrected.jsonl",
                )
                self.assertIn("correction_provenance", record)

        self.assertFalse(
            any(record["extraction_method"].startswith("tesseract") for record in self.canonical_records)
        )

    def test_selected_text_matches_its_declared_record_store(self) -> None:
        raw_by_key = {
            (record["source_pdf"], record["page_number"]): record
            for record in self.raw_records
        }
        corrected_by_key = {
            (record["source_pdf"], record["page_number"]): record
            for record in self.corrected_records
        }

        for selected in self.canonical_records:
            key = (selected["source_pdf"], selected["page_number"])
            source_records = (
                corrected_by_key
                if selected["canonical_source"]["record_store"]
                == "documents_corrected.jsonl"
                else raw_by_key
            )
            self.assertEqual(selected["text"], source_records[key]["text"])

    def test_preserves_page_identity_and_extraction_provenance(self) -> None:
        for record in self.canonical_records:
            self.assertIsInstance(record["source_pdf"], str)
            self.assertIsInstance(record["page_number"], int)
            self.assertIsInstance(record["page_count"], int)
            self.assertIsInstance(record["extraction_method"], str)
            if record["canonical_source"]["source_type"] == "corrected_extraction":
                self.assertIn("correction_provenance", record)

    def test_fixed_deposit_relationship_is_provision_scoped(self) -> None:
        relationship = next(
            item
            for item in POLICY_RELATIONSHIPS
            if item["relationship_id"] == "fd-2026-revision"
        )

        self.assertEqual(relationship["relationship_type"], "partial_supersession")
        self.assertEqual(relationship["effective_date"], "2026-03-01")
        self.assertEqual(
            {
                item["provision_id"]
                for item in relationship["superseded_provisions"]
            },
            {"fd-interest-rates", "fd-premature-withdrawal"},
        )
        self.assertEqual(
            {
                item["provision_id"]
                for item in relationship["replacing_provisions"]
            },
            {"fd-revised-interest-rates", "fd-revised-withdrawal-penalty"},
        )
        retained = {
            provision_id
            for item in relationship["retained_provisions"]
            for provision_id in item["provision_ids"]
        }
        self.assertIn("fd-eligibility", retained)
        self.assertIn("fd-other-conditions", retained)
        self.assertIn("fd-premature-withdrawal", retained)
        self.assertIn(
            "fd-eligibility",
            {
                provision["provision_id"]
                for provision in DOCUMENT_METADATA[
                    "01_Fixed_Deposit_Scheme_Circular_FD-2025-04.pdf"
                ]["provisions"]
            },
        )

    def test_locker_relationship_keeps_unsuperseded_rules_available(self) -> None:
        relationship = next(
            item
            for item in POLICY_RELATIONSHIPS
            if item["relationship_id"] == "locker-annual-rent-revision"
        )
        document_16 = "16_Locker_Suvidha_Niyam_LOC-2016-07.pdf"

        self.assertEqual(relationship["effective_date"], "2024-04-01")
        self.assertEqual(
            [item["provision_id"] for item in relationship["superseded_provisions"]],
            ["locker-annual-rent-2016"],
        )
        self.assertEqual(
            [item["provision_id"] for item in relationship["replacing_provisions"]],
            ["locker-revised-annual-rent"],
        )
        retained = set(relationship["retained_provisions"][0]["provision_ids"])
        self.assertIn("locker-late-penalty", retained)
        self.assertIn("locker-bank-liability", retained)
        self.assertNotIn("locker-annual-rent-2016", retained)

        document_metadata = self.by_document[document_16][0]["document_metadata"]
        available = {item["provision_id"] for item in document_metadata["provisions"]}
        self.assertIn("locker-late-penalty", available)
        self.assertIn("locker-bank-liability", available)
        rule_10 = next(
            item
            for item in document_metadata["provisions"]
            if item["provision_id"] == "locker-bank-liability"
        )
        self.assertEqual(rule_10["pages"], [1, 2])
        self.assertEqual(
            {record["page_number"] for record in self.by_document[document_16]},
            {1, 2},
        )


if __name__ == "__main__":
    unittest.main()
