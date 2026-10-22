import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from scripts.chunk_documents import (
    CHUNK_OVERLAP_WORDS,
    MAX_CHUNK_WORDS,
    _split_long_segment,
    build_chunks,
    build_file,
    _load_jsonl,
    _load_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = PROJECT_ROOT / "data" / "extracted" / "documents_canonical.jsonl"
METADATA_PATH = PROJECT_ROOT / "data" / "extracted" / "document_metadata.json"
GENERATED_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"


def load_chunks(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalized_text(text: str) -> str:
    return " ".join(text.split())


class ChunkDocumentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.canonical = _load_jsonl(CANONICAL_PATH)
        cls.metadata = _load_metadata(METADATA_PATH)
        cls.chunks = build_chunks(cls.canonical, cls.metadata)
        cls.by_document: dict[str, list[dict]] = defaultdict(list)
        for chunk in cls.chunks:
            cls.by_document[chunk["source_pdf"]].append(chunk)

    def find_chunk(self, filename_prefix: str, section_id: str) -> dict:
        return next(
            chunk
            for filename, chunks in self.by_document.items()
            if filename.startswith(filename_prefix)
            for chunk in chunks
            if section_id in chunk["section_ids"]
        )

    def test_all_canonical_pages_are_covered_once_or_as_overlap(self) -> None:
        canonical_pages = {
            (record["source_pdf"], record["page_number"])
            for record in self.canonical
        }
        chunk_pages = {
            (page["source_pdf"], page["page_number"])
            for chunk in self.chunks
            for page in chunk["source_pages"]
        }
        self.assertEqual(chunk_pages, canonical_pages)
        self.assertEqual(
            {chunk["source_pdf"] for chunk in self.chunks},
            {record["source_pdf"] for record in self.canonical},
        )
        for chunk in self.chunks:
            self.assertEqual(
                chunk["pages"],
                [page["page_number"] for page in chunk["source_pages"]],
            )
            self.assertEqual(chunk["page_start"], min(chunk["pages"]))
            self.assertEqual(chunk["page_end"], max(chunk["pages"]))

    def test_every_metadata_provision_is_assigned_to_a_chunk(self) -> None:
        for document in self.metadata["documents"]:
            expected = {
                provision["provision_id"]
                for provision in document.get("provisions", [])
            }
            actual = {
                provision_id
                for chunk in self.by_document[document["source_pdf"]]
                for provision_id in chunk["section_ids"]
            }
            self.assertEqual(actual, expected, document["source_pdf"])

    def test_chunks_only_use_canonical_sources_and_keep_provenance(self) -> None:
        ids = [chunk["chunk_id"] for chunk in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))
        for chunk in self.chunks:
            for page in chunk["source_pages"]:
                self.assertEqual(page["source_pdf"], chunk["source_pdf"])
                self.assertIsInstance(page["page_count"], int)
                self.assertIsInstance(page["extraction_method"], str)
                self.assertIn("canonical_source", page)
                if "correction_provenance" in page:
                    self.assertEqual(
                        page["correction_provenance"]["correction_set"],
                        "confirmed-devanagari-extraction-artifacts-v1",
                    )
                self.assertNotIn("tesseract", page["extraction_method"].lower())

    def test_cross_page_provisions_stay_complete_and_cite_both_pages(self) -> None:
        cases = (
            (
                "01_",
                "fd-premature-withdrawal",
                "penalty of 1.00%",
                "No penalty is charged where the deposit is closed prematurely",
            ),
            (
                "03_",
                "savings-joint-accounts",
                "5. Joint accounts and nomination",
                "Savings accounts may be held jointly by up to 4 persons",
            ),
            (
                "16_",
                "locker-bank-liability",
                "सौ गुना",
                "प्राकृतिक आपदा से हुई हानि",
            ),
        )
        for filename_prefix, section_id, first_text, continuation in cases:
            with self.subTest(filename=filename_prefix):
                chunk = self.find_chunk(filename_prefix, section_id)
                self.assertEqual(chunk["pages"], [1, 2])
                self.assertIn(first_text, chunk["text"])
                self.assertIn(continuation, chunk["text"])

    def test_fixed_deposit_table_keeps_labels_and_qualifying_conditions(self) -> None:
        chunk = self.find_chunk("02_", "fd-revised-interest-rates")
        table = chunk["tables"][0]
        self.assertEqual(
            table["columns"],
            [
                "Tenure",
                "General public (%)",
                "Senior citizens (%)",
                "Super senior citizens (%)",
            ],
        )
        first_row = table["rows"][0]
        self.assertEqual(
            first_row,
            {
                "Tenure": "7 days to 45 days",
                "General public (%)": "3.75",
                "Senior citizens (%)": "4.25",
                "Super senior citizens (%)": "4.50",
            },
        )
        self.assertEqual(
            [row["General public (%)"] for row in table["rows"]],
            ["3.75", "5.00", "6.25", "7.25", "7.00", "6.90", "6.50"],
        )
        self.assertEqual(
            [row["Senior citizens (%)"] for row in table["rows"]],
            ["4.25", "5.50", "6.75", "7.75", "7.50", "7.40", "7.00"],
        )
        self.assertEqual(
            [row["Super senior citizens (%)"] for row in table["rows"]],
            ["4.50", "5.75", "7.00", "8.00", "7.75", "7.65", "7.25"],
        )
        self.assertIn("domestic deposits below Rs. 2 crore", chunk["text"])
        self.assertIn("aged 80 years or above", chunk["text"])
        self.assertIn("on or after 01 March 2026", chunk["text"])
        penalty = self.find_chunk("02_", "fd-revised-withdrawal-penalty")
        self.assertEqual(
            penalty["tables"][0]["rows"],
            [
                {
                    "Deposit amount": "Up to Rs. 5,00,000",
                    "Penalty on premature withdrawal": "0.50%",
                },
                {
                    "Deposit amount": "Above Rs. 5,00,000",
                    "Penalty on premature withdrawal": "1.00% (unchanged)",
                },
            ],
        )

    def test_hindi_savings_table_keeps_balance_thresholds_with_rates(self) -> None:
        chunk = self.find_chunk("12_", "savings-revised-rates-hindi")
        self.assertEqual(
            chunk["tables"][0]["rows"],
            [
                {
                    "खाते में दैनिक शेष राशि": "1 लाख रुपये तक",
                    "ब्याज दर (वार्षिक)": "2.75 प्रतिशत",
                },
                {
                    "खाते में दैनिक शेष राशि": "1 लाख रुपये से अधिक और 10 लाख रुपये तक",
                    "ब्याज दर (वार्षिक)": "3.00 प्रतिशत",
                },
                {
                    "खाते में दैनिक शेष राशि": "10 लाख रुपये से अधिक",
                    "ब्याज दर (वार्षिक)": "3.50 प्रतिशत",
                },
            ],
        )
        self.assertIn("01 अप्रैल 2026", chunk["text"])

    def test_document_3_variant_and_charge_rows_are_labelled_with_provenance(self) -> None:
        variants = self.find_chunk("03_", "savings-variants")
        variant_table = next(
            table
            for table in variants["tables"]
            if table["table_id"] == "03-account-variants"
        )
        self.assertEqual(
            variant_table["columns"],
            [
                "Account variant",
                "Minimum Average Monthly Balance (MAB)",
                "Key features",
            ],
        )
        self.assertEqual(
            [
                (
                    row["Account variant"],
                    row["Minimum Average Monthly Balance (MAB)"],
                    row["Key features"],
                )
                for row in variant_table["rows"]
            ],
            [
                (
                    "Regular Savings",
                    "Metro/Urban: Rs. 5,000 Semi-urban: Rs. 2,500 Rural: Rs. 1,000",
                    "Unlimited branch transactions; Classic debit card",
                ),
                (
                    "Basic Savings",
                    "Nil",
                    "Maximum 4 withdrawals per month (including ATM); no cheque book",
                ),
                (
                    "Salary Plus",
                    "Nil while salary is credited",
                    "Converted to Regular Savings if no salary is credited for 3 consecutive months; Platinum debit card",
                ),
                (
                    "Senior Savings",
                    "Rs. 1,000 (all locations)",
                    "For customers aged 60 years or above; priority service at all branches",
                ),
            ],
        )

        charges = self.find_chunk("03_", "savings-service-charges")
        charge_table = next(
            table
            for table in charges["tables"]
            if table["table_id"] == "03-service-charges"
        )
        self.assertEqual(
            [
                (row["Service"], row["Charge (taxes extra)"])
                for row in charge_table["rows"]
            ],
            [
                (
                    "Non-maintenance of MAB (per quarter)",
                    "Metro/Urban Rs. 300; Semi-urban Rs. 150; Rural Rs. 75",
                ),
                ("Debit card annual fee", "Classic Rs. 250; Platinum Rs. 750"),
                ("Cheque book", "First 25 leaves per year free; thereafter Rs. 4 per leaf"),
                ("SMS alerts", "Rs. 15 per quarter"),
                (
                    "Cash deposit at a non-home branch",
                    "Free up to Rs. 25,000 per day; above that Rs. 3 per Rs. 1,000 (minimum Rs. 50)",
                ),
                (
                    "Withdrawals at other banks' ATMs",
                    "5 free per month (3 in metro locations); thereafter Rs. 21 per transaction",
                ),
                (
                    "Account closure",
                    "Within 14 days of opening: nil; 15 days to 1 year: Rs. 500; after 1 year: nil",
                ),
                ("Duplicate passbook", "Rs. 100"),
            ],
        )
        for table, chunk in ((variant_table, variants), (charge_table, charges)):
            self.assertEqual(table["source_pdf"], chunk["source_pdf"])
            self.assertEqual(table["pages"], [1])
            self.assertTrue(all(row["source_pdf"] == chunk["source_pdf"] for row in table["rows"]))
            self.assertTrue(all(row["pages"] == [1] for row in table["rows"]))
            row_ids = [row["row_id"] for row in table["rows"]]
            self.assertEqual(len(row_ids), len(set(row_ids)))
        self.assertEqual(variants["text"].count("Metro/Urban: Rs. 5,000"), 1)
        self.assertEqual(charges["text"].count("Rs. 300"), 1)

    def test_document_6_liability_rows_repeat_reporting_condition_and_provenance(self) -> None:
        chunk = self.find_chunk("06_", "grievance-electronic-transaction-liability")
        table = next(
            table
            for table in chunk["tables"]
            if table["table_id"] == "06-electronic-transaction-liability"
        )
        rows = table["rows"]
        self.assertEqual(len(rows), 5)
        self.assertEqual(
            [
                (row["Reporting time condition"], row.get("Account type"), row["Maximum customer liability"])
                for row in rows
            ],
            [
                ("Within 3 working days", None, "Zero liability"),
                ("4 to 7 working days", "Basic Savings accounts", "Rs. 5,000"),
                ("4 to 7 working days", "Other savings accounts", "Rs. 10,000"),
                ("4 to 7 working days", "Current accounts", "Rs. 25,000"),
                (
                    "Beyond 7 working days",
                    None,
                    "As decided under the Board-approved policy, case by case",
                ),
            ],
        )
        self.assertIn(
            "The disputed amount is credited to the customer's account as a shadow credit",
            chunk["text"],
        )
        self.assertEqual(table["source_pdf"], chunk["source_pdf"])
        self.assertEqual(table["pages"], [1])
        self.assertTrue(all(row["source_pdf"] == chunk["source_pdf"] for row in rows))
        self.assertTrue(all(row["pages"] == [1] for row in rows))
        row_ids = [row["row_id"] for row in rows]
        self.assertEqual(len(row_ids), len(set(row_ids)))
        self.assertEqual(chunk["text"].count("Reporting time condition: 4 to 7 working days"), 3)

    def test_rechunking_changes_only_document_3_and_6(self) -> None:
        rebuilt = build_chunks(self.canonical, self.metadata)
        existing = load_chunks(GENERATED_CHUNKS_PATH)
        existing_by_id = {chunk["chunk_id"]: chunk for chunk in existing}
        rebuilt_by_id = {chunk["chunk_id"]: chunk for chunk in rebuilt}
        target_prefixes = ("03_", "06_")
        self.assertEqual(
            {
                chunk_id: chunk
                for chunk_id, chunk in existing_by_id.items()
                if not chunk["source_pdf"].startswith(target_prefixes)
            },
            {
                chunk_id: chunk
                for chunk_id, chunk in rebuilt_by_id.items()
                if not chunk["source_pdf"].startswith(target_prefixes)
            },
        )

    def test_locker_rent_rows_keep_scope_and_conditions(self) -> None:
        chunk = self.find_chunk("11_", "locker-revised-annual-rent")
        rows = chunk["tables"][0]["rows"]
        self.assertEqual(
            [(row["Locker size"], row["Revised annual rent (from 01 April 2024)"]) for row in rows],
            [
                ("Small", "Rs. 2,000"),
                ("Medium", "Rs. 4,000"),
                ("Large", "Rs. 8,000"),
                ("Extra Large (select branches only)", "Rs. 12,000"),
            ],
        )
        self.assertIn("With effect from 01 April 2024", chunk["text"])
        text = normalized_text(chunk["text"])
        self.assertIn("supersedes paragraph 2 (annual rent)", text)
        self.assertIn("All other provisions of that circular remain in force", text)
        self.assertIn("25% lower", text)
        self.assertIn("Applicable taxes are charged extra", text)

    def test_only_declared_partial_supersessions_are_attached(self) -> None:
        metadata_relationships = {
            item["relationship_id"]: item
            for item in self.metadata["policy_relationships"]
        }
        fd = metadata_relationships["fd-2026-revision"]
        self.assertEqual(fd["effective_date"], "2026-03-01")
        self.assertEqual(
            {
                item["provision_id"]
                for item in fd["superseded_provisions"]
            },
            {"fd-interest-rates", "fd-premature-withdrawal"},
        )
        self.assertEqual(
            {
                item["provision_id"]
                for item in fd["replacing_provisions"]
            },
            {"fd-revised-interest-rates", "fd-revised-withdrawal-penalty"},
        )
        locker = metadata_relationships["locker-annual-rent-revision"]
        self.assertEqual(locker["effective_date"], "2024-04-01")
        self.assertEqual(
            [item["provision_id"] for item in locker["superseded_provisions"]],
            ["locker-annual-rent-2016"],
        )
        self.assertEqual(
            [item["provision_id"] for item in locker["replacing_provisions"]],
            ["locker-revised-annual-rent"],
        )
        self.assertIn(
            "fd-2026-revision",
            {
                relationship["relationship_id"]
                for relationship in self.find_chunk("01_", "fd-premature-withdrawal")[
                    "policy_relationships"
                ]
            },
        )
        self.assertIn(
            "locker-annual-rent-revision",
            {
                relationship["relationship_id"]
                for relationship in self.find_chunk("16_", "locker-bank-liability")[
                    "policy_relationships"
                ]
            },
        )
        self.assertEqual(
            {
                chunk["section_ids"][0]
                for chunk in self.by_document[
                    "01_Fixed_Deposit_Scheme_Circular_FD-2025-04.pdf"
                ]
            },
            {
                "fd-eligibility",
                "fd-deposit-amount",
                "fd-tenure",
                "fd-interest-rates",
                "fd-interest-payment",
                "fd-premature-withdrawal",
                "fd-renewal",
                "fd-loan-against-deposit",
                "fd-other-conditions",
            },
        )
        locker_sections = {
            chunk["section_ids"][0]
            for chunk in self.by_document[
                "16_Locker_Suvidha_Niyam_LOC-2016-07.pdf"
            ]
        }
        self.assertIn("locker-annual-rent-2016", locker_sections)
        self.assertIn("locker-late-penalty", locker_sections)
        self.assertIn("locker-bank-liability", locker_sections)

    def test_document_16_rule_4_keeps_verified_ten_percent_penalty(self) -> None:
        chunk = self.find_chunk("16_", "locker-late-penalty")
        text = normalized_text(chunk["text"])
        self.assertIn("किराया देय तिथि से 30 दिन के भीतर जमा न करने पर", text)
        self.assertIn("बकाया किराये पर 10 प्रतिशत दंड शुल्क", text)

    def test_build_is_deterministic_and_does_not_duplicate_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "chunks.jsonl"
            first = build_file(CANONICAL_PATH, METADATA_PATH, output)
            first_bytes = output.read_bytes()
            second = build_file(CANONICAL_PATH, METADATA_PATH, output)
            second_bytes = output.read_bytes()

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(
            [chunk["chunk_id"] for chunk in first],
            [chunk["chunk_id"] for chunk in second],
        )
        self.assertEqual(
            len({chunk["chunk_id"] for chunk in second}),
            len(second),
        )
        self.assertTrue(GENERATED_CHUNKS_PATH.parent.exists())

    def test_oversized_provision_splits_with_bounded_overlap(self) -> None:
        words = [f"word{index}" for index in range(MAX_CHUNK_WORDS * 2)]
        segment = {
            "lines": [(1, " ".join(words))],
            "provisions": [{"provision_id": "long-provision", "title": "Long"}],
        }

        chunks = _split_long_segment(segment)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(
                sum(len(line.split()) for _, line in chunk["lines"])
                <= MAX_CHUNK_WORDS
                for chunk in chunks
            )
        )
        self.assertEqual(
            chunks[0].get("overlap_words", 0),
            0,
        )
        self.assertEqual(chunks[1]["overlap_words"], CHUNK_OVERLAP_WORDS)


if __name__ == "__main__":
    unittest.main()
