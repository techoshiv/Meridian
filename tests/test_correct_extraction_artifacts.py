import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.correct_extraction_artifacts import (
    CORRECTIONS,
    CORRECTION_SET,
    correct_records,
    replace_complete_tokens,
    transform_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_INPUT_PATH = PROJECT_ROOT / "data" / "extracted" / "documents.jsonl"


class CorrectExtractionArtifactsTests(unittest.TestCase):
    def test_every_listed_token_is_corrected(self) -> None:
        records = [
            {
                "source_pdf": filename,
                "page_number": 1,
                "page_count": 1,
                "extraction_method": "pymupdf",
                "review_required": True,
                "text": " ".join(source for source, _ in rules),
            }
            for filename, rules in CORRECTIONS.items()
        ]

        corrected, counts = correct_records(records)

        for original, result in zip(records, corrected, strict=True):
            filename = original["source_pdf"]
            expected_text = " ".join(
                replacement for _, replacement in CORRECTIONS[filename]
            )
            self.assertEqual(result["text"], expected_text)
            self.assertEqual(result["page_number"], original["page_number"])
            self.assertEqual(result["page_count"], original["page_count"])
            self.assertEqual(
                result["extraction_method"],
                original["extraction_method"],
            )
            self.assertEqual(result["review_required"], original["review_required"])
            self.assertEqual(
                result["correction_provenance"]["correction_set"],
                CORRECTION_SET,
            )
            self.assertTrue(result["correction_provenance"]["applied"])
            self.assertTrue(all(count == 1 for count in counts[filename].values()))

    def test_replacements_require_complete_unicode_tokens(self) -> None:
        text = "ग्रााहक ग्रााहकों xग्रााहक ग्रााहक़"

        result, count = replace_complete_tokens(text, "ग्रााहक", "ग्राहक")

        self.assertEqual(
            result,
            "ग्राहक ग्रााहकों xग्रााहक ग्रााहक़",
        )
        self.assertEqual(count, 1)

    def test_unrelated_hindi_and_combining_marks_are_preserved(self) -> None:
        text = "हिंदी किंतु सुरक्षाी"

        result, count = replace_complete_tokens(text, "सुरक्षाा", "सुरक्षा")

        self.assertEqual(result, text)
        self.assertEqual(count, 0)

    def test_unlisted_documents_keep_text_and_receive_empty_provenance(self) -> None:
        record = {
            "source_pdf": "01_Unrelated.pdf",
            "page_number": 2,
            "page_count": 3,
            "review_reasons": ["existing review note"],
            "text": "ग्रााहक हिंदी किंतु",
        }

        corrected, _ = correct_records([record])

        self.assertEqual(corrected[0]["text"], record["text"])
        self.assertEqual(corrected[0]["review_reasons"], record["review_reasons"])
        self.assertEqual(
            corrected[0]["correction_provenance"],
            {"correction_set": CORRECTION_SET, "applied": []},
        )

    def test_transform_does_not_modify_raw_input_bytes(self) -> None:
        raw_bytes = RAW_INPUT_PATH.read_bytes()
        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "derived.jsonl"
            transform_jsonl(RAW_INPUT_PATH, output_path)
            derived_records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(hashlib.sha256(RAW_INPUT_PATH.read_bytes()).hexdigest(), raw_hash)
        self.assertEqual(RAW_INPUT_PATH.read_bytes(), raw_bytes)
        self.assertEqual(len(derived_records), 19)
        doc_12 = next(
            record
            for record in derived_records
            if record["source_pdf"].startswith("12_")
        )
        self.assertIn("सभी ग्राहकों", doc_12["text"])


if __name__ == "__main__":
    unittest.main()
