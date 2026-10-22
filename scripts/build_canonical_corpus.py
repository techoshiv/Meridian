"""Select one canonical page-record source and attach policy metadata."""

from __future__ import annotations

import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from .policy_metadata import (
        CANONICAL_SOURCE_RULES,
        DOCUMENT_METADATA,
        POLICY_RELATIONSHIPS,
    )
else:
    from policy_metadata import (
        CANONICAL_SOURCE_RULES,
        DOCUMENT_METADATA,
        POLICY_RELATIONSHIPS,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
SOURCE_DIR = PROJECT_ROOT / "Take-home task docs" / "Documents"
RAW_RECORDS_PATH = EXTRACTED_DIR / "documents.jsonl"
CORRECTED_RECORDS_PATH = EXTRACTED_DIR / "documents_corrected.jsonl"
CANONICAL_RECORDS_PATH = EXTRACTED_DIR / "documents_canonical.jsonl"
METADATA_PATH = EXTRACTED_DIR / "document_metadata.json"
CORRECTION_SET = "confirmed-devanagari-extraction-artifacts-v1"


class CanonicalSourceError(Exception):
    """Raised when input records do not satisfy canonical-source requirements."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CanonicalSourceError(f"Required records file does not exist: {path}")
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise CanonicalSourceError(
                        f"{path} line {line_number} is not a JSON object"
                    )
                records.append(record)
    except (OSError, json.JSONDecodeError) as error:
        raise CanonicalSourceError(f"Could not read {path}: {error}") from error
    return records


def _index_records(
    records: list[dict[str, Any]], label: str
) -> dict[str, dict[int, dict[str, Any]]]:
    indexed: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        filename = record.get("source_pdf")
        page_number = record.get("page_number")
        if not isinstance(filename, str) or not isinstance(page_number, int):
            raise CanonicalSourceError(
                f"{label} contains a record without source_pdf/page_number"
            )
        if page_number in indexed[filename]:
            raise CanonicalSourceError(
                f"{label} contains duplicate page {page_number} for {filename}"
            )
        if not isinstance(record.get("text"), str):
            raise CanonicalSourceError(
                f"{label} contains non-text content for {filename} page {page_number}"
            )
        indexed[filename][page_number] = record
    return indexed


def _validate_page_set(
    filename: str, pages: dict[int, dict[str, Any]], expected_count: int
) -> None:
    expected_pages = list(range(1, expected_count + 1))
    if sorted(pages) != expected_pages:
        raise CanonicalSourceError(
            f"{filename} has page records {sorted(pages)}; expected {expected_pages}"
        )
    for page_number, record in pages.items():
        if record.get("page_count") != expected_count:
            raise CanonicalSourceError(
                f"{filename} page {page_number} reports page_count "
                f"{record.get('page_count')}; expected {expected_count}"
            )


def _verify_transcription_source(
    filename: str,
    source_path: str | None,
    pages: dict[int, dict[str, Any]],
    project_root: Path,
) -> None:
    if source_path is None:
        raise CanonicalSourceError(
            f"Verified-transcription source path is missing for {filename}"
        )
    path = project_root / source_path
    if not path.is_file():
        raise CanonicalSourceError(f"Verified transcription does not exist: {path}")
    contents = path.read_text(encoding="utf-8")
    if f"Source PDF: {filename}" not in contents:
        raise CanonicalSourceError(
            f"Verified transcription {path} does not name {filename}"
        )
    for page_number, record in pages.items():
        if record.get("extraction_method") != "manually_verified_transcription":
            raise CanonicalSourceError(
                f"{filename} page {page_number} is not marked as a "
                "manually verified transcription"
            )


def select_canonical_records(
    raw_records: list[dict[str, Any]],
    corrected_records: list[dict[str, Any]],
    project_root: Path = PROJECT_ROOT,
    expected_page_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Return exactly one canonical record per page, with identity metadata."""
    raw_by_document = _index_records(raw_records, "raw records")
    corrected_by_document = _index_records(corrected_records, "corrected records")
    expected_counts = expected_page_counts or {}
    expected_sources = set(DOCUMENT_METADATA)

    if set(CANONICAL_SOURCE_RULES) != expected_sources:
        raise CanonicalSourceError("Canonical-source rules do not cover the metadata catalog")
    if set(raw_by_document) != expected_sources:
        raise CanonicalSourceError("Raw records do not exactly cover the 16 catalogued PDFs")
    if set(corrected_by_document) != expected_sources:
        raise CanonicalSourceError(
            "Corrected records do not exactly cover the 16 catalogued PDFs"
        )
    if expected_counts and set(expected_counts) != expected_sources:
        raise CanonicalSourceError("Expected page counts do not cover the metadata catalog")

    relation_ids: dict[str, list[str]] = defaultdict(list)
    for relationship in POLICY_RELATIONSHIPS:
        for reference in relationship["source_references"]:
            filename = reference["source_pdf"]
            if relationship["relationship_id"] not in relation_ids[filename]:
                relation_ids[filename].append(relationship["relationship_id"])

    canonical_records: list[dict[str, Any]] = []
    for filename in DOCUMENT_METADATA:
        rule = CANONICAL_SOURCE_RULES[filename]
        if rule["record_store"] == "documents_corrected.jsonl":
            selected_pages = corrected_by_document[filename]
            expected_method = "pymupdf"
            for page_number, record in selected_pages.items():
                provenance = record.get("correction_provenance")
                if (
                    not isinstance(provenance, dict)
                    or provenance.get("correction_set") != CORRECTION_SET
                ):
                    raise CanonicalSourceError(
                        f"{filename} page {page_number} lacks expected correction provenance"
                    )
        else:
            selected_pages = raw_by_document[filename]
            expected_method = (
                "manually_verified_transcription"
                if rule["source_type"] == "manually_verified_transcription"
                else "pymupdf"
            )

        count = expected_counts.get(
            filename,
            max(selected_pages, default=0),
        )
        if count < 1:
            raise CanonicalSourceError(f"No page records found for {filename}")
        _validate_page_set(filename, selected_pages, count)
        for page_number, record in selected_pages.items():
            if record.get("extraction_method") != expected_method:
                raise CanonicalSourceError(
                    f"{filename} page {page_number} uses "
                    f"{record.get('extraction_method')!r}; expected {expected_method!r}"
                )
        if rule["source_type"] == "manually_verified_transcription":
            _verify_transcription_source(
                filename,
                rule.get("text_source_path"),
                selected_pages,
                project_root,
            )

        metadata = copy.deepcopy(DOCUMENT_METADATA[filename])
        metadata["policy_relationship_ids"] = relation_ids.get(filename, [])
        for page_number in sorted(selected_pages):
            output_record = copy.deepcopy(selected_pages[page_number])
            output_record["canonical_source"] = {
                "source_type": rule["source_type"],
                "record_store": rule["record_store"],
                "text_source_path": rule["text_source_path"],
            }
            output_record["document_metadata"] = copy.deepcopy(metadata)
            canonical_records.append(output_record)

    keys = [
        (record["source_pdf"], record["page_number"])
        for record in canonical_records
    ]
    if len(keys) != len(set(keys)):
        raise CanonicalSourceError("Canonical output contains duplicate source pages")
    return canonical_records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path.replace(path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise


def build_outputs() -> list[dict[str, Any]]:
    pdf_paths = sorted(SOURCE_DIR.glob("*.pdf"))
    expected_names = set(DOCUMENT_METADATA)
    actual_names = {path.name for path in pdf_paths}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise CanonicalSourceError(
            f"Source PDF set mismatch; missing={missing}, unexpected={unexpected}"
        )

    import pymupdf

    page_counts = {}
    for path in pdf_paths:
        with pymupdf.open(path) as document:
            page_counts[path.name] = document.page_count
    raw_records = _load_jsonl(RAW_RECORDS_PATH)
    corrected_records = _load_jsonl(CORRECTED_RECORDS_PATH)
    canonical_records = select_canonical_records(
        raw_records,
        corrected_records,
        expected_page_counts=page_counts,
    )

    metadata_output = {
        "schema_version": 1,
        "documents": [
            {
                **copy.deepcopy(DOCUMENT_METADATA[filename]),
                "source_pdf": filename,
                "canonical_source": copy.deepcopy(CANONICAL_SOURCE_RULES[filename]),
            }
            for filename in DOCUMENT_METADATA
        ],
        "policy_relationships": copy.deepcopy(POLICY_RELATIONSHIPS),
    }
    _write_json(METADATA_PATH, metadata_output)
    _write_jsonl(CANONICAL_RECORDS_PATH, canonical_records)
    return canonical_records


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        records = build_outputs()
    except (CanonicalSourceError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    documents = {record["source_pdf"] for record in records}
    print(f"Canonical documents: {len(documents)}")
    print(f"Canonical page records: {len(records)}")
    print(f"Canonical records: {CANONICAL_RECORDS_PATH}")
    print(f"Document metadata and policy relationships: {METADATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
