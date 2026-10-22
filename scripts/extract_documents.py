"""Extract one page-level JSONL record per source document page."""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "Take-home task docs" / "Documents"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "extracted" / "documents.jsonl"

VERIFIED_TRANSCRIPTIONS = {
    "10_Branch_Working_Hours_Notice_ADM-2026-01.pdf": (
        SCRIPT_DIR / "document_10_verified.txt"
    ),
    "16_Locker_Suvidha_Niyam_LOC-2016-07.pdf": (
        SCRIPT_DIR / "document_16_verified.txt"
    ),
}
SOURCE_TEXT_STATUS_OVERRIDES = {
    "16_Locker_Suvidha_Niyam_LOC-2016-07.pdf": "legacy_font_garbled",
}
PAGE_MARKER = re.compile(r"(?m)^--- Page (\d+) of (\d+) ---\s*$")
SOURCE_PAGE = re.compile(r"(?m)^Source page:\s*(\d+) of (\d+)\s*$")
SOURCE_PAGES = re.compile(r"(?m)^Source pages:\s*(\d+)-(\d+) of (\d+)\s*$")
SOURCE_PDF = re.compile(r"(?m)^Source PDF:\s*(.+?)\s*$")


class IngestionError(Exception):
    """Raised when a source cannot be converted into trustworthy page records."""


def _check_source_name(text: str, pdf_name: str) -> None:
    match = SOURCE_PDF.search(text)
    if match is None or match.group(1) != pdf_name:
        raise IngestionError(
            f"Verified transcription does not identify expected PDF {pdf_name!r}"
        )


def _verified_page_texts(
    pdf_name: str, transcription_path: Path, expected_page_count: int
) -> tuple[list[tuple[int, str]], list[str]]:
    if not transcription_path.is_file():
        raise IngestionError(
            f"Verified transcription does not exist: {transcription_path}"
        )

    contents = transcription_path.read_text(encoding="utf-8")
    _check_source_name(contents, pdf_name)
    notes: list[str] = []

    if pdf_name == "10_Branch_Working_Hours_Notice_ADM-2026-01.pdf":
        source_page = SOURCE_PAGE.search(contents)
        if source_page is None:
            raise IngestionError("Document 10 transcription has no source page header")
        page_end = source_page.end()
        page_number = int(source_page.group(1))
        page_count = int(source_page.group(2))
        page_text = contents[page_end:].strip()
        pages = [(page_number, page_text)]
        if page_number != 1 or page_count != expected_page_count:
            raise IngestionError(
                f"Document 10 transcription reports page {page_number} of "
                f"{page_count}, but PDF has {expected_page_count} page(s)"
            )
    else:
        source_pages = SOURCE_PAGES.search(contents)
        if source_pages is None:
            raise IngestionError("Document 16 transcription has no source pages header")
        start_page = int(source_pages.group(1))
        end_page = int(source_pages.group(2))
        page_count = int(source_pages.group(3))
        if start_page != 1 or end_page != expected_page_count or page_count != expected_page_count:
            raise IngestionError(
                "Document 16 transcription page range does not match its PDF"
            )

        marker_matches = list(PAGE_MARKER.finditer(contents))
        if len(marker_matches) != expected_page_count:
            raise IngestionError(
                f"Expected {expected_page_count} page markers; "
                f"found {len(marker_matches)}"
            )
        pages = []
        for index, marker in enumerate(marker_matches):
            page_number = int(marker.group(1))
            marker_page_count = int(marker.group(2))
            if page_number != index + 1 or marker_page_count != expected_page_count:
                raise IngestionError(
                    f"Invalid or out-of-order page marker: {marker.group(0)}"
                )
            text_start = marker.end()
            text_end = (
                marker_matches[index + 1].start()
                if index + 1 < len(marker_matches)
                else len(contents)
            )
            pages.append((page_number, contents[text_start:text_end].strip()))

        note_start = source_pages.end()
        note_end = marker_matches[0].start()
        note_text = contents[note_start:note_end].strip()
        if note_text:
            notes.append(note_text)

    if len(pages) != expected_page_count:
        raise IngestionError(
            f"Verified transcription contains {len(pages)} page(s); "
            f"PDF has {expected_page_count}"
        )
    for page_number, page_text in pages:
        if not page_text:
            raise IngestionError(
                f"Verified transcription has no text for page {page_number}"
            )
    return pages, notes


def _review_reasons(text: str) -> list[str]:
    reasons: list[str] = []
    if not text.strip():
        reasons.append("source page has no extractable text")
    if "\ufffd" in text:
        reasons.append("text contains Unicode replacement characters")
    if any(unicodedata.category(char) == "Co" for char in text):
        reasons.append("text contains private-use Unicode characters")
    if any(
        unicodedata.category(char).startswith("C")
        and char not in "\n\r\t\f"
        for char in text
    ):
        reasons.append("text contains non-printing control characters")
    if any(
        unicodedata.category(text[index]).startswith("M")
        and text[index] == text[index - 1]
        for index in range(1, len(text))
    ):
        reasons.append("text contains adjacent repeated Unicode combining marks")
    return reasons


def _read_source_page_status(document: Any, page_number: int) -> str:
    try:
        source_text = document.load_page(page_number - 1).get_text("text")
    except Exception as error:
        return f"error: {error}"
    return "available" if source_text.strip() else "no_extractable_text"


def _document_records(
    pdf_path: Path,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    import pymupdf

    try:
        document = pymupdf.open(pdf_path)
    except Exception as error:
        raise IngestionError(f"Could not open PDF: {error}") from error

    try:
        pdf_name = pdf_path.name
        source_notes: list[str] = []
        page_errors: dict[int, str] = {}
        if pdf_name in VERIFIED_TRANSCRIPTIONS:
            page_texts, source_notes = _verified_page_texts(
                pdf_name,
                VERIFIED_TRANSCRIPTIONS[pdf_name],
                document.page_count,
            )
            method = "manually_verified_transcription"
        else:
            page_texts = []
            method = "pymupdf"
            for page_index in range(document.page_count):
                try:
                    page_texts.append(
                        (page_index + 1, document.load_page(page_index).get_text("text"))
                    )
                except Exception as error:
                    page_texts.append((page_index + 1, ""))
                    page_errors[page_index + 1] = str(error)
                    source_notes.append(
                        f"Page {page_index + 1} extraction error: {error}"
                    )

        records: list[dict[str, Any]] = []
        errors: list[str] = []
        for page_number, text in page_texts:
            source_status = _read_source_page_status(document, page_number)
            source_status = SOURCE_TEXT_STATUS_OVERRIDES.get(
                pdf_name,
                source_status,
            )
            reasons = _review_reasons(text) if method == "pymupdf" else []
            if page_number in page_errors:
                reasons.append(
                    f"PyMuPDF text extraction failed: {page_errors[page_number]}"
                )
                errors.append(
                    f"{pdf_name} page {page_number}: "
                    f"PyMuPDF text extraction failed: {page_errors[page_number]}"
                )
            if source_status.startswith("error:"):
                reasons.append(
                    f"could not inspect source PDF page text ({source_status[7:]})"
                )
                errors.append(
                    f"{pdf_name} page {page_number}: "
                    f"could not inspect source PDF page text ({source_status[7:]})"
                )
            record: dict[str, Any] = {
                "source_pdf": pdf_name,
                "page_number": page_number,
                "page_count": document.page_count,
                "extraction_method": method,
                "source_pdf_text_status": source_status,
                "review_required": bool(reasons),
                "review_reasons": reasons,
                "source_notes": source_notes if page_number == 1 else [],
                "text": text,
            }
            records.append(record)
        return records, source_notes, errors
    finally:
        document.close()


def _write_jsonl(records: list[dict[str, Any]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path.replace(OUTPUT_PATH)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> int:
    try:
        import pymupdf  # noqa: F401
    except ImportError as error:
        print(f"ERROR: PyMuPDF is unavailable: {error}", file=sys.stderr)
        return 1

    if not SOURCE_DIR.is_dir():
        print(f"ERROR: PDF source directory does not exist: {SOURCE_DIR}", file=sys.stderr)
        return 1

    pdf_paths = sorted(
        path
        for path in SOURCE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if len(pdf_paths) != 16:
        print(
            f"ERROR: Expected 16 source PDFs under {SOURCE_DIR}; found {len(pdf_paths)}",
            file=sys.stderr,
        )
        return 1

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    review_records: list[dict[str, Any]] = []
    documents_processed = 0
    for pdf_path in pdf_paths:
        try:
            document_records, _, page_errors = _document_records(pdf_path)
        except (IngestionError, OSError) as error:
            errors.append(f"{pdf_path.name}: {error}")
            continue
        records.extend(document_records)
        errors.extend(page_errors)
        documents_processed += 1
        review_records.extend(
            record
            for record in document_records
            if record["review_required"]
            or record["source_pdf_text_status"] != "available"
        )

    try:
        _write_jsonl(records)
    except OSError as error:
        print(f"ERROR: Could not write JSONL output: {error}", file=sys.stderr)
        return 1

    print(f"Documents processed: {documents_processed}/16")
    print(f"Page records written: {len(records)}")
    print(f"Output: {OUTPUT_PATH}")
    if review_records:
        print("Pages requiring review or source-text attention:")
        for record in review_records:
            print(
                f"  {record['source_pdf']} page {record['page_number']}: "
                f"{'; '.join(record['review_reasons']) or record['source_pdf_text_status']}"
            )
    if errors:
        print("Errors:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print("Errors: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
