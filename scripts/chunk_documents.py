"""Create provision-aware semantic chunks from the canonical page records."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from .build_canonical_corpus import CanonicalSourceError, _load_jsonl, _index_records
else:
    from build_canonical_corpus import CanonicalSourceError, _load_jsonl, _index_records


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_RECORDS_PATH = PROJECT_ROOT / "data" / "extracted" / "documents_canonical.jsonl"
DOCUMENT_METADATA_PATH = PROJECT_ROOT / "data" / "extracted" / "document_metadata.json"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
MAX_CHUNK_WORDS = 450
CHUNK_OVERLAP_WORDS = 50
NUMBERED_HEADING = re.compile(r"^\s*(\d{1,2})[.)]\s+\S")
WORD = re.compile(r"\S+")
CONTENT_PROVISION_MARKERS = {
    "10_Branch_Working_Hours_Notice_ADM-2026-01.pdf": {
        "branch-locker-hours": "Locker operations are permitted",
        "branch-digital-channel-hours": "Digital channels",
        "branch-cheque-holiday-processing": "Cheques falling due on a holiday",
    },
    "11_Locker_Rent_Revision_LOC-2024-03.pdf": {
        "locker-rural-rent": "Rent at rural branches",
        "locker-rent-notice": "Branches must inform all existing locker holders",
    },
}


class ChunkingError(Exception):
    """Raised when canonical input or metadata is inconsistent."""


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ChunkingError(f"Could not read document metadata {path}: {error}") from error
    if not isinstance(metadata, dict):
        raise ChunkingError(f"Document metadata is not an object: {path}")
    return metadata


def _provision_number(title: str) -> int | None:
    match = NUMBERED_HEADING.match(title)
    return int(match.group(1)) if match else None


def _heading_provision(
    line: str,
    provisions: list[dict[str, Any]],
    filename: str,
) -> dict[str, Any] | None:
    if filename.startswith("12_") and line.strip() == "खाते में दैनिक शेष राशि":
        return next(
            (
                provision
                for provision in provisions
                if provision.get("provision_id") == "savings-revised-rates-hindi"
            ),
            None,
        )
    match = NUMBERED_HEADING.match(line)
    if match:
        heading_number = int(match.group(1))
        return next(
            (
                provision
                for provision in provisions
                if _provision_number(provision.get("title", "")) == heading_number
            ),
            None,
        )

    stripped = line.strip()
    for provision in provisions:
        title = provision.get("title", "")
        title_prefix = title.split(":", 1)[0].strip()
        if title_prefix and stripped == title_prefix:
            return provision
    return None


def _segment_document(
    filename: str,
    page_records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    provisions = metadata.get("provisions", [])
    lines: list[tuple[int, str]] = []
    for record in sorted(page_records, key=lambda item: item["page_number"]):
        lines.extend(
            (record["page_number"], line)
            for line in record["text"].splitlines()
        )

    segments: list[dict[str, Any]] = []
    current_lines: list[tuple[int, str]] = []
    current_provision: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current_lines, current_provision
        if current_lines and any(text.strip() for _, text in current_lines):
            segments.append(
                {
                    "lines": current_lines,
                    "provisions": [copy.deepcopy(current_provision)]
                    if current_provision
                    else [],
                }
            )
        current_lines = []

    for page_number, line in lines:
        provision = _heading_provision(line, provisions, filename)
        if provision is not None:
            flush()
            current_provision = provision
        current_lines.append((page_number, line))
    flush()

    # Metadata may identify provisions on a page whose visual heading cannot be
    # matched to the extracted heading text. Keep the page intact and associate
    # all source-declared provisions on that page rather than guessing a split.
    if not any(segment["provisions"] for segment in segments):
        page_numbers = {record["page_number"] for record in page_records}
        page_provisions = [
            copy.deepcopy(provision)
            for provision in provisions
            if set(provision.get("pages", [])) & page_numbers
        ]
        if segments:
            segments[0]["provisions"] = page_provisions

    # Keep the first preamble with the first identified section so policy
    # identifiers and applicability language stay beside the provision.
    if len(segments) > 1 and not segments[0]["provisions"] and segments[1]["provisions"]:
        segments[1]["lines"] = segments[0]["lines"] + segments[1]["lines"]
        del segments[0]

    # Document 2's applicability and customer-category conditions qualify its
    # rate table, so retain sections 1-3 as one semantic unit.
    if filename.startswith("02_"):
        merged: list[dict[str, Any]] = []
        for segment in segments:
            ids = {item["provision_id"] for item in segment["provisions"]}
            if ids and ids <= {
                "fd-revised-interest-rates",
                "fd-super-senior-citizens",
                "fd-revised-rate-applicability",
            }:
                if merged and merged[-1].get("_fd_rate_group"):
                    merged[-1]["lines"].extend(segment["lines"])
                    merged[-1]["provisions"].extend(segment["provisions"])
                else:
                    segment["_fd_rate_group"] = True
                    merged.append(segment)
            else:
                merged.append(segment)
        segments = merged

    for segment in segments:
        text = "\n".join(line for _, line in segment["lines"])
        assigned = {
            provision["provision_id"]
            for provision in segment["provisions"]
        }
        for provision in provisions:
            marker = CONTENT_PROVISION_MARKERS.get(filename, {}).get(
                provision["provision_id"]
            )
            if marker and marker in text and provision["provision_id"] not in assigned:
                segment["provisions"].append(copy.deepcopy(provision))
                assigned.add(provision["provision_id"])

    return segments


def _format_known_tables(
    filename: str,
    lines: list[tuple[int, str]],
) -> tuple[list[tuple[int, str]], list[dict[str, Any]]]:
    """Render known flattened tables with labels while retaining exact cell text."""
    output = list(lines)
    tables: list[dict[str, Any]] = []
    table_specs: list[tuple[list[str], int, Any, str]] = []

    if filename.startswith("03_"):
        variants = _parse_account_variants(filename, output)
        if variants:
            output = _replace_table_region(
                output,
                "Variant",
                [
                    " | ".join(
                        f"{column}: {row[column]}"
                        for column in variants["columns"]
                    )
                    for row in variants["rows"]
                ],
            )
            tables.append(variants)
        if any(text.strip() == "Service" for _, text in output):
            service_charges = _parse_service_charges(filename, output)
            if service_charges:
                output = _replace_table_region(
                    output,
                    "Service",
                    [
                        " | ".join(
                            f"{column}: {row[column]}"
                            for column in service_charges["columns"]
                        )
                        for row in service_charges["rows"]
                    ],
                )
                tables.append(service_charges)
    elif filename.startswith("06_"):
        liability = _parse_liability_table(filename, output)
        if liability:
            output = _replace_table_region(
                output,
                "Time taken to report (from receipt of the Bank's alert)",
                [
                    " | ".join(
                        f"{column}: {row[column]}"
                        for column in liability["columns"]
                        if column in row
                    )
                    for row in liability["rows"]
                ],
                end_marker="The disputed amount is credited",
            )
            tables.append(liability)
    elif filename.startswith("01_"):
        table_specs.append(
            (
                ["Tenure", "General public (%)", "Senior citizens (%)"],
                2,
                re.compile(r"^(?:\d+\s+days|\d+\s+year|Above \d+\s+years)", re.I),
                "fd_rates",
            )
        )
    elif filename.startswith("02_"):
        table_specs.extend(
            [
                (
                    [
                        "Tenure",
                        "General public (%)",
                        "Senior citizens (%)",
                        "Super senior citizens (%)",
                    ],
                    3,
                    re.compile(r"^(?:\d+\s+days|\d+\s+year|Above \d+\s+years)", re.I),
                    "fd_rates",
                ),
                (
                    ["Deposit amount", "Penalty on premature withdrawal"],
                    1,
                    re.compile(r"^(?:Up to|Above) Rs\."),
                    "fd_penalties",
                ),
            ]
        )
    elif filename.startswith("11_"):
        table_specs.append(
            (
                [
                    "Locker size",
                    "Previous annual rent",
                    "Revised annual rent (from 01 April 2024)",
                ],
                2,
                re.compile(r"^(?:Small|Medium|Large|Extra Large\b)"),
                "locker_rents",
            )
        )
    elif filename.startswith("12_"):
        table_specs.append(
            (
                ["खाते में दैनिक शेष राशि", "ब्याज दर (वार्षिक)"],
                1,
                re.compile(r"^(?:\d+\s+लाख रुपये|1 लाख रुपये)"),
                "savings_rates",
            )
        )

    for headers, value_count, row_start, table_kind in table_specs:
        texts = [text.strip() for _, text in output]
        try:
            header_start = next(
                index
                for index in range(len(texts))
                if texts[index : index + len(headers)] == headers
            )
        except StopIteration:
            continue

        row_index = header_start + len(headers)
        rows: list[dict[str, str]] = []
        while row_index < len(texts) and row_start.match(texts[row_index]):
            label = texts[row_index]
            cells = texts[row_index + 1 : row_index + 1 + value_count]
            if len(cells) != value_count:
                raise ChunkingError(
                    f"Incomplete table row for {filename}: {label!r}"
                )
            if any(
                not cell
                or (
                    table_kind == "fd_rates"
                    and not re.fullmatch(r"\d+\.\d+", cell)
                )
                or (
                    table_kind == "fd_penalties"
                    and not re.fullmatch(r"\d+\.\d+%?(?: \(unchanged\))?", cell)
                )
                or (
                    table_kind == "locker_rents"
                    and not (cell.startswith("Rs.") or cell == "Not offered")
                )
                or (
                    table_kind == "savings_rates"
                    and not re.fullmatch(r"\d+\.\d+ प्रतिशत", cell)
                )
                for cell in cells
            ):
                raise ChunkingError(
                    f"Unexpected table cell for {filename}, row {label!r}: {cells}"
                )
            rows.append(
                {
                    headers[0]: label,
                    **dict(zip(headers[1:], cells, strict=True)),
                }
            )
            row_index += 1 + value_count

        if not rows:
            continue
        table_text = [
            " | ".join(headers),
            *[" | ".join(row[header] for header in headers) for row in rows],
        ]
        first_page = output[header_start][0]
        replacement = [(first_page, line) for line in table_text]
        output[header_start:row_index] = replacement
        tables.append({"columns": headers, "rows": rows})
    return output, tables


def _row_id(filename: str, table_id: str, index: int) -> str:
    return f"{filename[:2].lower()}-{table_id}-row-{index:02d}"


def _make_table(
    filename: str,
    table_id: str,
    columns: list[str],
    rows: list[dict[str, str]],
    pages: list[int],
) -> dict[str, Any]:
    rows_with_provenance = []
    for index, row in enumerate(rows, start=1):
        rows_with_provenance.append(
            {
                "row_id": _row_id(filename, table_id, index),
                "source_pdf": filename,
                "pages": pages,
                **row,
            }
        )
    return {
        "table_id": f"{filename[:2].lower()}-{table_id}",
        "source_pdf": filename,
        "pages": pages,
        "columns": columns,
        "rows": rows_with_provenance,
    }


def _replace_table_region(
    lines: list[tuple[int, str]],
    header: str,
    formatted_rows: list[str],
    end_marker: str | None = None,
) -> list[tuple[int, str]]:
    header_index = next(
        index for index, (_, text) in enumerate(lines) if text.strip() == header
    )
    page_number = lines[header_index][0]
    end_index = len(lines)
    if end_marker is not None:
        end_index = next(
            (
                index
                for index in range(header_index + 1, len(lines))
                if lines[index][1].strip().startswith(end_marker)
            ),
            len(lines),
        )
    return (
        lines[:header_index]
        + [(page_number, row_text) for row_text in formatted_rows]
        + lines[end_index:]
    )


def _parse_account_variants(
    filename: str,
    lines: list[tuple[int, str]],
) -> dict[str, Any] | None:
    texts = [text.strip() for _, text in lines]
    try:
        header_index = next(
            index
            for index in range(len(texts) - 3)
            if texts[index : index + 4]
            == [
                "Variant",
                "Minimum Average Monthly",
                "Balance (MAB)",
                "Key features",
            ]
        )
    except StopIteration:
        return None

    variant_rows = [
        ("Regular Savings", ["Regular", "Savings"]),
        ("Basic Savings", ["Basic", "Savings"]),
        ("Salary Plus", ["Salary Plus"]),
        ("Senior Savings", ["Senior", "Savings"]),
    ]
    starts: list[tuple[int, str, int]] = []
    search_from = header_index + 4
    for variant, source_lines in variant_rows:
        found = next(
            (
                index
                for index in range(search_from, len(texts))
                if texts[index : index + len(source_lines)] == source_lines
            ),
            None,
        )
        if found is None:
            raise ChunkingError(
                f"Could not locate account-variant row {variant!r} in {filename}"
            )
        starts.append((found, variant, len(source_lines)))
        search_from = found + len(source_lines)

    rows: list[dict[str, str]] = []
    for row_index, (start, variant, label_length) in enumerate(starts):
        end = starts[row_index + 1][0] if row_index + 1 < len(starts) else len(texts)
        cells = [text for text in texts[start + label_length : end] if text]
        if variant == "Regular Savings":
            mab_lines = cells[:3]
            if not all(
                line.startswith(prefix)
                for line, prefix in zip(
                    mab_lines,
                    ("Metro/Urban:", "Semi-urban:", "Rural:"),
                    strict=True,
                )
            ):
                raise ChunkingError(
                    f"Unexpected Regular Savings MAB rows in {filename}: {mab_lines}"
                )
            feature_lines = cells[3:]
        else:
            if len(cells) < 2:
                raise ChunkingError(
                    f"Incomplete account-variant row {variant!r} in {filename}"
                )
            mab_lines = cells[:1]
            feature_lines = cells[1:]
        if not feature_lines:
            raise ChunkingError(
                f"Missing key features for account variant {variant!r} in {filename}"
            )
        rows.append(
            {
                "Account variant": variant,
                "Minimum Average Monthly Balance (MAB)": " ".join(mab_lines),
                "Key features": " ".join(feature_lines),
            }
        )

    return _make_table(
        filename,
        "account-variants",
        [
            "Account variant",
            "Minimum Average Monthly Balance (MAB)",
            "Key features",
        ],
        rows,
        sorted({page for page, _ in lines}),
    )


def _parse_service_charges(
    filename: str,
    lines: list[tuple[int, str]],
) -> dict[str, Any] | None:
    texts = [text.strip() for _, text in lines]
    try:
        header_index = texts.index("Service")
    except ValueError:
        return None
    if texts[header_index + 1] != "Charge (taxes extra)":
        raise ChunkingError(f"Unexpected service-charge columns in {filename}")

    service_labels = [
        ("Non-maintenance of MAB (per quarter)", ["Non-maintenance of MAB (per", "quarter)"]),
        ("Debit card annual fee", ["Debit card annual fee"]),
        ("Cheque book", ["Cheque book"]),
        ("SMS alerts", ["SMS alerts"]),
        ("Cash deposit at a non-home branch", ["Cash deposit at a non-home branch"]),
        ("Withdrawals at other banks' ATMs", ["Withdrawals at other banks' ATMs"]),
        ("Account closure", ["Account closure"]),
        ("Duplicate passbook", ["Duplicate passbook"]),
    ]
    starts: list[tuple[int, str, int]] = []
    search_from = header_index + 2
    for label, source_lines in service_labels:
        found = next(
            (
                index
                for index in range(search_from, len(texts))
                if texts[index : index + len(source_lines)] == source_lines
            ),
            None,
        )
        if found is None:
            raise ChunkingError(
                f"Could not locate service-charge row {label!r} in {filename}"
            )
        starts.append((found, label, len(source_lines)))
        search_from = found + len(source_lines)

    rows: list[dict[str, str]] = []
    for row_index, (start, label, label_length) in enumerate(starts):
        end = starts[row_index + 1][0] if row_index + 1 < len(starts) else len(texts)
        value_lines = [text for text in texts[start + label_length : end] if text]
        if not value_lines:
            raise ChunkingError(
                f"Missing service-charge value for {label!r} in {filename}"
            )
        rows.append(
            {
                "Service": label,
                "Charge (taxes extra)": " ".join(value_lines),
            }
        )

    return _make_table(
        filename,
        "service-charges",
        ["Service", "Charge (taxes extra)"],
        rows,
        sorted({page for page, _ in lines}),
    )


def _parse_liability_table(
    filename: str,
    lines: list[tuple[int, str]],
) -> dict[str, Any] | None:
    texts = [text.strip() for _, text in lines]
    condition_header = "Time taken to report (from receipt of the Bank's alert)"
    try:
        header_index = texts.index(condition_header)
    except ValueError:
        return None
    if texts[header_index + 1] != "Maximum customer liability":
        raise ChunkingError(f"Unexpected liability-table columns in {filename}")

    cursor = header_index + 2
    source_rows: list[dict[str, str]] = []

    def add_row(condition: str, liability: str, account_type: str | None = None) -> None:
        row = {
            "Reporting time condition": condition,
            "Maximum customer liability": liability,
        }
        if account_type is not None:
            row["Account type"] = account_type
        source_rows.append(row)

    if texts[cursor] != "Within 3 working days" or texts[cursor + 1] != "Zero liability":
        raise ChunkingError(f"Unexpected first reporting-time band in {filename}")
    add_row(texts[cursor], texts[cursor + 1])
    cursor += 2

    if texts[cursor] != "4 to 7 working days":
        raise ChunkingError(f"Unexpected second reporting-time band in {filename}")
    condition = texts[cursor]
    cursor += 1
    for expected_type in (
        "Basic Savings accounts",
        "Other savings accounts",
        "Current accounts",
    ):
        line = texts[cursor]
        prefix = f"{expected_type}: "
        if not line.startswith(prefix):
            raise ChunkingError(
                f"Unexpected account liability row in {filename}: {line!r}"
            )
        add_row(condition, line[len(prefix) :], expected_type)
        cursor += 1

    if not texts[cursor].startswith("Beyond 7 working days"):
        raise ChunkingError(f"Unexpected final reporting-time band in {filename}")
    if texts[cursor] != "Beyond 7 working days":
        raise ChunkingError(
            f"Unexpected beyond-7-day condition in {filename}: {texts[cursor]!r}"
        )
    if texts[cursor + 1] != "As decided under the Board-approved policy, case by case":
        raise ChunkingError(f"Unexpected beyond-7-day liability in {filename}")
    add_row(texts[cursor], texts[cursor + 1])

    return _make_table(
        filename,
        "electronic-transaction-liability",
        ["Reporting time condition", "Account type", "Maximum customer liability"],
        source_rows,
        sorted({lines[header_index][0]}),
    )


def _split_long_segment(segment: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = [
        (page_number, word)
        for page_number, line in segment["lines"]
        for word in WORD.findall(line)
    ]
    if len(tokens) <= MAX_CHUNK_WORDS:
        return [segment]

    chunks: list[dict[str, Any]] = []
    step = MAX_CHUNK_WORDS - CHUNK_OVERLAP_WORDS
    for start in range(0, len(tokens), step):
        window = tokens[start : start + MAX_CHUNK_WORDS]
        lines: list[tuple[int, str]] = []
        for page_number, word in window:
            if lines and lines[-1][0] == page_number:
                lines[-1] = (page_number, f"{lines[-1][1]} {word}")
            else:
                lines.append((page_number, word))
        chunks.append(
            {
                **segment,
                "lines": lines,
                "overlap_words": CHUNK_OVERLAP_WORDS if start else 0,
            }
        )
        if start + MAX_CHUNK_WORDS >= len(tokens):
            break
    return chunks


def _relationship_is_relevant(
    relationship: dict[str, Any],
    filename: str,
    section_ids: set[str],
) -> bool:
    references = {
        item.get("source_pdf")
        for item in relationship.get("source_references", [])
    }
    if filename not in references:
        return False
    if not section_ids:
        return True
    relationship_ids = {
        item.get("provision_id")
        for field in (
            "superseded_provisions",
            "replacing_provisions",
        )
        for item in relationship.get(field, [])
    }
    relationship_ids.update(
        provision_id
        for item in relationship.get("retained_provisions", [])
        if item.get("source_pdf") == filename
        for provision_id in item.get("provision_ids", [])
    )
    return bool(section_ids & relationship_ids)


def _chunk_id(
    document_id: str,
    filename: str,
    section_ids: list[str],
    pages: list[int],
    text: str,
    ordinal: int,
) -> str:
    stable_payload = "\0".join(
        [filename, ",".join(section_ids), ",".join(map(str, pages)), text]
    )
    digest = hashlib.sha256(stable_payload.encode("utf-8")).hexdigest()[:10]
    section = "-".join(section_ids) if section_ids else "general"
    return f"{document_id.lower()}-{section}-p{pages[0]}-{ordinal:02d}-{digest}"


def build_chunks(
    canonical_records: list[dict[str, Any]],
    metadata_catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    indexed = _index_records(canonical_records, "canonical records")
    documents = {
        item["source_pdf"]: item
        for item in metadata_catalog.get("documents", [])
        if isinstance(item, dict) and isinstance(item.get("source_pdf"), str)
    }
    if not documents or set(indexed) != set(documents):
        raise ChunkingError("Canonical pages and metadata catalog cover different PDFs")
    relationships = metadata_catalog.get("policy_relationships", [])
    chunks: list[dict[str, Any]] = []

    for filename in documents:
        page_records = indexed[filename]
        page_counts = {record.get("page_count") for record in page_records.values()}
        if len(page_counts) != 1:
            raise ChunkingError(f"Inconsistent page_count for {filename}")
        expected_count = next(iter(page_counts))
        if sorted(page_records) != list(range(1, expected_count + 1)):
            raise ChunkingError(f"Missing canonical page records for {filename}")
        document_metadata = documents[filename]
        segments = _segment_document(
            filename,
            list(page_records.values()),
            document_metadata,
        )
        document_chunks: list[dict[str, Any]] = []
        for segment in segments:
            formatted_lines, tables = _format_known_tables(filename, segment["lines"])
            segment = {**segment, "lines": formatted_lines}
            for split_segment in _split_long_segment(segment):
                text = "\n".join(line for _, line in split_segment["lines"]).strip()
                if not text:
                    continue
                page_numbers = sorted(
                    {page_number for page_number, _ in split_segment["lines"]}
                )
                section_ids = [
                    provision["provision_id"]
                    for provision in split_segment["provisions"]
                ]
                section_titles = [
                    provision["title"]
                    for provision in split_segment["provisions"]
                ]
                source_pages = []
                for page_number in page_numbers:
                    page_record = page_records[page_number]
                    source_page = {
                        "source_pdf": filename,
                        "page_number": page_number,
                        "page_count": page_record["page_count"],
                        "extraction_method": page_record["extraction_method"],
                        "canonical_source": copy.deepcopy(
                            page_record.get("canonical_source")
                        ),
                    }
                    for field in (
                        "source_pdf_text_status",
                        "review_required",
                        "review_reasons",
                        "source_notes",
                        "correction_provenance",
                    ):
                        if field in page_record:
                            source_page[field] = copy.deepcopy(page_record[field])
                    source_pages.append(source_page)
                relevant_relationships = [
                    copy.deepcopy(relationship)
                    for relationship in relationships
                    if _relationship_is_relevant(
                        relationship,
                        filename,
                        set(section_ids),
                    )
                ]
                ordinal = len(document_chunks) + 1
                document_chunks.append(
                    {
                        "chunk_id": _chunk_id(
                            document_metadata["document_id"],
                            filename,
                            section_ids,
                            page_numbers,
                            text,
                            ordinal,
                        ),
                        "source_pdf": filename,
                        "page_start": page_numbers[0],
                        "page_end": page_numbers[-1],
                        "pages": page_numbers,
                        "section_ids": section_ids,
                        "section_titles": section_titles,
                        "text": text,
                        "word_count": len(WORD.findall(text)),
                        "overlap_words": split_segment.get("overlap_words", 0),
                        "tables": copy.deepcopy(tables),
                        "source_pages": source_pages,
                        "document_metadata": copy.deepcopy(document_metadata),
                        "policy_relationships": relevant_relationships,
                    }
                )
        chunks.extend(document_chunks)

    ids = [chunk["chunk_id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ChunkingError("Chunk IDs are not unique")
    covered_pages: dict[str, set[int]] = defaultdict(set)
    for chunk in chunks:
        covered_pages[chunk["source_pdf"]].update(chunk["pages"])
    for filename, page_records in indexed.items():
        if covered_pages[filename] != set(page_records):
            raise ChunkingError(
                f"Chunk coverage mismatch for {filename}: "
                f"{sorted(covered_pages[filename])} != {sorted(page_records)}"
            )
    return chunks


def _write_chunks(path: Path, chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
            for chunk in chunks:
                output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        temporary_path.replace(path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise


def build_file(
    canonical_path: Path = CANONICAL_RECORDS_PATH,
    metadata_path: Path = DOCUMENT_METADATA_PATH,
    output_path: Path = CHUNKS_PATH,
) -> list[dict[str, Any]]:
    canonical_records = _load_jsonl(canonical_path)
    metadata_catalog = _load_metadata(metadata_path)
    chunks = build_chunks(canonical_records, metadata_catalog)
    _write_chunks(output_path, chunks)
    return chunks


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        chunks = build_file()
    except (ChunkingError, CanonicalSourceError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Semantic chunks: {len(chunks)}")
    print(f"Chunk records: {CHUNKS_PATH}")
    print(
        f"Settings: max_words={MAX_CHUNK_WORDS}, "
        f"overlap_words={CHUNK_OVERLAP_WORDS} (only for oversized provisions)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
