"""Create a traceable corrected layer from raw page-level extraction records."""

from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_INPUT_PATH = PROJECT_ROOT / "data" / "extracted" / "documents.jsonl"
CORRECTED_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "extracted" / "documents_corrected.jsonl"
)
CORRECTION_SET = "confirmed-devanagari-extraction-artifacts-v1"

CORRECTIONS = {
    "12_Bachat_Khata_Byaj_Dar_Suchna_SB-2026-03.pdf": (
        ("ग्रााहकों", "ग्राहकों"),
        ("त्रैैमासिक", "त्रैमासिक"),
        ("द्वाारा", "द्वारा"),
    ),
    "13_Varishth_Nagrik_Suvidhayen_SC-2026-01.pdf": (
        ("ग्रााहक", "ग्राहक"),
        ("ग्रााहकों", "ग्राहकों"),
        ("निःःशुल्क", "निःशुल्क"),
    ),
    "14_Digital_Dhokhadhadi_Se_Bachav_CA-2025-09.pdf": (
        ("ग्रााहक", "ग्राहक"),
        ("धोखाधड़ीी", "धोखाधड़ी"),
        ("सुरक्षाा", "सुरक्षा"),
        ("यात्राा", "यात्रा"),
    ),
    "15_Grih_Rin_Niti_HL-2025-08.pdf": (
        ("क्रेेडिट", "क्रेडिट"),
    ),
}


def _is_token_character(character: str) -> bool:
    return (
        character == "_"
        or unicodedata.category(character)[0] in {"L", "M", "N"}
    )


def replace_complete_tokens(text: str, source: str, replacement: str) -> tuple[str, int]:
    """Replace exact tokens while treating Unicode letters and marks as token parts."""
    pieces: list[str] = []
    cursor = 0
    count = 0

    while (match_start := text.find(source, cursor)) != -1:
        match_end = match_start + len(source)
        left_is_boundary = (
            match_start == 0 or not _is_token_character(text[match_start - 1])
        )
        right_is_boundary = (
            match_end == len(text) or not _is_token_character(text[match_end])
        )
        if left_is_boundary and right_is_boundary:
            pieces.extend((text[cursor:match_start], replacement))
            cursor = match_end
            count += 1
        else:
            pieces.append(text[cursor:match_end])
            cursor = match_end

    if not count:
        return text, 0
    pieces.append(text[cursor:])
    return "".join(pieces), count


def correct_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Counter[str]]]:
    corrected_records: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = {
        filename: Counter() for filename in CORRECTIONS
    }

    for record in records:
        corrected = dict(record)
        filename = record.get("source_pdf")
        applied: list[dict[str, Any]] = []
        rules = sorted(
            CORRECTIONS.get(filename, ()),
            key=lambda correction: len(correction[0]),
            reverse=True,
        )
        for source, replacement in rules:
            text, count = replace_complete_tokens(
                corrected["text"],
                source,
                replacement,
            )
            corrected["text"] = text
            counts[filename][source] += count
            if count:
                applied.append(
                    {
                        "source_token": source,
                        "replacement_token": replacement,
                        "occurrences": count,
                    }
                )
        corrected["correction_provenance"] = {
            "correction_set": CORRECTION_SET,
            "applied": applied,
        }
        corrected_records.append(corrected)

    return corrected_records, counts


def transform_jsonl(input_path: Path, output_path: Path) -> dict[str, Counter[str]]:
    """Write corrected records separately and return per-document token counts."""
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"Line {line_number} is not a text extraction record")
            records.append(record)

    corrected_records, counts = correct_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
            for record in corrected_records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path.replace(output_path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise
    return counts


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    if not RAW_INPUT_PATH.is_file():
        print(f"ERROR: Raw extraction file does not exist: {RAW_INPUT_PATH}", file=sys.stderr)
        return 1

    try:
        counts = transform_jsonl(RAW_INPUT_PATH, CORRECTED_OUTPUT_PATH)
    except (OSError, ValueError) as error:
        print(f"ERROR: Could not create corrected extraction: {error}", file=sys.stderr)
        return 1

    print(f"Corrected JSONL: {CORRECTED_OUTPUT_PATH}")
    missing_tokens: list[tuple[str, str]] = []
    for filename, rules in CORRECTIONS.items():
        print(filename)
        for source, replacement in rules:
            count = counts[filename][source]
            print(f"  {source} -> {replacement}: {count}")
            if count == 0:
                missing_tokens.append((filename, source))
    if missing_tokens:
        print("Configured tokens not found:")
        for filename, source in missing_tokens:
            print(f"  {filename}: {source}")
    else:
        print("Configured tokens not found: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
