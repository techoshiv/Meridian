# Meridian

## PDF OCR scripts

The OCR scripts use PyMuPDF, pytesseract, and Pillow. Recreate the Python
environment with:

```powershell
uv sync
```

Tesseract OCR is a separate, external program; it is not installed by `uv`.
Install Tesseract for Windows and make sure `tesseract.exe` is available on
`PATH`. Document 10 uses the English language data (`eng`); Document 16 uses
the Hindi language data (`hin`). Install the corresponding
`*.traineddata` files in Tesseract's `tessdata` directory, then verify with:

```powershell
tesseract --list-langs
```

Run an OCR script from the project root with:

```powershell
uv run python scripts\ocr_document_10.py
uv run python scripts\ocr_document_16.py
```

## Document ingestion

Extract the 16 source PDFs in `Take-home task docs\Documents` into page-level
JSON Lines records with:

```powershell
uv run python scripts\extract_documents.py
```

The script writes `data\extracted\documents.jsonl`. Each JSON object contains
the source PDF filename, page number and count, extraction method, source PDF
text status, review flags, and page text. Documents 10 and 16 use their
manually verified transcriptions; their raw OCR files are not ingested.
Documents with no embedded text or suspicious extracted characters are
reported for review. Re-running the script replaces the generated JSONL file
instead of appending duplicate page records.

Create a separate corrected layer for the visually confirmed Devanagari
extraction artifacts, without changing the raw records, with:

```powershell
uv run python scripts\correct_extraction_artifacts.py
```

The derived file is `data\extracted\documents_corrected.jsonl`. Each record
retains its raw extraction metadata and identifies the correction set and
exact token replacements applied.

## Canonical sources and policy metadata

Build the canonical page records and the separate document/policy metadata
catalog with:

```powershell
uv run python scripts\build_canonical_corpus.py
```

This writes `data\extracted\documents_canonical.jsonl` and
`data\extracted\document_metadata.json`. Each PDF page is selected from exactly
one source: the verified transcriptions for Documents 10 and 16, corrected
records for Documents 12–15, and direct extraction for the remaining
documents. The metadata catalog records document identifiers, policy numbers,
source-supported dates, covered provisions, and explicitly scoped policy
relationships. A status of `not_assessed` means the files do not establish
which policy is currently in force beyond the specific relationships stated.
The raw and corrected extraction files remain separate and unchanged.

## Semantic chunking

Create provision-aware chunks from the canonical records only:

```powershell
uv run python scripts\chunk_documents.py
```

The script writes `data\chunks\chunks.jsonl`. It groups numbered provisions,
keeps headings with their text, and joins provisions that continue across PDF
pages. Known flattened rate and rent tables are emitted as labelled rows using
their source column order, alongside the applicable source conditions. Each
chunk retains source-page extraction details, correction provenance, document
metadata, and any policy relationships that reference that document.
The Document 3 account-variant/service-charge tables and Document 6
unauthorised-transaction liability table also expose stable, provenance-bearing
row records within their existing single chunks. Their labelled rows appear
once in chunk text; structured table fields are companion metadata, not
additional retrieval chunks.

The target maximum is 450 whitespace-delimited words. A provision exceeding
that size is split at line boundaries with up to 50 words of overlap; normal
provisions and tables remain intact. This corpus currently fits without
requiring long-provision splits. Chunk IDs are deterministic for unchanged
canonical input and metadata.

Run the project tests with:

```powershell
uv run python -m unittest discover -s tests -v
```

## Local hybrid retrieval

Retrieval uses the validated `data\chunks\chunks.jsonl` corpus. Lexical-only
mode is the default and requires no embedding package. To install optional
semantic support:

```powershell
uv sync --extra semantic
```

The semantic extra uses
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, a compact
multilingual model that supports English and Hindi, including Devanagari. The
model weights are not included in the Python package: the first semantic run
downloads roughly 470 MB of model files, then caches them in the Hugging Face
model cache. Allow approximately 1–2 GB of available RAM for CPU inference;
actual use depends on the installed PyTorch build and other processes.

Enable semantic retrieval explicitly from PowerShell:

```powershell
uv run python scripts\retrieval.py "home loan ke liye maximum tenure kitna hai?" --semantic
```

By default, semantic mode may download the model if it is not cached. To require
an already-cached local model and prevent downloads:

```powershell
uv run python scripts\retrieval.py "home loan ke liye maximum tenure kitna hai?" --semantic --local-model-only
```

The retrieval response reports `semantic_status` (`disabled`, `loaded`, or
`unavailable`) and includes `semantic_error` when model loading fails. In that
case it uses lexical-only scoring; lexical scores are not discounted because
semantic scoring is unavailable. Chunk embeddings are encoded once per
retriever instance and reused for subsequent questions. The default score
threshold of `0.18` is a configurable retrieval cutoff, not a probability.