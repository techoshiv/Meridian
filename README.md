# Meridian
# Multilingual Banking RAG Service

A retrieval-augmented generation (RAG) service for answering questions about banking policies using English and Hindi PDF documents, including a legacy PDF that uses a non-Unicode Devanagari font.

The service exposes a FastAPI endpoint, retrieves relevant document evidence, generates answers using a locally hosted Ollama model, and returns source citations. When the corpus does not provide sufficient evidence, the service is designed to abstain rather than invent banking information.

## Features

- Page-level PDF text extraction with source provenance.
- OCR workflows for image-based and legacy-font documents.
- Separate raw, corrected and canonical extraction layers.
- Document metadata and explicitly scoped policy relationships.
- Provision-aware chunking, including selected table structures and provisions spanning pages.
- Lexical retrieval with optional multilingual semantic retrieval.
- Local answer generation using Ollama.
- FastAPI `POST /ask` endpoint returning an answer, citations and a `found` flag.
- Automated tests for corpus construction, chunking, retrieval, answer generation and API behaviour.

## Architecture

The processing and answering pipeline consists of:

1. **Document extraction:** Extract text page by page and retain source metadata.
2. **OCR and verification:** Process documents requiring OCR and preserve verified transcriptions for difficult documents.
3. **Extraction corrections:** Apply targeted corrections to visually confirmed extraction artefacts without modifying raw records.
4. **Canonical corpus:** Select the appropriate source for each page and construct document and policy metadata.
5. **Chunking:** Group related provisions and tables while retaining page references and provenance.
6. **Retrieval:** Rank candidate chunks using lexical scoring and, optionally, multilingual semantic scoring.
7. **Answer generation:** Send the question and retrieved evidence to the local Ollama model.
8. **API response:** Return the answer, citations and `found` status.

## Requirements

- Python version specified in `.python-version`.
- `uv` for Python environment and dependency management.
- Ollama installed locally.
- The `llama3.2:latest` model available in Ollama.
- Tesseract OCR and the required language data if the OCR scripts need to be rerun.
- `pytest` to run the automated test suite.

## Installation

Run the following commands from the project root.

Install the project dependencies:

```bash
uv sync
```

If `pytest` is not already available in the environment, install it:

```bash
uv pip install --python .venv\Scripts\python.exe pytest
```

Check that Ollama is installed and the model is available:

```bash
ollama list
```

If the model is not installed, download it:

```bash
ollama pull llama3.2
```

Ensure that the Ollama service is running before sending supported questions to the API.

## PDF OCR

The OCR scripts use PyMuPDF, pytesseract and Pillow. Tesseract is a separate external program and is not installed by `uv`.

On Windows, install Tesseract OCR and ensure `tesseract.exe` is available on `PATH`. Install the required language data in Tesseract's `tessdata` directory.

Verify the installed languages:

```bash
tesseract --list-langs
```

Document 10 uses English language data (`eng`), while Document 16 uses Hindi language data (`hin`).

Run the OCR scripts separately from the project root when OCR needs to be reproduced:

```bash
uv run python scripts/ocr_document_10.py
uv run python scripts/ocr_document_16.py
```

The repository retains the raw OCR outputs and manually verified transcriptions. Documents 10 and 16 use their verified transcriptions as the authoritative text in the canonical corpus; their raw OCR files are retained for reference and are not ingested as authoritative content.

## Document ingestion

Extract the 16 source PDFs in `Take-home task docs/Documents` into page-level JSON Lines records:

```bash
uv run python scripts/extract_documents.py
```

Output:

`data/extracted/documents.jsonl`

Each JSON object contains the source PDF filename, page number and count, extraction method, source PDF text status, review flags and page text.

Documents with no embedded text or suspicious extracted characters are reported for review. Re-running the script replaces the generated JSONL file rather than appending duplicate page records.

## Extraction corrections

Create a corrected layer for visually confirmed Devanagari extraction artefacts:

```bash
uv run python scripts/correct_extraction_artifacts.py
```

Output:

`data/extracted/documents_corrected.jsonl`

Each record retains the raw extraction metadata and identifies the correction set and exact token replacements applied. The raw extraction records remain unchanged.

## Canonical corpus and policy metadata

Build canonical page records and the document/policy metadata catalogue:

```bash
uv run python scripts/build_canonical_corpus.py
```

Outputs:

- `data/extracted/documents_canonical.jsonl`
- `data/extracted/document_metadata.json`

Each PDF page is selected from one source: verified transcriptions for Documents 10 and 16, corrected records for Documents 12–15, and direct extraction for the remaining documents.

The metadata catalogue records document identifiers, policy numbers, source-supported dates, covered provisions and explicitly scoped policy relationships.

A status of `not_assessed` means the available documents do not establish which policy is currently in force beyond the specific relationships stated. Partial supersession is recorded only for provisions supported by the supplied documents and does not imply that an entire older policy has been replaced.

## Semantic chunking

Create provision-aware chunks from the canonical records:

```bash
uv run python scripts/chunk_documents.py
```

Output:

`data/chunks/chunks.jsonl`

The chunking pipeline:

- Groups numbered provisions and keeps headings with their text.
- Joins provisions that continue across PDF pages.
- Represents selected rate, rent and liability tables as labelled rows.
- Preserves source pages, extraction details, correction provenance, document metadata and policy relationships.
- Maintains deterministic chunk identifiers for unchanged canonical input and metadata.

The target maximum chunk size is 450 whitespace-delimited words. A provision exceeding that size is split at line boundaries with up to 50 words of overlap. The current corpus fits without requiring long-provision splits.

## Retrieval

Retrieval uses `data/chunks/chunks.jsonl`.

Lexical-only mode is the default and does not require an embedding package. It uses deterministic lexical matching and additional query signals to rank candidate evidence.

The configurable score threshold is a retrieval cutoff, not a calibrated probability of correctness.

### Optional semantic retrieval

Install the optional semantic dependencies:

```bash
uv sync --extra semantic
```

The optional model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, which supports multilingual retrieval, including English and Hindi.

The model weights are downloaded separately from the Python packages. The first semantic run may download approximately 470 MB of model files. Additional disk space and RAM are needed for model loading and CPU inference.

From PowerShell, run a query with semantic retrieval enabled:

```powershell
uv run python scripts\retrieval.py "home loan ke liye maximum tenure kitna hai?" --semantic
```

To require an already-cached local model and prevent downloads:

```powershell
uv run python scripts\retrieval.py "home loan ke liye maximum tenure kitna hai?" --semantic --local-model-only
```

The retrieval response reports `semantic_status` (`disabled`, `loaded` or `unavailable`) and includes `semantic_error` when model loading fails. If the model cannot be loaded, retrieval falls back to lexical-only scoring. Check the reported status before claiming semantic retrieval was used in a particular run.

The default score threshold of `0.18` is configurable and should not be interpreted as a probability.

## Running the FastAPI service

Start the API from the project root:

```bash
uv run uvicorn main:app --reload
```

The API is available at:

`http://127.0.0.1:8000`

Interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Ask a question

Send a POST request to `/ask` with a non-empty `question` field:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is the maximum home-loan tenure?\"}"
```

Example request body:

```json
{
  "question": "What is the maximum home-loan tenure?"
}
```

The response contains:

- `answer`: the generated answer or an abstention message.
- `citations`: source references derived from retrieved document metadata.
- `found`: a boolean indicating whether sufficient evidence was found.

Citations should be checked against the source PDF and the relevant section, rule, table or page metadata. The service is designed to abstain when the corpus does not support an answer rather than inventing banking terms or fees.

Supported questions require the local Ollama service to be available. Questions rejected by retrieval as unsupported should be handled without generating an answer from unsupported evidence.

## Evaluation: Five Supplied Sample Questions

The five supplied sample questions were tested through the FastAPI endpoint. Expected answers and sources are reproduced from the supplied sample-question document; actual answers and citations are recorded from the API responses.

| # | Query | Exact expected answer | Expected source | Actual API answer | Actual citation / evidence | Found | Result |
|---|---|---|---|---|---|---|---|
| S1 | What is the maximum loan-to-value ratio for a loan against debt mutual fund units? | 75% of the market value of the units. | `04_Loan_Against_Securities_Policy_LAS-2026-01.pdf`, section 2 | The maximum loan-to-value ratio for a loan against debt mutual fund units is 75%. | `04_Loan_Against_Securities_Policy_LAS-2026-01.pdf`, p. 1; `las-eligible-securities-ltv` | `true` | Passed |
| S2 | 10 लाख रुपये से अधिक शेष वाले बचत खाते पर ब्याज दर क्या है? | 3.50 प्रतिशत वार्षिक (01 अप्रैल 2026 से प्रभावी)। | `12_Bachat_Khata_Byaj_Dar_Suchna_SB-2026-03.pdf`, rate table | ब्याज दर (वार्षिक): 3.50 प्रतिशत 10 लाख रुपये से अधिक (01 अप्रैल 2026 से)। | `12_Bachat_Khata_Byaj_Dar_Suchna_SB-2026-03.pdf`, p. 1; `savings-revised-rates-hindi` | `true` | Passed |
| S3 | home loan ke liye maximum tenure kitna hai? | Home loan ki adhiktam avadhi 30 varsh (30 years) hai. | `15_Grih_Rin_Niti_HL-2025-08.pdf`, section 3 | grih rin kii adhikatam avadhi 30 varsh hai. | `15_Grih_Rin_Niti_HL-2025-08.pdf`, p. 1; `home-loan-tenure` | `true` | Passed |
| S4 | लॉकर का किराया देय तिथि से 30 दिन के भीतर जमा न करने पर कितना दंड लगता है? | बकाया किराये पर 10 प्रतिशत दंड शुल्क लगाया जाता है। | `16_Locker_Suvidha_Niyam_LOC-2016-07.pdf`, rule 4 | दंड शुल्क 10 प्रतिशत है। | `16_Locker_Suvidha_Niyam_LOC-2016-07.pdf`, p. 1; `locker-late-penalty` | `true` | Passed |
| S5 | What is the annual fee for the Bank's credit cards? | Not found. The documents do not cover credit cards (they only list debit card fees), so the system should say it could not find the answer rather than quote the debit card fee. | None (out of scope) | I don't have sufficient evidence in the retrieved sources to answer that. | No citations returned; abstention is appropriate. | `false` | Passed |

**Evaluation summary:** 5/5 sample questions passed. The system answered the four supported questions with citations and correctly abstained on the unsupported credit-card fee question.

### Evaluation criteria

The sample questions were assessed for:

1. Factual correctness against the supplied source documents.
2. Citation accuracy, including the source and relevant section, rule or table.
3. Handling of English, Devanagari Hindi and Romanised Hindi queries.
4. Abstention when the corpus does not support an answer.
5. Avoidance of unsupported rates, dates, fees or conditions.

The fifth question checks whether the system abstains rather than inferring a credit-card fee from unrelated debit-card information.

## Running the tests

Run the test suite with:

```bash
uv run python -m pytest -q
```

The tests cover corpus construction, extraction corrections, chunking, retrieval, answer generation and API behaviour. Some tests use mocks and do not require a running Ollama service.

The development test dependency must be available in the Python environment before running this command.

## Limitations

- OCR and extracted text may contain errors, particularly in image-only documents and legacy-font PDFs. Verified transcriptions and targeted corrections reduce but do not eliminate this risk.
- Retrieval may miss relevant evidence or rank an irrelevant chunk highly. The retrieval threshold is configurable and is not a probability of correctness.
- Optional semantic retrieval depends on the embedding model being installed and loaded successfully. Check `semantic_status` to establish whether semantic scoring was active.
- The answer generator is a local language model and may still produce inaccurate wording. Answers and citations must be evaluated against source evidence.
- Policy metadata records only relationships supported by the supplied documents. It does not independently establish the current status of every policy.
- The five supplied questions are a limited evaluation set and do not guarantee performance on unseen questions.

## AI-use disclosure

AI tools were used during development for architecture discussions, implementation assistance, code and test review, debugging, and documentation. The accompanying AI-use declaration identifies the tools used, the purpose of that assistance, the work personally designed or verified, and the rationale for significant implementation choices.

## Repository structure

```text
Meridian/
├── data/
│   ├── extracted/
│   └── chunks/
├── scripts/
│   ├── extract_documents.py
│   ├── correct_extraction_artifacts.py
│   ├── build_canonical_corpus.py
│   ├── chunk_documents.py
│   ├── retrieval.py
│   └── answer_generation.py
├── tests/
├── main.py
├──.gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Design priorities

The implementation prioritises source provenance, reproducible document processing, explicit handling of OCR-sensitive text, traceable retrieval results, source-derived citations and abstention when evidence is insufficient. These choices make the system easier to inspect, test and debug than a pipeline relying solely on unconstrained language-model generation.
