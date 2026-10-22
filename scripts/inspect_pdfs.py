"""Inspect PDF text in the project's take-home task documents."""

from pathlib import Path
import sys

try:
    import pymupdf
except ImportError:
    pymupdf = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_ROOT = PROJECT_ROOT / "Take-home task docs"
SAMPLE_LIMIT = 300


def inspect_pdf(pdf_path: Path) -> bool:
    """Print page text details and a sample; return whether an error occurred."""
    had_error = False
    relative_path = pdf_path.relative_to(PDF_ROOT)
    sample_parts: list[str] = []

    print(f"\n{relative_path}")

    try:
        document = pymupdf.open(pdf_path)
    except Exception as error:
        print(f"  ERROR: Could not open PDF: {error}", file=sys.stderr)
        return True

    try:
        print(f"  Pages: {document.page_count}")
        for page_number in range(document.page_count):
            try:
                page = document.load_page(page_number)
                text = page.get_text("text")
            except Exception as error:
                print(
                    f"  ERROR: Could not extract page {page_number + 1}: {error}",
                    file=sys.stderr,
                )
                had_error = True
                continue

            print(f"  Page {page_number + 1}: {len(text)} characters", end="")
            if not text.strip():
                print(" [NO EXTRACTED TEXT]")
            else:
                print()
                sample_parts.append(text)
    except Exception as error:
        print(f"  ERROR: Could not read PDF pages: {error}", file=sys.stderr)
        had_error = True
    finally:
        document.close()

    sample = " ".join(" ".join(sample_parts).split())
    if sample:
        if len(sample) > SAMPLE_LIMIT:
            sample = sample[:SAMPLE_LIMIT].rstrip() + "..."
        print(f"  Text sample: {sample}")
    else:
        print("  Text sample: [No text extracted]")

    return had_error


def main() -> int:
    if pymupdf is None:
        print(
            "ERROR: PyMuPDF is not installed in this Python environment. "
            "Install it with uv before running this script.",
            file=sys.stderr,
        )
        return 1

    if not PDF_ROOT.is_dir():
        print(f"ERROR: PDF folder does not exist: {PDF_ROOT}", file=sys.stderr)
        return 1

    had_error = False

    def report_walk_error(error: OSError) -> None:
        nonlocal had_error
        had_error = True
        print(f"ERROR: Could not scan folder: {error}", file=sys.stderr)

    pdf_paths: list[Path] = []
    for directory, _, filenames in PDF_ROOT.walk(on_error=report_walk_error):
        pdf_paths.extend(
            directory / filename
            for filename in filenames
            if Path(filename).suffix.lower() == ".pdf"
        )

    pdf_paths.sort()
    if not pdf_paths:
        print(f"No PDF files found under: {PDF_ROOT}")
        return 1 if had_error else 0

    print(f"Found {len(pdf_paths)} PDF file(s).")
    for pdf_path in pdf_paths:
        had_error = inspect_pdf(pdf_path) or had_error

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())