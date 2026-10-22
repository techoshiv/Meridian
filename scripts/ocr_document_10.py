"""OCR Document 10 and save the recognised text beside this script."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = (
    PROJECT_ROOT
    / "Take-home task docs"
    / "Documents"
    / "10_Branch_Working_Hours_Notice_ADM-2026-01.pdf"
)
OUTPUT_PATH = Path(__file__).resolve().parent / "document_10_ocr.txt"
DPI = 300


def main() -> int:
    try:
        import pymupdf
        import pytesseract
        from PIL import Image
    except ImportError as error:
        print(
            f"ERROR: OCR dependency is unavailable: {error}. "
            "This script requires PyMuPDF, pytesseract, and Pillow.",
            file=sys.stderr,
        )
        return 1

    if not PDF_PATH.is_file():
        print(f"ERROR: PDF does not exist: {PDF_PATH}", file=sys.stderr)
        return 1

    try:
        document = pymupdf.open(PDF_PATH)
    except Exception as error:
        print(f"ERROR: Could not open PDF: {error}", file=sys.stderr)
        return 1

    page_results: list[str] = []
    had_error = False
    scale = DPI / 72
    matrix = pymupdf.Matrix(scale, scale)

    try:
        for page_number in range(document.page_count):
            try:
                page = document.load_page(page_number)
                pixmap = page.get_pixmap(
                    matrix=matrix,
                    colorspace=pymupdf.csRGB,
                    alpha=False,
                )
                image = Image.frombytes(
                    "RGB",
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )
                text = pytesseract.image_to_string(image, lang="eng")
                page_results.append(f"--- Page {page_number + 1} ---\n{text.rstrip()}")
            except Exception as error:
                had_error = True
                print(
                    f"ERROR: Could not OCR page {page_number + 1}: {error}",
                    file=sys.stderr,
                )
                page_results.append(
                    f"--- Page {page_number + 1} ---\n[OCR ERROR: {error}]"
                )
    finally:
        document.close()

    try:
        OUTPUT_PATH.write_text("\n\n".join(page_results) + "\n", encoding="utf-8")
    except OSError as error:
        print(f"ERROR: Could not save OCR text: {error}", file=sys.stderr)
        return 1

    print(f"Saved OCR text to: {OUTPUT_PATH}")
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
