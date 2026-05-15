# Office format conversion: PDF ↔ DOCX / PPTX / XLSX — v0.3.41
#
# Demonstrates bidirectional office format conversion:
#   1. Build a PDF from markdown
#   2. Export to DOCX bytes (PDF → DOCX)
#   3. Export to PPTX bytes (PDF → PPTX)
#   4. Export to XLSX bytes (PDF → XLSX)
#   5. Round-trip: import DOCX back to PDF
#
# Run: python main.py

from __future__ import annotations

import os

import pdf_oxide


OUT_DIR = "output"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Build a sample PDF
    pdf_bytes = pdf_oxide.Pdf.from_markdown(
        "# Office Conversion Demo\n\nThis PDF will be exported to DOCX, PPTX, and XLSX."
    ).to_bytes()
    print(f"Built sample PDF: {len(pdf_bytes):,} bytes")

    doc = pdf_oxide.PdfDocument.from_bytes(pdf_bytes)

    # 1. PDF → DOCX
    docx_bytes = doc.to_docx_bytes()
    assert docx_bytes[:2] == b"PK", "DOCX output is not a valid ZIP/DOCX"
    print(f"PDF → DOCX: {len(docx_bytes):,} bytes — PASS")
    with open(os.path.join(OUT_DIR, "output.docx"), "wb") as f:
        f.write(docx_bytes)

    # 2. PDF → PPTX
    pptx_bytes = doc.to_pptx_bytes()
    assert pptx_bytes[:2] == b"PK", "PPTX output is not a valid ZIP/PPTX"
    print(f"PDF → PPTX: {len(pptx_bytes):,} bytes — PASS")
    with open(os.path.join(OUT_DIR, "output.pptx"), "wb") as f:
        f.write(pptx_bytes)

    # 3. PDF → XLSX
    xlsx_bytes = doc.to_xlsx_bytes()
    assert xlsx_bytes[:2] == b"PK", "XLSX output is not a valid ZIP/XLSX"
    print(f"PDF → XLSX: {len(xlsx_bytes):,} bytes — PASS")
    with open(os.path.join(OUT_DIR, "output.xlsx"), "wb") as f:
        f.write(xlsx_bytes)

    # Round-trips: office → PDF → office
    docx_pdf = pdf_oxide.OfficeConverter.from_docx_bytes(docx_bytes)
    docx_pdf_bytes = docx_pdf.to_bytes()
    assert docx_pdf_bytes[:5] == b"%PDF-", "DOCX → PDF failed"
    docx2 = pdf_oxide.PdfDocument.from_bytes(docx_pdf_bytes).to_docx_bytes()
    assert docx2[:2] == b"PK", "DOCX round-trip invalid"
    print(f"DOCX → PDF → DOCX: {len(docx2):,} bytes — PASS")

    pptx_pdf = pdf_oxide.OfficeConverter.from_pptx_bytes(pptx_bytes)
    pptx_pdf_bytes = pptx_pdf.to_bytes()
    assert pptx_pdf_bytes[:5] == b"%PDF-", "PPTX → PDF failed"
    pptx2 = pdf_oxide.PdfDocument.from_bytes(pptx_pdf_bytes).to_pptx_bytes()
    assert pptx2[:2] == b"PK", "PPTX round-trip invalid"
    print(f"PPTX → PDF → PPTX: {len(pptx2):,} bytes — PASS")

    xlsx_pdf = pdf_oxide.OfficeConverter.from_xlsx_bytes(xlsx_bytes)
    xlsx_pdf_bytes = xlsx_pdf.to_bytes()
    assert xlsx_pdf_bytes[:5] == b"%PDF-", "XLSX → PDF failed"
    xlsx2 = pdf_oxide.PdfDocument.from_bytes(xlsx_pdf_bytes).to_xlsx_bytes()
    assert xlsx2[:2] == b"PK", "XLSX round-trip invalid"
    print(f"XLSX → PDF → XLSX: {len(xlsx2):,} bytes — PASS")

    print("\nAll office conversion checks passed.")


if __name__ == "__main__":
    main()
