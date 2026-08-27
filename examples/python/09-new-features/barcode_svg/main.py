# Barcode SVG generation
#
# Demonstrates generating 1D barcodes and QR codes as vector SVG strings.
# Run: python main.py

from __future__ import annotations

import os

import pdf_oxide


OUT_DIR = "output"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1D barcode — Code 128 SVG (barcode_type=0)
    svg = pdf_oxide.generate_barcode_svg(0, "PDF-OXIDE-0341")
    assert svg.startswith("<svg"), f"expected SVG, got: {svg[:40]}"
    path = os.path.join(OUT_DIR, "code128.svg")
    with open(path, "w") as f:
        f.write(svg)
    print(f"Written: {path} ({len(svg)} bytes)")

    # 1D barcode — EAN-13 SVG (barcode_type=2)
    svg = pdf_oxide.generate_barcode_svg(2, "5901234123457")
    assert svg.startswith("<svg")
    path = os.path.join(OUT_DIR, "ean13.svg")
    with open(path, "w") as f:
        f.write(svg)
    print(f"Written: {path} ({len(svg)} bytes)")

    # Data Matrix SVG (size=256)
    svg = pdf_oxide.generate_datamatrix_svg("https://github.com/yfedoseev/pdf_oxide", 1, 256)
    assert svg.startswith("<svg")
    assert "<rect" in svg, "QR SVG must contain rect elements"
    path = os.path.join(OUT_DIR, "datamatrix_code.svg")
    with open(path, "w") as f:
        f.write(svg)
    print(f"Written: {path} ({len(svg)} bytes)")

    # QR code SVG (error_correction=1=Medium, size=256)
    svg = pdf_oxide.generate_qr_svg("https://github.com/yfedoseev/pdf_oxide", 1, 256)
    assert svg.startswith("<svg")
    assert "<rect" in svg, "QR SVG must contain rect elements"
    path = os.path.join(OUT_DIR, "qr_code.svg")
    with open(path, "w") as f:
        f.write(svg)
    print(f"Written: {path} ({len(svg)} bytes)")

    print("All barcode SVG checks passed.")


if __name__ == "__main__":
    main()
