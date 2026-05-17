# PKCS#12 CMS signing
#
# Run: python main.py

from __future__ import annotations

import os

import pdf_oxide


OUT_DIR = "output"

P12_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "tests", "fixtures", "test_signing.p12"
)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(P12_PATH):
        print(f"  SKIP: {P12_PATH} not found")
        return

    try:
        with open(P12_PATH, "rb") as f:
            p12_data = f.read()

        cert = pdf_oxide.Certificate.load_pkcs12(p12_data, "testpass")
        print(f"  Certificate subject: {cert.subject}")

        pdf_bytes: bytes = (
            pdf_oxide.DocumentBuilder()
            .title("Signed Invoice")
            .letter_page()
            .font("Helvetica", 12)
            .at(72, 720)
            .heading(1, "Signed Invoice")
            .at(72, 690)
            .paragraph("This document carries a CMS/PKCS#7 digital signature.")
            .done()
            .build()
        )

        signed: bytes = pdf_oxide.sign_pdf_bytes(pdf_bytes, cert, reason="Approved", location="HQ")

        path = os.path.join(OUT_DIR, "signed_document.pdf")
        with open(path, "wb") as f:
            f.write(signed)
        print(f"Written: {path} ({len(signed)} bytes)")
        assert b"/ByteRange" in signed, "ByteRange missing from signed PDF"
        print("  Signature verified: /ByteRange present.")
    except (NotImplementedError, AttributeError) as e:
        print(f"  SKIP: {e}")


if __name__ == "__main__":
    main()
