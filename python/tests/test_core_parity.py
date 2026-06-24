"""Core functional test-parity suite (Python).

Mirrors the shared cross-language spec
(docs/releases/plans/v0.3.61/core-test-parity-spec.md) using the idiomatic
Python API. Same behaviors are asserted in every binding."""

import os

import pytest

import pdf_oxide


HERE = os.path.dirname(__file__)
FIXTURE = os.path.normpath(os.path.join(HERE, "..", "..", "tests", "fixtures", "simple.pdf"))


def _open():
    return pdf_oxide.PdfDocument(FIXTURE)


def _build_bytes() -> bytes:
    b = pdf_oxide.DocumentBuilder()
    (
        b.letter_page()
        .font("Helvetica", 12)
        .at(72, 720)
        .heading(1, "Core Parity")
        .at(72, 690)
        .paragraph("Functional parity across all language bindings.")
        .done()
    )
    return b.build()


def test_open_and_page_count():
    assert _open().page_count() == 1


def test_extract_text():
    assert isinstance(_open().extract_text(0), str)


def test_convert_markdown():
    assert isinstance(_open().to_markdown(0), str)


def test_convert_html():
    assert isinstance(_open().to_html(0), str)


def test_convert_plain():
    assert isinstance(_open().to_plain_text(0), str)


def test_search():
    assert isinstance(_open().search("the"), list)


def test_structured():
    assert _open().extract_structured(0) is not None


def test_create_pdf():
    assert _build_bytes().startswith(b"%PDF")


def test_from_bytes():
    assert pdf_oxide.PdfDocument.from_bytes(_build_bytes()).page_count() == 1


def test_encrypt_roundtrip():
    plain = _build_bytes()
    enc = pdf_oxide.PdfDocument.from_bytes(plain).to_bytes_encrypted(user_password="user123")
    assert enc.startswith(b"%PDF")
    assert enc != plain  # encryption changed the bytes


def test_open_error():
    with pytest.raises(Exception):  # noqa: B017
        pdf_oxide.PdfDocument("/no/such/file/does/not/exist.pdf")


def test_version():
    assert pdf_oxide.VERSION == "0.3.69"
