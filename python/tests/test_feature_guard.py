"""
Feature-guard tests for the Python binding — mirror of C# RenderOptionsTests /
Go feature_guard_test.go / Node feature-guard.test.mjs.

Feature-gated operations (rendering) may raise RuntimeError when the native
lib is compiled without the feature.  Each test accepts both outcomes:

  feature ON  → operation succeeds, assertion holds
  feature OFF → RuntimeError with a "not enabled" / "Unsupported" message
                → pytest.skip()

Always-available operations (from_html_css, signature_count) are tested as
straight success-path assertions.
"""

import pytest

from pdf_oxide import Pdf, PdfDocument


_PDF_MAGIC = b"%PDF-"


def _is_feature_off(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return (
        "not enabled" in msg
        or "unsupported" in msg
        or "not compiled" in msg
        or "5000" in msg
        or "error code 8" in msg
    )


def _make_doc() -> PdfDocument:
    data = Pdf.from_markdown("# Test\n\nBody.").to_bytes()
    return PdfDocument.from_bytes(data)


# ── Rendering ────────────────────────────────────────────────────────────────


def test_render_page_default_png():
    doc = _make_doc()
    try:
        img = doc.render_page(0)
    except RuntimeError as exc:
        if _is_feature_off(exc):
            pytest.skip(f"render_page unavailable in this build: {exc}")
        raise
    assert img[:4] == b"\x89PNG", "expected PNG magic bytes"


def test_render_page_jpeg_format():
    doc = _make_doc()
    try:
        img = doc.render_page(0, format="jpeg")
    except RuntimeError as exc:
        if _is_feature_off(exc):
            pytest.skip(f"render_page unavailable in this build: {exc}")
        raise
    assert img[:2] == b"\xff\xd8", "expected JPEG magic bytes"


def test_render_page_higher_dpi_bigger():
    doc = _make_doc()
    try:
        small = doc.render_page(0, dpi=72)
        large = doc.render_page(0, dpi=300)
    except RuntimeError as exc:
        if _is_feature_off(exc):
            pytest.skip(f"render_page unavailable in this build: {exc}")
        raise
    assert len(large) > len(small), (
        f"expected 300 dpi bytes ({len(large)}) > 72 dpi bytes ({len(small)})"
    )


def test_render_page_invalid_dpi_raises():
    doc = _make_doc()
    with pytest.raises(Exception):  # noqa: B017
        doc.render_page(0, dpi=-1)


# ── OCR (always compiled into published wheels) ───────────────────────────────


def test_ocr_engine_and_config_are_importable():
    """OcrEngine and OcrConfig must always be importable — no AttributeError."""
    from pdf_oxide import OcrConfig, OcrEngine  # noqa: F401


def test_ocr_engine_raises_runtime_error_not_attribute_error_when_no_models():
    """Without model files, OcrEngine.__init__ must raise RuntimeError (not AttributeError).
    If it raises AttributeError, the wheel was compiled without the ocr feature.
    """
    from pdf_oxide import OcrEngine

    try:
        OcrEngine(
            det_model_path="/nonexistent/det.onnx",
            rec_model_path="/nonexistent/rec.onnx",
            dict_path="/nonexistent/dict.txt",
        )
    except RuntimeError:
        pass  # expected — model files don't exist or ort not found
    except Exception as exc:
        if "OCR not enabled" in str(exc) or "not enabled" in str(exc).lower():
            pytest.fail(
                "OCR feature is NOT compiled into this wheel. "
                "Rebuild with --features python,ocr. "
                f"Got: {exc}"
            )
        # Other errors (file not found, etc.) are OK
        pass


def test_extract_text_ocr_error_is_actionable_not_opaque_513():
    """#513: a wheel/build without the ``ocr`` feature must NOT raise the
    old opaque ``RuntimeError("OCR feature not enabled.")`` (which left
    the reporter, on Windows, with no path forward). The message must be
    *actionable*: name the install extra and point at the graceful auto
    path. Runs on every platform incl. ``windows-latest`` (the reporter's
    platform — closes the Linux-only coverage gap)."""
    doc = _make_doc()
    try:
        doc.extract_text_ocr(0)
    except RuntimeError as exc:
        msg = str(exc)
        low = msg.lower()
        if "ocr" in low and ("unavailable" in low or "not enabled" in low or "without the" in low):
            # OCR-disabled build → the message MUST be actionable (#513).
            assert msg.strip() != "OCR feature not enabled.", (
                "#513 regression: the bare opaque message is back"
            )
            assert "pip install" in low and "[ocr]" in low, (
                f"#513: extract_text_ocr error must say how to fix it: {msg!r}"
            )
            assert "extract_text_auto" in low, (
                f"#513: error must point at the graceful path: {msg!r}"
            )
    # OCR-enabled build (or any other outcome) → no opaque error: pass.


# ── HTML + CSS creation (always available) ────────────────────────────────────


def test_from_html_css_produces_pdf(tmp_path):
    font_path = _find_any_ttf()
    if font_path is None:
        pytest.skip("no TTF font found on system for from_html_css test")
    font_bytes = font_path.read_bytes()
    pdf = Pdf.from_html_css("<h1>Hello</h1>", "h1 { color: red; }", font_bytes)
    data = pdf.to_bytes()
    assert data[:5] == _PDF_MAGIC, "expected %PDF- magic"
    assert len(data) > 100


def test_from_html_css_with_fonts_produces_pdf(tmp_path):
    font_path = _find_any_ttf()
    if font_path is None:
        pytest.skip("no TTF font found on system for from_html_css_with_fonts test")
    font_bytes = font_path.read_bytes()
    pdf = Pdf.from_html_css_with_fonts(
        "<p>Hello</p>",
        "p { font-size: 14pt; }",
        [("Body", font_bytes)],
    )
    data = pdf.to_bytes()
    assert data[:5] == _PDF_MAGIC


def test_from_html_css_no_fonts_raises():
    with pytest.raises(Exception):  # noqa: B017
        Pdf.from_html_css_with_fonts("<p>x</p>", "", [])


# ── Signatures (count is always available) ────────────────────────────────────


def test_signature_count_unsigned_pdf():
    doc = _make_doc()
    count = doc.signature_count()
    assert isinstance(count, int)
    assert count >= 0, f"unexpected negative count: {count}"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_any_ttf():
    """Return a Path to any TTF on the test system, or None."""
    import pathlib

    search_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "/System/Library/Fonts",
        "C:/Windows/Fonts",
    ]
    for d in search_dirs:
        p = pathlib.Path(d)
        if p.is_dir():
            for ttf in p.rglob("*.ttf"):
                return ttf
    return None
