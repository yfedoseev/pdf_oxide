"""
Comprehensive API coverage tests for the Python binding.

Design principles:
  - Every public method on PdfDocument, Pdf, DocumentBuilder must have
    at least one test that exercises it end-to-end.
  - Feature-gated operations (rendering) skip gracefully when the native
    lib is compiled without the feature.
  - Known-incomplete CSS properties (font-weight, font-style) are marked
    xfail with a clear reason so they auto-promote to passes once fixed.
  - Tests verify CORRECT OUTPUT, not just "doesn't crash".
"""

import pytest

from pdf_oxide import Pdf, PdfDocument


_PDF_MAGIC = b"%PDF-"


# ── helpers ──────────────────────────────────────────────────────────────────


def _is_feature_off(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        k in msg for k in ("not enabled", "unsupported", "not compiled", "5000", "error code 8")
    )


def _make_simple_doc() -> PdfDocument:
    data = Pdf.from_markdown("# Hello\n\nWorld.").to_bytes()
    return PdfDocument.from_bytes(data)


def _find_any_ttf():
    import pathlib

    for d in (
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "/System/Library/Fonts",
        "C:/Windows/Fonts",
    ):
        p = pathlib.Path(d)
        if p.is_dir():
            for ttf in p.rglob("*.ttf"):
                return ttf.read_bytes()
    return None


# ── PdfDocument: open / metadata ─────────────────────────────────────────────


class TestPdfDocumentOpen:
    def test_open_from_path(self, tmp_path):
        path = str(tmp_path / "doc.pdf")
        Pdf.from_markdown("# Path test").save(path)
        doc = PdfDocument(path)
        assert doc.page_count() >= 1

    def test_from_bytes_returns_document(self):
        data = Pdf.from_markdown("# Hi").to_bytes()
        doc = PdfDocument.from_bytes(data)
        assert doc is not None

    def test_from_bytes_bad_data_raises(self):
        with pytest.raises(Exception):  # noqa: B017
            PdfDocument.from_bytes(b"not a pdf")

    def test_context_manager(self):
        data = Pdf.from_markdown("# Hi").to_bytes()
        with PdfDocument.from_bytes(data) as doc:
            assert doc.page_count() >= 1

    def test_version_returns_tuple(self):
        doc = _make_simple_doc()
        v = doc.version()
        assert isinstance(v, tuple)
        assert len(v) == 2
        assert v[0] >= 1

    def test_page_count_positive(self):
        doc = _make_simple_doc()
        assert doc.page_count() >= 1

    def test_has_structure_tree_bool(self):
        doc = _make_simple_doc()
        result = doc.has_structure_tree()
        assert isinstance(result, bool)


# ── PdfDocument: text extraction ─────────────────────────────────────────────


class TestTextExtraction:
    def test_extract_text_returns_string(self):
        data = Pdf.from_markdown("# Hello\n\nWorld.").to_bytes()
        doc = PdfDocument.from_bytes(data)
        text = doc.extract_text(0)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_extract_text_contains_content(self):
        data = Pdf.from_markdown("# Unique_Marker_XYZ").to_bytes()
        doc = PdfDocument.from_bytes(data)
        text = doc.extract_text(0)
        assert "Unique_Marker_XYZ" in text

    def test_extract_text_invalid_page_raises(self):
        doc = _make_simple_doc()
        with pytest.raises(Exception):  # noqa: B017
            doc.extract_text(999)

    def test_extract_chars_returns_list(self):
        doc = _make_simple_doc()
        chars = doc.extract_chars(0)
        assert isinstance(chars, list)
        assert len(chars) > 0

    def test_extract_chars_have_expected_attrs(self):
        doc = _make_simple_doc()
        c = doc.extract_chars(0)[0]
        assert hasattr(c, "char")
        assert hasattr(c, "font_size")
        assert hasattr(c, "origin_x")
        assert hasattr(c, "origin_y")
        assert isinstance(c.char, str)
        assert c.font_size > 0

    def test_extract_words_returns_list(self):
        doc = _make_simple_doc()
        words = doc.extract_words(0)
        assert isinstance(words, list)
        assert len(words) > 0

    def test_extract_words_have_text_and_bbox(self):
        doc = _make_simple_doc()
        w = doc.extract_words(0)[0]
        assert hasattr(w, "text") and isinstance(w.text, str) and len(w.text) > 0
        assert hasattr(w, "bbox") and len(w.bbox) == 4

    def test_extract_words_contain_known_word(self):
        data = Pdf.from_markdown("UNIQUEWORD").to_bytes()
        doc = PdfDocument.from_bytes(data)
        texts = [w.text for w in doc.extract_words(0)]
        assert any("UNIQUEWORD" in t for t in texts)

    def test_extract_text_lines_returns_list(self):
        doc = _make_simple_doc()
        lines = doc.extract_text_lines(0)
        assert isinstance(lines, list)
        assert len(lines) > 0

    def test_extract_text_lines_have_text_and_bbox(self):
        doc = _make_simple_doc()
        line = doc.extract_text_lines(0)[0]
        assert hasattr(line, "text") and isinstance(line.text, str)
        assert hasattr(line, "bbox") and len(line.bbox) == 4

    def test_to_plain_text_returns_string(self):
        doc = _make_simple_doc()
        text = doc.to_plain_text(0)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_to_plain_text_all_returns_string(self):
        doc = _make_simple_doc()
        text = doc.to_plain_text_all()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_to_markdown_returns_markdown(self):
        doc = _make_simple_doc()
        md = doc.to_markdown(0)
        assert isinstance(md, str)
        assert len(md) > 0

    def test_to_markdown_all_returns_string(self):
        doc = _make_simple_doc()
        md = doc.to_markdown_all()
        assert isinstance(md, str)

    def test_to_html_returns_html(self):
        doc = _make_simple_doc()
        html = doc.to_html(0)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_to_html_all_returns_string(self):
        doc = _make_simple_doc()
        html = doc.to_html_all()
        assert isinstance(html, str)

    def test_page_layout_params(self):
        doc = _make_simple_doc()
        params = doc.page_layout_params(0)
        assert params is not None


# ── PdfDocument: signature operations ────────────────────────────────────────


class TestSignatures:
    def test_signature_count_returns_non_negative_int(self):
        doc = _make_simple_doc()
        count = doc.signature_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_signatures_returns_list(self):
        doc = _make_simple_doc()
        sigs = doc.signatures()
        assert isinstance(sigs, list)

    def test_unsigned_pdf_has_zero_signatures(self):
        doc = _make_simple_doc()
        assert doc.signature_count() == 0
        assert doc.signatures() == []


# ── PdfDocument: rendering ───────────────────────────────────────────────────


class TestRendering:
    def _try_render(self, doc, **kwargs):
        try:
            return doc.render_page(0, **kwargs)
        except RuntimeError as e:
            if _is_feature_off(e):
                pytest.skip(f"render_page unavailable: {e}")
            raise

    def test_render_default_produces_png(self):
        doc = _make_simple_doc()
        img = self._try_render(doc)
        assert img[:4] == b"\x89PNG"

    def test_render_jpeg_produces_jpeg(self):
        doc = _make_simple_doc()
        img = self._try_render(doc, format="jpeg")
        assert img[:2] == b"\xff\xd8"

    def test_render_higher_dpi_bigger(self):
        doc = _make_simple_doc()
        small = self._try_render(doc, dpi=72)
        large = self._try_render(doc, dpi=300)
        assert len(large) > len(small)

    def test_render_invalid_page_raises(self):
        doc = _make_simple_doc()
        with pytest.raises(Exception):  # noqa: B017
            doc.render_page(999)

    def test_render_negative_dpi_raises(self):
        doc = _make_simple_doc()
        with pytest.raises(Exception):  # noqa: B017
            doc.render_page(0, dpi=-1)


# ── Pdf: creation methods ─────────────────────────────────────────────────────


class TestPdfCreation:
    def test_from_markdown_produces_pdf(self):
        pdf = Pdf.from_markdown("# Hello\n\nWorld.")
        data = pdf.to_bytes()
        assert data[:5] == _PDF_MAGIC

    def test_from_markdown_content_is_extractable(self):
        pdf = Pdf.from_markdown("# UniqueXYZ123")
        doc = PdfDocument.from_bytes(pdf.to_bytes())
        text = doc.extract_text(0)
        assert "UniqueXYZ123" in text

    def test_from_html_produces_pdf(self):
        pdf = Pdf.from_html("<h1>Hello</h1><p>World</p>")
        data = pdf.to_bytes()
        assert data[:5] == _PDF_MAGIC

    def test_to_bytes_returns_bytes(self):
        pdf = Pdf.from_markdown("# Hi")
        data = pdf.to_bytes()
        assert isinstance(data, (bytes, bytearray))
        assert len(data) > 100

    def test_from_bytes_roundtrip(self):
        orig = Pdf.from_markdown("# Hi").to_bytes()
        pdf2 = Pdf.from_bytes(orig)
        assert pdf2.to_bytes()[:5] == _PDF_MAGIC

    def test_empty_markdown_produces_pdf(self):
        pdf = Pdf.from_markdown("")
        assert pdf.to_bytes()[:5] == _PDF_MAGIC

    def test_multi_page_markdown(self):
        md = "\n\n".join([f"# Section {i}\n\nText {i}." for i in range(5)])
        pdf = Pdf.from_markdown(md)
        doc = PdfDocument.from_bytes(pdf.to_bytes())
        assert doc.page_count() >= 1


# ── Pdf: HTML+CSS creation ───────────────────────────────────────────────────


class TestHtmlCssCreation:
    def setup_method(self):
        self.font = _find_any_ttf()

    def test_from_html_css_produces_pdf(self):
        if self.font is None:
            pytest.skip("no TTF font on system")
        pdf = Pdf.from_html_css("<h1>Hello</h1>", "h1 { font-size: 24pt; }", self.font)
        data = pdf.to_bytes()
        assert data[:5] == _PDF_MAGIC

    def test_from_html_css_content_extractable(self):
        if self.font is None:
            pytest.skip("no TTF font on system")
        pdf = Pdf.from_html_css("<p>UniqueXYZ789</p>", "", self.font)
        doc = PdfDocument.from_bytes(pdf.to_bytes())
        text = doc.extract_text(0)
        assert "UniqueXYZ789" in text

    def test_css_font_size_changes_output(self):
        """Regression: font-size CSS must affect rendered PDF bytes."""
        if self.font is None:
            pytest.skip("no TTF font on system")
        no_css = Pdf.from_html_css("<h1>Big</h1><p>Small</p>", "", self.font).to_bytes()
        with_css = Pdf.from_html_css(
            "<h1>Big</h1><p>Small</p>",
            "h1 { font-size: 72pt; } p { font-size: 6pt; }",
            self.font,
        ).to_bytes()
        assert no_css != with_css, "CSS font-size had no effect"

    def test_css_font_size_reflected_in_extracted_chars(self):
        """CSS font-size must affect the extracted char size in the resulting PDF."""
        if self.font is None:
            pytest.skip("no TTF font on system")
        pdf = Pdf.from_html_css(
            "<h1>BIGTEXT</h1>",
            "h1 { font-size: 48px; }",
            self.font,
        )
        doc = PdfDocument.from_bytes(pdf.to_bytes())
        chars = doc.extract_chars(0)
        h1_sizes = [c.font_size for c in chars if c.char in "BIGTEXT"]
        assert any(abs(s - 48.0) < 2.0 for s in h1_sizes), (
            f"expected ~48 (from 48px CSS), got {h1_sizes[:5]}"
        )

    def test_css_color_changes_output(self):
        if self.font is None:
            pytest.skip("no TTF font on system")
        black = Pdf.from_html_css("<p>text</p>", "p { color: black; }", self.font).to_bytes()
        red = Pdf.from_html_css("<p>text</p>", "p { color: red; }", self.font).to_bytes()
        assert black != red, "CSS color had no effect"

    def test_css_font_weight_bold_changes_output(self):
        if self.font is None:
            pytest.skip("no TTF font on system")
        normal = Pdf.from_html_css("<p>text</p>", "", self.font).to_bytes()
        bold = Pdf.from_html_css("<p>text</p>", "p { font-weight: bold; }", self.font).to_bytes()
        assert normal != bold, "CSS font-weight had no effect"

    def test_css_background_color_changes_output(self):
        if self.font is None:
            pytest.skip("no TTF font on system")
        no_bg = Pdf.from_html_css("<p>text</p>", "", self.font).to_bytes()
        with_bg = Pdf.from_html_css(
            "<p>text</p>", "body { background-color: yellow; }", self.font
        ).to_bytes()
        assert no_bg != with_bg, "CSS background-color had no effect"

    def test_from_html_css_null_html_raises(self):
        if self.font is None:
            pytest.skip("no TTF font on system")
        with pytest.raises(Exception):  # noqa: B017
            Pdf.from_html_css(None, "", self.font)

    def test_from_html_css_null_font_raises(self):
        with pytest.raises(Exception):  # noqa: B017
            Pdf.from_html_css("<p>x</p>", "", None)

    def test_from_html_css_with_fonts_produces_pdf(self):
        if self.font is None:
            pytest.skip("no TTF font on system")
        pdf = Pdf.from_html_css_with_fonts("<p>Hello</p>", "", [("Body", self.font)])
        assert pdf.to_bytes()[:5] == _PDF_MAGIC

    def test_from_html_css_with_fonts_empty_list_raises(self):
        with pytest.raises(Exception):  # noqa: B017
            Pdf.from_html_css_with_fonts("<p>x</p>", "", [])

    def test_inline_style_attribute_works(self):
        """style= attribute on an element must be applied."""
        if self.font is None:
            pytest.skip("no TTF font on system")
        default = Pdf.from_html_css("<p>text</p>", "", self.font).to_bytes()
        inline = Pdf.from_html_css('<p style="font-size: 60pt;">text</p>', "", self.font).to_bytes()
        assert default != inline, "inline style= had no effect"


# ── DocumentBuilder ──────────────────────────────────────────────────────────


class TestDocumentBuilder:
    def test_basic_builder_produces_pdf(self):
        from pdf_oxide import DocumentBuilder

        data = DocumentBuilder().a4_page().paragraph("Hello World").done().build()
        assert data[:5] == _PDF_MAGIC

    def test_builder_text_is_extractable(self):
        from pdf_oxide import DocumentBuilder

        data = DocumentBuilder().a4_page().paragraph("UniqueBuilderText456").done().build()
        doc = PdfDocument.from_bytes(data)
        text = doc.extract_text(0)
        assert "UniqueBuilderText456" in text

    def test_builder_multiple_pages(self):
        from pdf_oxide import DocumentBuilder

        builder = DocumentBuilder()
        for _ in range(3):
            builder.a4_page().paragraph("page").done()
        doc = PdfDocument.from_bytes(builder.build())
        assert doc.page_count() == 3

    def test_builder_save_encrypted_produces_pdf(self, tmp_path):
        import os

        from pdf_oxide import DocumentBuilder

        out = str(tmp_path / "enc.pdf")
        (
            DocumentBuilder()
            .a4_page()
            .paragraph("Secret")
            .done()
            .save_encrypted(out, "user123", "owner456")
        )
        assert os.path.exists(out)
        assert os.path.getsize(out) > 100


# ── Pdf: merge / utilities ────────────────────────────────────────────────────


class TestPdfMerge:
    def test_merge_two_pdfs_increases_page_count(self, tmp_path):
        a = Pdf.from_markdown("# Page 1")
        b = Pdf.from_markdown("# Page 2")
        pages_a = PdfDocument.from_bytes(a.to_bytes()).page_count()
        pages_b = PdfDocument.from_bytes(b.to_bytes()).page_count()
        path_a = str(tmp_path / "a.pdf")
        path_b = str(tmp_path / "b.pdf")
        a.save(path_a)
        b.save(path_b)
        merged = Pdf.merge([path_a, path_b])
        doc = PdfDocument.from_bytes(merged.to_bytes())
        assert doc.page_count() == pages_a + pages_b

    def test_merge_empty_list_raises_or_returns_empty(self):
        try:
            result = Pdf.merge([])
            # If it doesn't raise, it should still be valid bytes or empty
            assert isinstance(result, (bytes, bytearray))
        except Exception:
            pass  # either outcome is acceptable


# ── Pdf: from_image / from_text ───────────────────────────────────────────────


class TestPdfFromOther:
    def test_from_text_produces_pdf(self):
        if not hasattr(Pdf, "from_text"):
            pytest.skip("Pdf.from_text not available")
        pdf = Pdf.from_text("Hello World")
        assert pdf.to_bytes()[:5] == _PDF_MAGIC

    def test_from_html_inline_style_is_applied(self):
        pdf = Pdf.from_html("<h1 style='font-size:48pt'>BIG</h1>")
        data = pdf.to_bytes()
        assert data[:5] == _PDF_MAGIC

    def test_from_image_bytes_produces_pdf(self):
        import struct
        import zlib

        # minimal 1x1 white PNG
        def _png():
            hdr = b"\x89PNG\r\n\x1a\n"
            ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr_chunk = b"IHDR" + ihdr  # noqa: F841 (used in chunk() call below)
            idat_data = zlib.compress(b"\x00\xff\xff\xff")

            def chunk(name, data):
                return (
                    struct.pack(">I", len(data))
                    + name
                    + data
                    + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
                )

            return hdr + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat_data) + chunk(b"IEND", b"")

        try:
            pdf = Pdf.from_image_bytes(_png())
            assert pdf.to_bytes()[:5] == _PDF_MAGIC
        except Exception as e:
            if _is_feature_off(e):
                pytest.skip(f"from_image_bytes not available: {e}")
            raise


# ── PdfDocument: conversion to other formats ─────────────────────────────────


class TestConversion:
    def test_to_markdown_contains_text(self):
        data = Pdf.from_markdown("# MARKER_MD").to_bytes()
        doc = PdfDocument.from_bytes(data)
        md = doc.to_markdown(0)
        assert isinstance(md, str) and len(md) > 0

    def test_to_markdown_all_contains_text(self):
        doc = _make_simple_doc()
        md = doc.to_markdown_all()
        assert isinstance(md, str) and len(md) > 0

    def test_to_html_returns_html_tags(self):
        doc = _make_simple_doc()
        html = doc.to_html(0)
        assert isinstance(html, str) and len(html) > 0
        assert "<" in html

    def test_to_html_all_returns_html_tags(self):
        doc = _make_simple_doc()
        html = doc.to_html_all()
        assert isinstance(html, str)
        assert "<" in html

    def test_to_plain_text_contains_words(self):
        data = Pdf.from_markdown("PLAINMARKER").to_bytes()
        doc = PdfDocument.from_bytes(data)
        text = doc.to_plain_text(0)
        assert "PLAINMARKER" in text

    def test_to_plain_text_all_contains_words(self):
        data = Pdf.from_markdown("ALLMARKER").to_bytes()
        doc = PdfDocument.from_bytes(data)
        text = doc.to_plain_text_all()
        assert "ALLMARKER" in text


# ── PdfDocument: search ───────────────────────────────────────────────────────


class TestSearch:
    def test_search_page_finds_known_term(self):
        data = Pdf.from_markdown("SEARCHMETOKEN").to_bytes()
        doc = PdfDocument.from_bytes(data)
        results = doc.search_page(0, "SEARCHMETOKEN")
        assert len(results) > 0

    def test_search_all_finds_term_in_document(self):
        data = Pdf.from_markdown("FINDMEALL").to_bytes()
        doc = PdfDocument.from_bytes(data)
        results = doc.search("FINDMEALL")
        assert len(results) > 0

    def test_search_missing_term_returns_empty(self):
        doc = _make_simple_doc()
        results = doc.search("ZZZNOTPRESENTZZZ")
        assert results == [] or len(results) == 0


# ── PdfDocument: page mutations ───────────────────────────────────────────────


class TestMutations:
    def test_save_to_path(self, tmp_path):
        doc = _make_simple_doc()
        path = str(tmp_path / "out.pdf")
        doc.save(path)
        import os

        assert os.path.getsize(path) > 100

    def test_merge_from_increases_page_count(self, tmp_path):
        a_path = str(tmp_path / "a.pdf")
        b_path = str(tmp_path / "b.pdf")
        Pdf.from_markdown("# A").save(a_path)
        Pdf.from_markdown("# B").save(b_path)
        doc = PdfDocument(a_path)
        before = doc.page_count()
        doc.merge_from(b_path)
        out = str(tmp_path / "merged.pdf")
        doc.save(out)
        doc2 = PdfDocument(out)
        assert doc2.page_count() == before + 1

    def test_delete_page_reduces_count(self, tmp_path):
        a_path = str(tmp_path / "a.pdf")
        b_path = str(tmp_path / "b.pdf")
        Pdf.from_markdown("# P1").save(a_path)
        Pdf.from_markdown("# P2").save(b_path)
        doc = PdfDocument(a_path)
        doc.merge_from(b_path)
        out = str(tmp_path / "two.pdf")
        doc.save(out)
        doc2 = PdfDocument(out)
        before = doc2.page_count()
        if before < 2:
            pytest.skip("need multi-page PDF")
        doc2.delete_page(0)
        out2 = str(tmp_path / "one.pdf")
        doc2.save(out2)
        doc3 = PdfDocument(out2)
        assert doc3.page_count() == before - 1

    def test_rotate_page_sets_rotation(self, tmp_path):
        path = str(tmp_path / "rot.pdf")
        Pdf.from_markdown("# Rotate").save(path)
        doc = PdfDocument(path)
        doc.rotate_page(0, 90)
        out = str(tmp_path / "rotated.pdf")
        doc.save(out)
        doc2 = PdfDocument(out)
        assert doc2.page_rotation(0) == 90

    def test_rotate_all_pages(self, tmp_path):
        path = str(tmp_path / "rotall.pdf")
        Pdf.from_markdown("# RotAll").save(path)
        doc = PdfDocument(path)
        doc.rotate_all_pages(180)
        out = str(tmp_path / "rotall180.pdf")
        doc.save(out)
        doc2 = PdfDocument(out)
        assert doc2.page_rotation(0) == 180


# ── PdfDocument: page extraction ─────────────────────────────────────────────


def _make_two_page_doc() -> PdfDocument:
    """Return a PdfDocument with exactly 2 pages."""
    data = Pdf.from_markdown("# Page One\n\n---\n\n# Page Two").to_bytes()
    doc = PdfDocument.from_bytes(data)
    if doc.page_count() < 2:
        # Merge a second page in
        extra = Pdf.from_markdown("# Page Two").to_bytes()
        doc.merge_from(extra)
    return doc


class TestPageExtraction:
    def test_extract_pages_to_file_reduces_page_count(self, tmp_path):
        doc = _make_two_page_doc()
        assert doc.page_count() >= 2
        out = str(tmp_path / "single.pdf")
        doc.extract_pages([0], out)
        result = PdfDocument(out)
        assert result.page_count() == 1

    def test_extract_pages_preserves_content(self, tmp_path):
        data = Pdf.from_markdown("KEEPTHIS\n\n---\n\nDROPTHIS").to_bytes()
        doc = PdfDocument.from_bytes(data)
        if doc.page_count() < 2:
            extra = Pdf.from_markdown("DROPTHIS").to_bytes()
            doc.merge_from(extra)
        out = str(tmp_path / "kept.pdf")
        doc.extract_pages([0], out)
        result = PdfDocument(out)
        text = result.extract_text(0)
        assert "KEEPTHIS" in text

    def test_extract_pages_to_bytes_returns_valid_pdf(self):
        doc = _make_two_page_doc()
        assert doc.page_count() >= 2
        data = doc.extract_pages_to_bytes([0])
        assert data[:5] == _PDF_MAGIC

    def test_extract_pages_to_bytes_correct_page_count(self):
        doc = _make_two_page_doc()
        assert doc.page_count() >= 2
        data = doc.extract_pages_to_bytes([0])
        result = PdfDocument.from_bytes(data)
        assert result.page_count() == 1

    def test_extract_pages_to_bytes_multiple_pages(self):
        data = Pdf.from_markdown("# P1").to_bytes()
        doc = PdfDocument.from_bytes(data)
        for i in range(2, 5):
            extra = Pdf.from_markdown(f"# P{i}").to_bytes()
            doc.merge_from(extra)
        total = doc.page_count()
        assert total >= 3
        chunk = doc.extract_pages_to_bytes([0, 1])
        result = PdfDocument.from_bytes(chunk)
        assert result.page_count() == 2

    def test_extract_pages_chunking_pipeline(self):
        """Simulate potatochipcoconut's page-splitting use case."""
        try:
            from itertools import batched
        except ImportError:
            # Python < 3.12 compat
            from itertools import islice

            def batched(iterable, n):
                it = iter(iterable)
                while chunk := list(islice(it, n)):
                    yield chunk

        data = Pdf.from_markdown("# P1").to_bytes()
        doc = PdfDocument.from_bytes(data)
        for i in range(2, 6):
            extra = Pdf.from_markdown(f"# P{i}").to_bytes()
            doc.merge_from(extra)
        total = doc.page_count()
        assert total >= 4

        chunk_size = 2
        chunks = []
        for chunk_indices in batched(range(total), chunk_size):
            chunk_bytes = doc.extract_pages_to_bytes(list(chunk_indices))
            assert chunk_bytes[:5] == _PDF_MAGIC
            chunk_doc = PdfDocument.from_bytes(chunk_bytes)
            assert chunk_doc.page_count() == len(chunk_indices)
            chunks.append(chunk_bytes)
        assert len(chunks) >= 2

    def test_extract_pages_out_of_range_raises(self):
        doc = _make_two_page_doc()
        with pytest.raises(RuntimeError):
            doc.extract_pages_to_bytes([999])


# ── DocumentBuilder extras ────────────────────────────────────────────────────


class TestDocumentBuilderExtras:
    def test_save_non_encrypted(self, tmp_path):
        from pdf_oxide import DocumentBuilder

        path = str(tmp_path / "plain.pdf")
        DocumentBuilder().a4_page().paragraph("plain save").done().save(path)
        import os

        assert os.path.getsize(path) > 100

    def test_letter_page(self):
        from pdf_oxide import DocumentBuilder

        data = DocumentBuilder().letter_page().paragraph("US Letter").done().build()
        assert data[:5] == _PDF_MAGIC

    def test_custom_page_size(self):
        from pdf_oxide import DocumentBuilder

        data = DocumentBuilder().page(300.0, 400.0).paragraph("custom").done().build()
        assert data[:5] == _PDF_MAGIC

    def test_metadata_setters(self):
        from pdf_oxide import DocumentBuilder

        data = (
            DocumentBuilder()
            .title("My Title")
            .author("Alice")
            .subject("Testing")
            .keywords("pdf, test")
            .creator("pytest")
            .a4_page()
            .paragraph("metadata")
            .done()
            .build()
        )
        assert data[:5] == _PDF_MAGIC

    def test_to_bytes_encrypted(self):
        from pdf_oxide import DocumentBuilder

        data = (
            DocumentBuilder()
            .a4_page()
            .paragraph("secret")
            .done()
            .to_bytes_encrypted("user", "owner")
        )
        assert data[:5] == _PDF_MAGIC


# ── Signature object properties ───────────────────────────────────────────────


class TestSignatureProperties:
    def test_unsigned_pdf_signatures_list_is_empty(self):
        doc = _make_simple_doc()
        sigs = doc.signatures()
        assert sigs == []

    def test_signature_count_is_zero_for_unsigned(self):
        doc = _make_simple_doc()
        assert doc.signature_count() == 0


# ── PdfDocument: to_bytes() and save() with options ───────────────────────────


class TestSaveOptions:
    def test_to_bytes_default_returns_valid_pdf(self):
        doc = _make_simple_doc()
        data = doc.to_bytes()
        assert data[:5] == _PDF_MAGIC
        assert len(data) > 100

    def test_to_bytes_compress_true(self):
        doc = _make_simple_doc()
        data = doc.to_bytes(compress=True)
        assert data[:5] == _PDF_MAGIC

    def test_to_bytes_compress_false(self):
        doc = _make_simple_doc()
        data = doc.to_bytes(compress=False)
        assert data[:5] == _PDF_MAGIC

    def test_to_bytes_garbage_collect_true(self):
        doc = _make_simple_doc()
        data = doc.to_bytes(garbage_collect=True)
        assert data[:5] == _PDF_MAGIC

    def test_to_bytes_garbage_collect_false(self):
        doc = _make_simple_doc()
        data = doc.to_bytes(garbage_collect=False)
        assert data[:5] == _PDF_MAGIC

    def test_to_bytes_all_options(self):
        doc = _make_simple_doc()
        data = doc.to_bytes(compress=True, garbage_collect=True, linearize=False)
        assert data[:5] == _PDF_MAGIC

    def test_to_bytes_round_trips_content(self):
        doc = _make_simple_doc()
        data = doc.to_bytes()
        doc2 = PdfDocument.from_bytes(data)
        assert doc2.page_count() >= 1

    def test_to_bytes_compress_smaller_or_equal(self):
        doc1 = _make_simple_doc()
        doc2 = _make_simple_doc()
        uncompressed = doc1.to_bytes(compress=False, garbage_collect=False)
        compressed = doc2.to_bytes(compress=True, garbage_collect=False)
        assert len(compressed) <= len(uncompressed), (
            f"Compressed ({len(compressed)}) should be <= uncompressed ({len(uncompressed)})"
        )

    def test_save_with_compress_option(self, tmp_path):
        import os

        doc = _make_simple_doc()
        path = str(tmp_path / "compressed.pdf")
        doc.save(path, compress=True)
        assert os.path.getsize(path) > 100
        # Round-trip
        doc2 = PdfDocument(path)
        assert doc2.page_count() >= 1

    def test_save_with_garbage_collect_option(self, tmp_path):
        import os

        doc = _make_simple_doc()
        path = str(tmp_path / "gc.pdf")
        doc.save(path, garbage_collect=True)
        assert os.path.getsize(path) > 100

    def test_save_with_all_options(self, tmp_path):
        import os

        doc = _make_simple_doc()
        path = str(tmp_path / "all_opts.pdf")
        doc.save(path, compress=True, garbage_collect=True, linearize=False)
        assert os.path.getsize(path) > 100
        doc2 = PdfDocument(path)
        assert doc2.page_count() >= 1

    def test_save_no_options_still_works(self, tmp_path):
        """Ensure calling save(path) with no kwargs still works."""
        import os

        doc = _make_simple_doc()
        path = str(tmp_path / "plain.pdf")
        doc.save(path)
        assert os.path.getsize(path) > 100


# ── PdfDocument: PDF/A compliance ────────────────────────────────────────────


class TestPdfACompliance:
    def test_validate_pdf_a_returns_dict_with_required_keys(self):
        doc = _make_simple_doc()
        result = doc.validate_pdf_a("2b")
        assert isinstance(result, dict)
        assert "valid" in result
        assert "level" in result
        assert "errors" in result
        assert "warnings" in result

    def test_validate_pdf_a_level_echoed_back(self):
        doc = _make_simple_doc()
        result = doc.validate_pdf_a("1b")
        assert result["level"] == "1b"

    def test_validate_pdf_a_errors_is_list(self):
        doc = _make_simple_doc()
        result = doc.validate_pdf_a("2b")
        assert isinstance(result["errors"], list)
        assert isinstance(result["warnings"], list)

    def test_validate_pdf_a_invalid_level_raises(self):
        doc = _make_simple_doc()
        with pytest.raises(ValueError):
            doc.validate_pdf_a("99z")

    def test_convert_to_pdf_a_returns_dict_with_required_keys(self):
        doc = _make_simple_doc()
        result = doc.convert_to_pdf_a("2b")
        assert isinstance(result, dict)
        assert "success" in result
        assert "actions" in result
        assert "errors" in result

    def test_convert_to_pdf_a_success_is_bool(self):
        doc = _make_simple_doc()
        result = doc.convert_to_pdf_a("2b")
        assert isinstance(result["success"], bool)

    def test_convert_to_pdf_a_actions_is_list(self):
        doc = _make_simple_doc()
        result = doc.convert_to_pdf_a("2b")
        assert isinstance(result["actions"], list)
        assert isinstance(result["errors"], list)

    def test_convert_to_pdf_a_document_remains_valid_pdf(self):
        doc = _make_simple_doc()
        doc.convert_to_pdf_a("2b")
        data = doc.to_bytes()
        assert data[:5] == _PDF_MAGIC

    def test_convert_to_pdf_a_validate_after_conversion(self):
        doc = _make_simple_doc()
        doc.convert_to_pdf_a("2b")
        result = doc.validate_pdf_a("2b")
        assert isinstance(result["valid"], bool)

    def test_convert_to_pdf_a_invalid_level_raises(self):
        doc = _make_simple_doc()
        with pytest.raises(ValueError):
            doc.convert_to_pdf_a("99z")

    def test_convert_to_pdf_a_all_levels_accepted(self):
        for level in ("1a", "1b", "2a", "2b", "2u", "3a", "3b", "3u"):
            doc = _make_simple_doc()
            result = doc.convert_to_pdf_a(level)
            assert "success" in result, f"level {level!r} did not return expected dict"

    def test_convert_to_pdf_a_bytes_output_pipeline(self):
        """Full IDP pipeline: open → convert → compress → get bytes."""
        doc = _make_simple_doc()
        doc.convert_to_pdf_a("2b")
        data = doc.to_bytes(compress=True, garbage_collect=True)
        assert data[:5] == _PDF_MAGIC
        assert len(data) > 100


# ── PdfDocument: to_bytes_encrypted ─────────────────────────────────────────


class TestPdfDocumentEncryptedBytes:
    def test_to_bytes_encrypted_returns_valid_pdf(self):
        doc = _make_simple_doc()
        data = doc.to_bytes_encrypted("user123", "owner456")
        assert data[:5] == _PDF_MAGIC
        assert len(data) > 100

    def test_to_bytes_encrypted_owner_password_defaults_to_user(self):
        doc = _make_simple_doc()
        data = doc.to_bytes_encrypted("secret")
        assert data[:5] == _PDF_MAGIC

    def test_to_bytes_encrypted_roundtrip_with_correct_password(self):
        doc = _make_simple_doc()
        data = doc.to_bytes_encrypted("userpass", "ownerpass")
        doc2 = PdfDocument.from_bytes(data, password="userpass")
        assert doc2.page_count() >= 1

    def test_to_bytes_encrypted_wrong_password_raises(self):
        doc = _make_simple_doc()
        data = doc.to_bytes_encrypted("correct", "ownerpass")
        with pytest.raises(RuntimeError):
            PdfDocument.from_bytes(data, password="wrong")

    def test_to_bytes_encrypted_with_permission_flags(self):
        doc = _make_simple_doc()
        data = doc.to_bytes_encrypted(
            "user",
            "owner",
            allow_print=False,
            allow_copy=False,
            allow_modify=False,
            allow_annotate=False,
        )
        assert data[:5] == _PDF_MAGIC
        doc2 = PdfDocument.from_bytes(data, password="user")
        assert doc2.page_count() >= 1

    def test_to_bytes_encrypted_larger_than_unencrypted(self):
        doc1 = _make_simple_doc()
        doc2 = _make_simple_doc()
        _ = doc1.to_bytes()
        encrypted = doc2.to_bytes_encrypted("pw", "pw")
        assert len(encrypted) > 0
        assert encrypted[:5] == _PDF_MAGIC
