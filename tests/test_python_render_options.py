"""
Tests for the expanded PdfDocument.render_page options (#384 Python gap O).

The Rust core's RenderOptions (src/rendering/page_renderer.rs:41) exposes
dpi, format, background (RGBA), render_annotations, and jpeg_quality. Python
used to see only dpi + format; this suite locks in the full surface.
"""

import pytest

from pdf_oxide import Pdf, PdfDocument


@pytest.fixture()
def one_page_doc():
    bytes_ = Pdf.from_markdown("# Page\n\nRender me.").to_bytes()
    return PdfDocument.from_bytes(bytes_)


def _is_png(buf: bytes) -> bool:
    return buf.startswith(b"\x89PNG\r\n\x1a\n")


def _is_jpeg(buf: bytes) -> bool:
    return len(buf) >= 3 and buf[0] == 0xFF and buf[1] == 0xD8 and buf[2] == 0xFF


def test_render_default_is_png(one_page_doc):
    png = one_page_doc.render_page(0)
    assert _is_png(png)


def test_render_explicit_jpeg_has_jpeg_magic(one_page_doc):
    jpeg = one_page_doc.render_page(0, format="jpeg")
    assert _is_jpeg(jpeg)


def test_render_jpeg_quality_produces_smaller_bytes(one_page_doc):
    low = one_page_doc.render_page(0, format="jpeg", jpeg_quality=20)
    high = one_page_doc.render_page(0, format="jpeg", jpeg_quality=95)
    assert _is_jpeg(low) and _is_jpeg(high)
    assert len(low) <= len(high), "lower JPEG quality must not yield a larger file"


def test_render_high_dpi_produces_bigger_image(one_page_doc):
    small = one_page_doc.render_page(0, dpi=72)
    large = one_page_doc.render_page(0, dpi=300)
    assert _is_png(small) and _is_png(large)
    assert len(large) > len(small)


def test_render_no_annotations_accepts_flag(one_page_doc):
    png = one_page_doc.render_page(0, render_annotations=False)
    assert _is_png(png)


def test_render_background_rgba_accepts_tuple(one_page_doc):
    png = one_page_doc.render_page(0, background=(0.0, 0.0, 0.0, 1.0))
    assert _is_png(png)


def test_render_transparent_flag_accepts(one_page_doc):
    png = one_page_doc.render_page(0, transparent=True)
    assert _is_png(png)


def test_render_rejects_invalid_jpeg_quality(one_page_doc):
    with pytest.raises((ValueError, RuntimeError)):
        one_page_doc.render_page(0, format="jpeg", jpeg_quality=0)


def test_render_rejects_bad_background_tuple(one_page_doc):
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        one_page_doc.render_page(0, background=(1.0, 1.0))  # not 4 channels


# ── Raw RGBA pixel buffer tests (issue #446) ─────────────────────────────────


def test_render_pixmap_size(one_page_doc):
    px = one_page_doc.render_pixmap(0, dpi=72)
    assert px.width > 0
    assert px.height > 0
    assert len(px.data) == px.width * px.height * 4


def test_render_pixmap_not_png(one_page_doc):
    px = one_page_doc.render_pixmap(0)
    assert px.data[:4] != b"\x89PNG", "render_pixmap should return raw pixels, not PNG"


def test_render_pixmap_dimensions_match_png(one_page_doc):
    import struct

    png = one_page_doc.render_page(0, dpi=72)
    px = one_page_doc.render_pixmap(0, dpi=72)
    # PNG IHDR: width at bytes 16-19, height at 20-23 (big-endian)
    (png_w,) = struct.unpack(">I", png[16:20])
    (png_h,) = struct.unpack(">I", png[20:24])
    assert px.width == png_w, f"width mismatch: pixmap {px.width} vs PNG {png_w}"
    assert px.height == png_h, f"height mismatch: pixmap {px.height} vs PNG {png_h}"


def test_render_pixmap_returns_rendered_pixmap_type(one_page_doc):
    from pdf_oxide import RenderedPixmap

    px = one_page_doc.render_pixmap(0)
    assert isinstance(px, RenderedPixmap)
    assert hasattr(px, "data") and hasattr(px, "width") and hasattr(px, "height")
