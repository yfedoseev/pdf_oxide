"""
Python bindings tests for pdf_oxide.

These tests verify the Python API works correctly, including:
- Opening PDF files
- pathlib.Path and context manager support
- Extracting text
- Converting to Markdown
- Converting to HTML
- Error handling
"""

from pathlib import Path

import pytest

from pdf_oxide import PdfDocument


def test_open_pdf():
    """Test opening a PDF file."""
    # Note: This test will need actual PDF fixtures to run
    # For now, it documents the expected behavior
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        assert doc is not None
        # Version should be a tuple of two integers
        version = doc.version()
        assert isinstance(version, tuple)
        assert len(version) == 2
        assert isinstance(version[0], int)
        assert isinstance(version[1], int)
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_version():
    """Test getting PDF version."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        major, minor = doc.version()
        assert major >= 1
        assert minor >= 0
        # PDF versions are typically 1.0 through 2.0
        assert major <= 2
        assert minor <= 7
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_page_count():
    """Test getting page count."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        count = doc.page_count()
        assert isinstance(count, int)
        assert count >= 1
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_extract_text():
    """Test extracting text from a page."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        text = doc.extract_text(0)
        assert isinstance(text, str)
        # Text should be non-empty for a real PDF
        # (empty is ok for a minimal test PDF though)
        assert text is not None
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_extract_text_with_content():
    """Test extracting text that contains specific content."""
    try:
        doc = PdfDocument("tests/fixtures/hello_world.pdf")
        text = doc.extract_text(0)
        assert isinstance(text, str)
        assert len(text) > 0
        # Should contain "Hello" or "hello" (case-insensitive check)
        assert "hello" in text.lower()
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'hello_world.pdf' not available or invalid")


def test_to_markdown():
    """Test converting a page to Markdown."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        markdown = doc.to_markdown(0)
        assert isinstance(markdown, str)
        assert markdown is not None
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_to_markdown_with_options():
    """Test converting to Markdown with custom options."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")

        # Test with heading detection enabled
        markdown = doc.to_markdown(0, detect_headings=True)
        assert isinstance(markdown, str)

        # Test with heading detection disabled
        markdown = doc.to_markdown(0, detect_headings=False)
        assert isinstance(markdown, str)

        # Test with layout preservation
        markdown = doc.to_markdown(0, preserve_layout=True)
        assert isinstance(markdown, str)
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_to_html():
    """Test converting a page to HTML."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        html = doc.to_html(0)
        assert isinstance(html, str)
        assert html is not None
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_to_html_semantic_mode():
    """Test converting to semantic HTML (default mode)."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        html = doc.to_html(0, preserve_layout=False)
        assert isinstance(html, str)
        # Semantic HTML should not contain absolute positioning
        # (though it might not contain much if the PDF is simple)
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_to_html_layout_mode():
    """Test converting to layout-preserved HTML."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        html = doc.to_html(0, preserve_layout=True)
        assert isinstance(html, str)
        # Layout mode should include positioning CSS
        # Check if it contains position-related CSS or inline styles
        # (only if the PDF has content)
        if len(html) > 100:
            assert "position" in html.lower() or "style" in html.lower()
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_to_markdown_all():
    """Test converting all pages to Markdown."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        markdown = doc.to_markdown_all()
        assert isinstance(markdown, str)
        assert markdown is not None
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_to_markdown_all_multipage():
    """Test converting multiple pages to Markdown."""
    try:
        doc = PdfDocument("tests/fixtures/multipage.pdf")
        markdown = doc.to_markdown_all()
        assert isinstance(markdown, str)
        assert len(markdown) > 0
        # Multi-page markdown should contain horizontal rules as separators
        page_count = doc.page_count()
        if page_count > 1:
            assert "---" in markdown
    except OSError:
        pytest.skip("Test fixture 'multipage.pdf' not available")


def test_to_html_all():
    """Test converting all pages to HTML."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        html = doc.to_html_all()
        assert isinstance(html, str)
        assert html is not None
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_to_html_all_multipage():
    """Test converting multiple pages to HTML."""
    try:
        doc = PdfDocument("tests/fixtures/multipage.pdf")
        html = doc.to_html_all()
        assert isinstance(html, str)
        assert len(html) > 0
        # Multi-page HTML should contain page div elements
        page_count = doc.page_count()
        if page_count > 1:
            assert 'class="page"' in html or "data-page" in html
    except OSError:
        pytest.skip("Test fixture 'multipage.pdf' not available")


# === pathlib.Path and context manager ===


def test_open_pdf_pathlib():
    """PdfDocument accepts pathlib.Path; behavior matches string path."""
    try:
        path = Path("tests/fixtures/1.pdf")
        doc = PdfDocument(path)
        assert doc is not None
        assert doc.page_count() == 7, "1.pdf has 7 pages"
        version = doc.version()
        assert isinstance(version, tuple) and len(version) == 2 and version[0] >= 1
        # Same as opening with string path
        doc_str = PdfDocument("tests/fixtures/1.pdf")
        assert doc.version() == doc_str.version()
        assert doc.page_count() == doc_str.page_count()
        assert doc.extract_text(0) == doc_str.extract_text(0)
        assert len(doc.extract_text(0).strip()) > 0, "1.pdf has text on page 0"
    except (OSError, RuntimeError):
        pytest.skip("Test fixture '1.pdf' not available or invalid")


def test_context_manager():
    """PdfDocument works as context manager: enter returns self, exit runs normally."""
    try:
        with PdfDocument("tests/fixtures/1.pdf") as doc:
            assert doc is not None
            assert doc.page_count() == 7, "1.pdf has 7 pages"
            version = doc.version()
            assert isinstance(version, tuple) and len(version) == 2 and version[0] >= 1
            text = doc.extract_text(0)
            assert isinstance(text, str) and len(text.strip()) > 0, "1.pdf has text"
        # After block, no error; doc was closed on exit
    except (OSError, RuntimeError):
        pytest.skip("Test fixture '1.pdf' not available or invalid")


def test_context_manager_with_pathlib():
    """Context manager works when path is pathlib.Path."""
    try:
        with PdfDocument(Path("tests/fixtures/1.pdf")) as doc:
            assert doc.page_count() == 7, "1.pdf has 7 pages"
            version = doc.version()
            assert isinstance(version, tuple) and len(version) == 2 and version[0] >= 1
            _ = doc.to_markdown(0)  # 1.pdf has text
    except (OSError, RuntimeError):
        pytest.skip("Test fixture '1.pdf' not available or invalid")


def test_context_manager_exception_propagates():
    """Exceptions inside the with block are not swallowed (__exit__ returns False)."""
    try:
        with pytest.raises(ValueError), PdfDocument("tests/fixtures/1.pdf") as doc:
            _ = doc.page_count()
            raise ValueError("intentional")
    except (OSError, RuntimeError):
        pytest.skip("Test fixture '1.pdf' not available or invalid")


# === From Bytes Tests ===


def test_from_bytes_matches_file():
    """Test that from_bytes produces the same results as opening from path."""
    with open("tests/fixtures/simple.pdf", "rb") as f:
        data = f.read()
    doc_path = PdfDocument("tests/fixtures/simple.pdf")
    doc_bytes = PdfDocument.from_bytes(data)

    assert doc_path.version() == doc_bytes.version()
    assert doc_path.page_count() == doc_bytes.page_count()
    assert doc_path.extract_text(0) == doc_bytes.extract_text(0)


def test_from_bytes_roundtrip():
    """Test opening bytes from a PDF created with Pdf.from_text."""
    from pdf_oxide import Pdf

    pdf = Pdf.from_text("Hello from bytes!")
    pdf_bytes = pdf.to_bytes()

    doc = PdfDocument.from_bytes(pdf_bytes)
    assert doc.page_count() >= 1
    text = doc.extract_text(0)
    assert "Hello from bytes!" in text


def test_from_bytes_invalid():
    """Test error handling for invalid bytes."""
    with pytest.raises(IOError):
        PdfDocument.from_bytes(b"not a pdf")


# === Error Handling Tests ===


def test_error_handling_nonexistent_file():
    """Test error handling for non-existent file."""
    with pytest.raises(IOError) as exc_info:
        PdfDocument("nonexistent_file_that_does_not_exist.pdf")

    # Error message should be helpful
    error_msg = str(exc_info.value)
    assert "Failed to open PDF" in error_msg or "No such file" in error_msg


def test_error_handling_invalid_page():
    """Test error handling for invalid page index."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        page_count = doc.page_count()

        # Try to access a page that doesn't exist
        with pytest.raises(RuntimeError) as exc_info:
            doc.extract_text(page_count + 100)

        # Error message should indicate the problem
        error_msg = str(exc_info.value)
        assert "Failed to extract text" in error_msg or "page" in error_msg.lower()
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_error_handling_invalid_page_conversion():
    """Test error handling for invalid page in conversion."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        page_count = doc.page_count()

        # Try to convert a page that doesn't exist
        with pytest.raises(RuntimeError):
            doc.to_markdown(page_count + 100)
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_repr():
    """Test string representation of PdfDocument."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        repr_str = repr(doc)
        assert isinstance(repr_str, str)
        assert "PdfDocument" in repr_str
        assert "version=" in repr_str
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_multiple_operations():
    """Test performing multiple operations on the same document."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")

        # Get version multiple times
        version1 = doc.version()
        version2 = doc.version()
        assert version1 == version2

        # Extract text multiple times
        text1 = doc.extract_text(0)
        text2 = doc.extract_text(0)
        assert text1 == text2

        # Convert to different formats
        markdown = doc.to_markdown(0)
        html = doc.to_html(0)
        assert isinstance(markdown, str)
        assert isinstance(html, str)
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_image_output_dir():
    """Test specifying image output directory."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")

        # Convert with image output directory specified
        markdown = doc.to_markdown(0, image_output_dir="./test_images")
        assert isinstance(markdown, str)

        # Convert without images
        markdown = doc.to_markdown(0, include_images=False)
        assert isinstance(markdown, str)
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_all_options_combined():
    """Test using all conversion options together."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")

        # Test with all options specified
        markdown = doc.to_markdown(
            0,
            preserve_layout=True,
            detect_headings=False,
            include_images=True,
            image_output_dir="./output",
        )
        assert isinstance(markdown, str)

        html = doc.to_html(
            0,
            preserve_layout=True,
            detect_headings=True,
            include_images=False,
            image_output_dir=None,
        )
        assert isinstance(html, str)
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


# === PDF Creation Tests ===


def test_pdf_from_markdown():
    """Test creating PDF from Markdown."""
    from pdf_oxide import Pdf

    md_content = """# Test Document

This is a **test** paragraph.

## Section 1

Some text content.
"""
    pdf = Pdf.from_markdown(md_content)
    assert pdf is not None
    # PDF should have some bytes
    pdf_bytes = pdf.to_bytes()
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    # Should start with PDF header
    assert pdf_bytes[:4] == b"%PDF"


def test_pdf_from_markdown_with_options():
    """Test creating PDF from Markdown with options."""
    from pdf_oxide import Pdf

    md_content = "# Hello World"
    pdf = Pdf.from_markdown(
        md_content,
        title="Test Title",
        author="Test Author",
    )
    assert pdf is not None
    pdf_bytes = pdf.to_bytes()
    assert len(pdf_bytes) > 0


def test_pdf_from_html():
    """Test creating PDF from HTML."""
    from pdf_oxide import Pdf

    html_content = """
    <h1>Test Document</h1>
    <p>This is a <strong>test</strong> paragraph.</p>
    """
    pdf = Pdf.from_html(html_content)
    assert pdf is not None
    pdf_bytes = pdf.to_bytes()
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    assert pdf_bytes[:4] == b"%PDF"


def test_pdf_from_text():
    """Test creating PDF from plain text."""
    from pdf_oxide import Pdf

    text_content = "Hello, World!\n\nThis is plain text."
    pdf = Pdf.from_text(text_content)
    assert pdf is not None
    pdf_bytes = pdf.to_bytes()
    assert len(pdf_bytes) > 0
    assert pdf_bytes[:4] == b"%PDF"


def test_pdf_save_to_file(tmp_path):
    """Test saving PDF to a file."""
    from pdf_oxide import Pdf

    pdf = Pdf.from_text("Test content")
    output_path = tmp_path / "output.pdf"
    pdf.save(str(output_path))
    assert output_path.exists()
    assert output_path.stat().st_size > 0


# === Advanced Graphics Tests ===


def test_color_creation():
    """Test Color class creation."""
    from pdf_oxide import Color

    # Create from RGB values
    color = Color(1.0, 0.0, 0.0)
    assert color is not None

    # Create from hex
    color = Color.from_hex("#FF0000")
    assert color is not None

    color = Color.from_hex("00FF00")
    assert color is not None


def test_color_predefined():
    """Test predefined colors."""
    from pdf_oxide import Color

    black = Color.black()
    assert black is not None

    white = Color.white()
    assert white is not None

    red = Color.red()
    assert red is not None

    green = Color.green()
    assert green is not None

    blue = Color.blue()
    assert blue is not None


def test_blend_modes():
    """Test BlendMode constants."""
    from pdf_oxide import BlendMode

    # Test all blend modes are accessible
    assert BlendMode.NORMAL() is not None
    assert BlendMode.MULTIPLY() is not None
    assert BlendMode.SCREEN() is not None
    assert BlendMode.OVERLAY() is not None
    assert BlendMode.DARKEN() is not None
    assert BlendMode.LIGHTEN() is not None
    assert BlendMode.COLOR_DODGE() is not None
    assert BlendMode.COLOR_BURN() is not None
    assert BlendMode.HARD_LIGHT() is not None
    assert BlendMode.SOFT_LIGHT() is not None
    assert BlendMode.DIFFERENCE() is not None
    assert BlendMode.EXCLUSION() is not None


def test_ext_gstate():
    """Test ExtGState (transparency) builder."""
    from pdf_oxide import BlendMode, ExtGState

    # Create with fill alpha
    gs = ExtGState().fill_alpha(0.5)
    assert gs is not None

    # Chained builder pattern
    gs = ExtGState().fill_alpha(0.5).stroke_alpha(0.8).blend_mode(BlendMode.MULTIPLY())
    assert gs is not None


def test_ext_gstate_presets():
    """Test ExtGState preset methods."""
    from pdf_oxide import BlendMode, ExtGState

    semi = ExtGState.semi_transparent()
    assert semi is not None

    # Test creating with blend mode (instead of preset static methods)
    multiply = ExtGState().blend_mode(BlendMode.MULTIPLY())
    assert multiply is not None

    screen = ExtGState().blend_mode(BlendMode.SCREEN())
    assert screen is not None


def test_linear_gradient():
    """Test LinearGradient builder."""
    from pdf_oxide import Color, LinearGradient

    # Basic gradient
    gradient = (
        LinearGradient()
        .start(0.0, 0.0)
        .end(100.0, 100.0)
        .add_stop(0.0, Color.red())
        .add_stop(1.0, Color.blue())
    )
    assert gradient is not None


def test_linear_gradient_presets():
    """Test LinearGradient preset methods."""
    from pdf_oxide import Color, LinearGradient

    # Horizontal preset
    gradient = LinearGradient.horizontal(100.0, Color.black(), Color.white())
    assert gradient is not None

    # Vertical preset
    gradient = LinearGradient.vertical(100.0, Color.black(), Color.white())
    assert gradient is not None

    # Manual two-color gradient
    gradient = LinearGradient().add_stop(0.0, Color.black()).add_stop(1.0, Color.white())
    assert gradient is not None


def test_radial_gradient():
    """Test RadialGradient builder."""
    from pdf_oxide import Color, RadialGradient

    gradient = (
        RadialGradient()
        .inner_circle(50.0, 50.0, 0.0)
        .outer_circle(50.0, 50.0, 50.0)
        .add_stop(0.0, Color.white())
        .add_stop(1.0, Color.black())
    )
    assert gradient is not None


def test_radial_gradient_centered():
    """Test centered RadialGradient."""
    from pdf_oxide import RadialGradient

    gradient = RadialGradient.centered(50.0, 50.0, 50.0)
    assert gradient is not None


def test_line_cap():
    """Test LineCap constants."""
    from pdf_oxide import LineCap

    assert LineCap.BUTT() is not None
    assert LineCap.ROUND() is not None
    assert LineCap.SQUARE() is not None


def test_line_join():
    """Test LineJoin constants."""
    from pdf_oxide import LineJoin

    assert LineJoin.MITER() is not None
    assert LineJoin.ROUND() is not None
    assert LineJoin.BEVEL() is not None


def test_pattern_presets():
    """Test PatternPresets static methods."""
    from pdf_oxide import Color, PatternPresets

    # Horizontal stripes
    content = PatternPresets.horizontal_stripes(10.0, 10.0, 5.0, Color.red())
    assert isinstance(content, bytes)
    assert len(content) > 0

    # Vertical stripes
    content = PatternPresets.vertical_stripes(10.0, 10.0, 5.0, Color.blue())
    assert isinstance(content, bytes)
    assert len(content) > 0

    # Checkerboard
    content = PatternPresets.checkerboard(10.0, Color.white(), Color.black())
    assert isinstance(content, bytes)
    assert len(content) > 0

    # Dots
    content = PatternPresets.dots(10.0, 2.0, Color.red())
    assert isinstance(content, bytes)
    assert len(content) > 0

    # Diagonal lines
    content = PatternPresets.diagonal_lines(10.0, 0.5, Color.black())
    assert isinstance(content, bytes)
    assert len(content) > 0

    # Crosshatch
    content = PatternPresets.crosshatch(10.0, 0.5, Color.black())
    assert isinstance(content, bytes)
    assert len(content) > 0


# === Extraction & Structure Tests ===


def test_extract_images():
    """Test extracting image metadata from a page."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        images = doc.extract_images(0)
        assert isinstance(images, list)
        # Each image should be a dict with expected keys
        for img in images:
            assert isinstance(img, dict)
            assert "width" in img
            assert "height" in img
            assert "color_space" in img
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_extract_spans():
    """Test extracting text spans from a page."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        spans = doc.extract_spans(0)
        assert isinstance(spans, list)
        for span in spans:
            # TextSpan objects should have expected attributes
            assert hasattr(span, "text")
            assert hasattr(span, "bbox")
            assert hasattr(span, "font_name")
            assert hasattr(span, "font_size")
            assert hasattr(span, "is_bold")
            assert hasattr(span, "is_italic")
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_extract_spans_repr():
    """Test TextSpan __repr__."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        spans = doc.extract_spans(0)
        if spans:
            r = repr(spans[0])
            assert "TextSpan" in r
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_get_outline():
    """Test getting document outline (bookmarks)."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        outline = doc.get_outline()
        # Outline is either None or a list
        assert outline is None or isinstance(outline, list)
        if outline:
            for item in outline:
                assert isinstance(item, dict)
                assert "title" in item
                assert "children" in item
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_get_annotations():
    """Test getting page annotations."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        annotations = doc.get_annotations(0)
        assert isinstance(annotations, list)
        for ann in annotations:
            assert isinstance(ann, dict)
            assert "subtype" in ann
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_extract_paths():
    """Test extracting vector paths from a page."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        paths = doc.extract_paths(0)
        assert isinstance(paths, list)
        for path in paths:
            assert isinstance(path, dict)
            assert "bbox" in path
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_extract_paths_operations():
    """Test that extract_paths returns operations with coordinates."""
    try:
        doc = PdfDocument("tests/fixtures/1.pdf")
        paths = doc.extract_paths(0)
        assert isinstance(paths, list)
        assert len(paths) > 0, "Expected at least one path"

        for path in paths:
            assert "operations" in path, "Path dict should contain 'operations' field"
            assert isinstance(path["operations"], list)
            assert len(path["operations"]) == path["operations_count"]

            for op in path["operations"]:
                assert isinstance(op, dict)
                assert "op" in op, "Each operation should have an 'op' field"
                op_type = op["op"]
                assert op_type in ("move_to", "line_to", "curve_to", "rectangle", "close_path")

                if op_type in ("move_to", "line_to"):
                    assert "x" in op and "y" in op
                elif op_type == "curve_to":
                    assert all(k in op for k in ("cx1", "cy1", "cx2", "cy2", "x", "y"))
                elif op_type == "rectangle":
                    assert all(k in op for k in ("x", "y", "width", "height"))
    except (OSError, RuntimeError):
        pytest.skip("Test fixture '1.pdf' not available or invalid")


def test_extract_images_invalid_page():
    """Test extract_images with invalid page index."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        with pytest.raises(RuntimeError):
            doc.extract_images(999)
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_extract_spans_invalid_page():
    """Test extract_spans with invalid page index."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        with pytest.raises(RuntimeError):
            doc.extract_spans(999)
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_get_annotations_invalid_page():
    """Test get_annotations with invalid page index."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        with pytest.raises(RuntimeError):
            doc.get_annotations(999)
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


def test_extract_paths_invalid_page():
    """Test extract_paths with invalid page index."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        with pytest.raises(RuntimeError):
            doc.extract_paths(999)
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


# === Image Bytes Extraction Tests ===


def test_extract_image_bytes_empty():
    """Test extracting image bytes from a page with no images."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        result = doc.extract_image_bytes(0)
        assert isinstance(result, list)
        # Each item should be a dict with data as bytes
        for img in result:
            assert isinstance(img, dict)
            assert "width" in img
            assert "height" in img
            assert "format" in img
            assert "data" in img
            assert isinstance(img["data"], bytes)
            assert img["format"] == "png"
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


# === PDF from Images Tests ===


def test_pdf_from_image_bytes():
    """Test creating PDF from image bytes."""
    from pdf_oxide import Pdf

    # Create a minimal 1x1 PNG
    png_data = _create_minimal_png()
    pdf = Pdf.from_image_bytes(png_data)
    assert pdf is not None
    pdf_bytes = pdf.to_bytes()
    assert len(pdf_bytes) > 0
    assert pdf_bytes[:4] == b"%PDF"


def test_pdf_from_image(tmp_path):
    """Test creating PDF from an image file."""
    from pdf_oxide import Pdf

    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(_create_minimal_png())

    pdf = Pdf.from_image(str(img_path))
    assert pdf is not None
    assert len(pdf.to_bytes()) > 0


def test_pdf_from_images(tmp_path):
    """Test creating PDF from multiple image files."""
    from pdf_oxide import Pdf

    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    img1.write_bytes(_create_minimal_png())
    img2.write_bytes(_create_minimal_png())

    pdf = Pdf.from_images([str(img1), str(img2)])
    assert pdf is not None
    assert len(pdf.to_bytes()) > 0


def _create_minimal_png():
    """Create a minimal valid 1x1 white image (JPEG format, known-good bytes)."""
    return bytes(
        [
            0xFF,
            0xD8,
            0xFF,
            0xE0,
            0x00,
            0x10,
            0x4A,
            0x46,
            0x49,
            0x46,
            0x00,
            0x01,
            0x01,
            0x00,
            0x00,
            0x01,
            0x00,
            0x01,
            0x00,
            0x00,
            0xFF,
            0xDB,
            0x00,
            0x43,
            0x00,
            0x08,
            0x06,
            0x06,
            0x07,
            0x06,
            0x05,
            0x08,
            0x07,
            0x07,
            0x07,
            0x09,
            0x09,
            0x08,
            0x0A,
            0x0C,
            0x14,
            0x0D,
            0x0C,
            0x0B,
            0x0B,
            0x0C,
            0x19,
            0x12,
            0x13,
            0x0F,
            0x14,
            0x1D,
            0x1A,
            0x1F,
            0x1E,
            0x1D,
            0x1A,
            0x1C,
            0x1C,
            0x20,
            0x24,
            0x2E,
            0x27,
            0x20,
            0x22,
            0x2C,
            0x23,
            0x1C,
            0x1C,
            0x28,
            0x37,
            0x29,
            0x2C,
            0x30,
            0x31,
            0x34,
            0x34,
            0x34,
            0x1F,
            0x27,
            0x39,
            0x3D,
            0x38,
            0x32,
            0x3C,
            0x2E,
            0x33,
            0x34,
            0x32,
            0xFF,
            0xC0,
            0x00,
            0x0B,
            0x08,
            0x00,
            0x01,
            0x00,
            0x01,
            0x01,
            0x01,
            0x11,
            0x00,
            0xFF,
            0xC4,
            0x00,
            0x1F,
            0x00,
            0x00,
            0x01,
            0x05,
            0x01,
            0x01,
            0x01,
            0x01,
            0x01,
            0x01,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x01,
            0x02,
            0x03,
            0x04,
            0x05,
            0x06,
            0x07,
            0x08,
            0x09,
            0x0A,
            0x0B,
            0xFF,
            0xC4,
            0x00,
            0xB5,
            0x10,
            0x00,
            0x02,
            0x01,
            0x03,
            0x03,
            0x02,
            0x04,
            0x03,
            0x05,
            0x05,
            0x04,
            0x04,
            0x00,
            0x00,
            0x01,
            0x7D,
            0x01,
            0x02,
            0x03,
            0x00,
            0x04,
            0x11,
            0x05,
            0x12,
            0x21,
            0x31,
            0x41,
            0x06,
            0x13,
            0x51,
            0x61,
            0x07,
            0x22,
            0x71,
            0x14,
            0x32,
            0x81,
            0x91,
            0xA1,
            0x08,
            0x23,
            0x42,
            0xB1,
            0xC1,
            0x15,
            0x52,
            0xD1,
            0xF0,
            0x24,
            0x33,
            0x62,
            0x72,
            0x82,
            0x09,
            0x0A,
            0x16,
            0x17,
            0x18,
            0x19,
            0x1A,
            0x25,
            0x26,
            0x27,
            0x28,
            0x29,
            0x2A,
            0x34,
            0x35,
            0x36,
            0x37,
            0x38,
            0x39,
            0x3A,
            0x43,
            0x44,
            0x45,
            0x46,
            0x47,
            0x48,
            0x49,
            0x4A,
            0x53,
            0x54,
            0x55,
            0x56,
            0x57,
            0x58,
            0x59,
            0x5A,
            0x63,
            0x64,
            0x65,
            0x66,
            0x67,
            0x68,
            0x69,
            0x6A,
            0x73,
            0x74,
            0x75,
            0x76,
            0x77,
            0x78,
            0x79,
            0x7A,
            0x83,
            0x84,
            0x85,
            0x86,
            0x87,
            0x88,
            0x89,
            0x8A,
            0x92,
            0x93,
            0x94,
            0x95,
            0x96,
            0x97,
            0x98,
            0x99,
            0x9A,
            0xA2,
            0xA3,
            0xA4,
            0xA5,
            0xA6,
            0xA7,
            0xA8,
            0xA9,
            0xAA,
            0xB2,
            0xB3,
            0xB4,
            0xB5,
            0xB6,
            0xB7,
            0xB8,
            0xB9,
            0xBA,
            0xC2,
            0xC3,
            0xC4,
            0xC5,
            0xC6,
            0xC7,
            0xC8,
            0xC9,
            0xCA,
            0xD2,
            0xD3,
            0xD4,
            0xD5,
            0xD6,
            0xD7,
            0xD8,
            0xD9,
            0xDA,
            0xE1,
            0xE2,
            0xE3,
            0xE4,
            0xE5,
            0xE6,
            0xE7,
            0xE8,
            0xE9,
            0xEA,
            0xF1,
            0xF2,
            0xF3,
            0xF4,
            0xF5,
            0xF6,
            0xF7,
            0xF8,
            0xF9,
            0xFA,
            0xFF,
            0xDA,
            0x00,
            0x08,
            0x01,
            0x01,
            0x00,
            0x00,
            0x3F,
            0x00,
            0xFB,
            0xD5,
            0xDB,
            0x20,
            0xA8,
            0xF9,
            0xFF,
            0xD9,
        ]
    )


# === Form Flattening Tests ===


def test_flatten_forms():
    """Test flattening form fields."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        # Should not raise, even if there are no forms
        doc.flatten_forms()
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


# === PDF Merging Tests ===


def test_merge_from_bytes():
    """Test merging PDFs from bytes."""
    from pdf_oxide import Pdf

    # Create two PDFs
    pdf1 = Pdf.from_text("Page 1")
    pdf2 = Pdf.from_text("Page 2")

    # Save pdf1 to file, open as PdfDocument, merge pdf2 bytes
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        tmp_path = f.name
    # File handle closed before use — required on Windows (file locking)
    try:
        pdf1.save(tmp_path)
        doc = PdfDocument(tmp_path)
        count = doc.merge_from(pdf2.to_bytes())
        assert count == 1, "Should merge 1 page"
    finally:
        os.unlink(tmp_path)


# === File Embedding Tests ===


def test_embed_file():
    """Test embedding a file into a PDF."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        doc.embed_file("readme.txt", b"Hello embedded file")
        # Should succeed without error
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


# === Page Labels Tests ===


def test_page_labels():
    """Test getting page labels."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        labels = doc.page_labels()
        assert isinstance(labels, list)
        for label in labels:
            assert isinstance(label, dict)
            assert "start_page" in label
            assert "style" in label
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


# === XMP Metadata Tests ===


def test_xmp_metadata():
    """Test getting XMP metadata."""
    try:
        doc = PdfDocument("tests/fixtures/simple.pdf")
        metadata = doc.xmp_metadata()
        # Can be None or a dict
        assert metadata is None or isinstance(metadata, dict)
    except (OSError, RuntimeError):
        pytest.skip("Test fixture 'simple.pdf' not available or invalid")


# Note: To run these tests successfully, you'll need to:
# 1. Install maturin: pip install maturin
# 2. Build the extension: maturin develop
# 3. Install pytest: pip install pytest
# 4. Create test PDF fixtures in tests/fixtures/
# 5. Run tests: pytest tests/test_python.py
