"""Type stubs for pdf_oxide"""

from typing import Optional, Tuple

class PdfDocument:
    """
    PDF document parser and converter with specification compliance.

    Provides high-performance PDF parsing with multiple output formats,
    all supporting automatic reading order detection for multi-column layouts.

    Features:
        - ISO 32000-1:2008 PDF specification compliance
        - 70-80% character recovery with advanced font support
        - Automatic multi-column layout detection (4 strategies)
        - Complex script support (RTL, CJK, Devanagari, Thai)
        - OCR support for scanned PDFs (optional)
        - 47.9× faster than PyMuPDF4LLM

    Methods:
        - to_plain_text(page, ...): Plain text with optional layout preservation
        - to_markdown(page, ...): Formatted markdown with automatic reading order
        - to_html(page, ...): Semantic HTML with automatic reading order

    Example:
        >>> doc = PdfDocument("sample.pdf")
        >>> print(doc.version())
        (1, 7)
        >>> # Plain text (with reading order)
        >>> text = doc.to_plain_text(0)
        >>> # Formatted output with reading order
        >>> markdown = doc.to_markdown(0, detect_headings=True)
    """

    def __init__(self, path: str) -> None:
        """
        Open a PDF file.

        Args:
            path: Path to the PDF file

        Raises:
            IOError: If the file cannot be opened or is not a valid PDF

        Example:
            >>> doc = PdfDocument("sample.pdf")
        """
        ...

    def version(self) -> Tuple[int, int]:
        """
        Get PDF version.

        Returns:
            Tuple of (major, minor) version numbers, e.g. (1, 7) for PDF 1.7

        Example:
            >>> doc = PdfDocument("sample.pdf")
            >>> major, minor = doc.version()
            >>> print(f"PDF {major}.{minor}")
        """
        ...

    def page_count(self) -> int:
        """
        Get the number of pages in the document.

        Returns:
            Number of pages

        Raises:
            RuntimeError: If page count cannot be determined

        Example:
            >>> doc = PdfDocument("sample.pdf")
            >>> print(f"Pages: {doc.page_count()}")
        """
        ...

    def to_markdown(
        self,
        page: int,
        preserve_layout: bool = False,
        detect_headings: bool = True,
        include_images: bool = True,
        image_output_dir: Optional[str] = None,
    ) -> str:
        """
        Convert a page to Markdown with intelligent layout handling.

        Uses pluggable reading order strategies for accurate multi-column detection.
        Automatically handles complex scripts and maintains logical structure.

        Args:
            page: Page index (0-based)
            preserve_layout: If True, preserve visual layout (default: False)
            detect_headings: If True, detect headings by specification analysis (default: True)
            include_images: If True, include embedded images (default: True)
            image_output_dir: Directory to save extracted images, or None to skip (default: None)

        Returns:
            Markdown text with proper reading order and formatting

        Raises:
            RuntimeError: If conversion fails

        Example:
            >>> doc = PdfDocument("research_paper.pdf")
            >>> markdown = doc.to_markdown(0, detect_headings=True)
            >>> with open("output.md", "w") as f:
            ...     f.write(markdown)
        """
        ...

    def to_html(
        self,
        page: int,
        preserve_layout: bool = False,
        detect_headings: bool = True,
        include_images: bool = True,
        image_output_dir: Optional[str] = None,
    ) -> str:
        """
        Convert a page to HTML with semantic structure.

        Produces semantic HTML with proper reading order. Automatically detects
        multi-column layouts and converts to single-column HTML structure.

        Args:
            page: Page index (0-based)
            preserve_layout: If True, preserve visual layout with CSS positioning (default: False)
            detect_headings: If True, detect headings semantically (default: True)
            include_images: If True, embed extracted images (default: True)
            image_output_dir: Directory to save extracted images, or None to skip (default: None)

        Returns:
            Semantic HTML text with proper reading order

        Raises:
            RuntimeError: If conversion fails

        Example:
            >>> doc = PdfDocument("article.pdf")
            >>> html = doc.to_html(0, preserve_layout=False)
            >>> with open("output.html", "w") as f:
            ...     f.write(html)
        """
        ...

    def to_markdown_all(
        self,
        preserve_layout: bool = False,
        detect_headings: bool = True,
        include_images: bool = True,
        image_output_dir: Optional[str] = None,
    ) -> str:
        """
        Convert all pages to Markdown format.

        Pages are separated by horizontal rules (---).

        Args:
            preserve_layout: If True, preserve visual layout (default: False)
            detect_headings: If True, detect headings based on font size (default: True)
            include_images: If True, include images in output (default: True)
            image_output_dir: Directory to save images, or None to skip saving (default: None)

        Returns:
            Markdown text containing all pages

        Raises:
            RuntimeError: If conversion fails

        Example:
            >>> doc = PdfDocument("book.pdf")
            >>> markdown = doc.to_markdown_all(detect_headings=True)
            >>> with open("book.md", "w") as f:
            ...     f.write(markdown)
        """
        ...

    def to_html_all(
        self,
        preserve_layout: bool = False,
        detect_headings: bool = True,
        include_images: bool = True,
        image_output_dir: Optional[str] = None,
    ) -> str:
        """
        Convert all pages to HTML format.

        Each page is wrapped in a div.page element with a data-page attribute.

        Args:
            preserve_layout: If True, preserve visual layout with CSS positioning (default: False)
            detect_headings: If True, detect headings based on font size (default: True)
            include_images: If True, include images in output (default: True)
            image_output_dir: Directory to save images, or None to skip saving (default: None)

        Returns:
            HTML text containing all pages

        Raises:
            RuntimeError: If conversion fails

        Example:
            >>> doc = PdfDocument("book.pdf")
            >>> html = doc.to_html_all(preserve_layout=True)
            >>> with open("book.html", "w") as f:
            ...     f.write(html)
        """
        ...

    def __repr__(self) -> str:
        """
        String representation of the document.

        Returns:
            Representation showing PDF version
        """
        ...

__version__: str
__all__: list[str]
