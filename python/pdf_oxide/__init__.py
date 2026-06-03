# SPDX-License-Identifier: MIT OR Apache-2.0
"""
PDF Oxide - The Complete PDF Toolkit

Extract, create, and edit PDFs with one library.
Rust core with Python bindings. Fast, safe, dependency-free.

# Extract. Create. Edit.

## Extract
- Text with reading order and layout analysis
- Images (JPEG, PNG, TIFF)
- Forms and annotations
- Convert to Markdown, HTML, PlainText

## Create
- Fluent API: `Pdf.create()`
- Tables, images, graphics
- Colors, gradients, patterns

## Edit
- Annotations (highlights, notes, stamps)
- Form fields (text, checkbox, radio)
- Round-trip: modify existing PDFs

# Quick Start

```python
from pdf_oxide import PdfDocument, Pdf

# Extract
doc = PdfDocument("input.pdf")
text = doc.to_plain_text(0)

# Create
pdf = Pdf.create()
pdf.add_page().text("Hello!", x=72, y=720, size=24)
pdf.save("output.pdf")
```

# License

Dual-licensed under MIT OR Apache-2.0.
"""

import os as _os
from typing import NamedTuple


def _setup_ort_dylib_path() -> None:
    """Point ort's dynamic loader at the onnxruntime library shipped by the
    Python ``onnxruntime`` package, if installed and not already overridden.

    The ``ort`` Rust crate (used for OCR) searches for ``libonnxruntime.so``
    (Linux), ``libonnxruntime.dylib`` (macOS), or ``onnxruntime.dll``
    (Windows) via ``ORT_DYLIB_PATH``.  The Python ``onnxruntime`` package
    ships this library inside its own ``capi/`` directory, so we locate it
    here and set the env-var before the native module is loaded.
    """
    if _os.environ.get("ORT_DYLIB_PATH"):
        return  # already set by user — respect it
    try:
        import importlib.util as _ilu

        spec = _ilu.find_spec("onnxruntime")
        if spec is None or spec.origin is None:
            return
        import pathlib as _pl

        capi_dir = _pl.Path(spec.origin).parent / "capi"
        candidates = [
            capi_dir / "libonnxruntime.so",
            capi_dir / "libonnxruntime.dylib",
            capi_dir / "onnxruntime.dll",
        ]
        # Also match versioned names like libonnxruntime.so.1.20.1
        if capi_dir.is_dir():
            for f in capi_dir.iterdir():
                name = f.name
                if (
                    name.startswith("libonnxruntime.so")
                    or name.startswith("libonnxruntime.dylib")
                    or name == "onnxruntime.dll"
                ):
                    candidates.insert(0, f)
        for candidate in candidates:
            if candidate.exists():
                _os.environ["ORT_DYLIB_PATH"] = str(candidate)
                break
    except Exception:
        pass  # non-fatal — OCR will raise a clear error if ort can't load


_setup_ort_dylib_path()


def _setup_default_log_levels() -> None:
    """Quiet stderr-spam from internal pdf_oxide warnings under default
    Python logging config.

    pdf_oxide routes every internal ``log::warn!`` through the ``pyo3_log``
    bridge, which forwards records to Python's ``logging`` module. Python's
    default root-logger config emits ``WARNING``-level records to stderr,
    so every quirky-but-recoverable PDF produces noise like
    ``SPEC VIOLATION: No newline after stream keyword``,
    ``Type0 font 'X' has no ToUnicode entry!``, etc. — observed at ~150
    lines per PDF in `pdfa_001.pdf`.

    This function attaches a ``NullHandler`` to each of the four
    highest-frequency internal targets and disables propagation, so
    records stop at the pdf_oxide logger boundary instead of bubbling
    up to the root logger's default stderr handler.

    This is the standard Python library convention (see PEP 282 + the
    ``logging`` HOWTO): a library never owns the user's logger level
    or root handler config; it provides a NullHandler so records have
    somewhere to land, and ``propagate = False`` so its own records
    don't surface unless the caller explicitly opts in.

    Callers who want the warnings back can:

    - Use ``logging.getLogger("pdf_oxide.parser").propagate = True``
      to re-enable bubbling for a single category, OR add a handler
      to that logger directly.
    - Use ``doc.structured_warnings()`` to receive the warnings as
      structured ``Warning`` dicts (category, page, message,
      spec_section) instead of stderr text.
      (``doc.flatten_warnings()`` is the pre-existing form-flattening
      surface returning ``list[str]`` — different feature.)

    The setup is idempotent: repeated calls are harmless (NullHandler
    is added at most once via instance check). Genuine ERROR-level
    events bubble through the ``Result`` chain into Python exceptions,
    not through ``log::warn!``, so this does not hide real errors.

    See ``docs/releases/plans/v0.3.56/cluster-diagnostics-noise.md``.
    """
    import logging as _logging

    _quiet_targets = (
        "pdf_oxide.parser",
        "pdf_oxide.content",
        "pdf_oxide.fonts",
        "pdf_oxide.document",
    )
    for _target in _quiet_targets:
        _logger = _logging.getLogger(_target)
        if not any(isinstance(h, _logging.NullHandler) for h in _logger.handlers):
            _logger.addHandler(_logging.NullHandler())
        _logger.propagate = False


_setup_default_log_levels()


class RenderedPixmap(NamedTuple):
    """Raw premultiplied RGBA8888 pixel buffer from :meth:`PdfDocument.render_pixmap`.

    Attributes:
        data (bytes): Row-major, top-left origin, 4 bytes (R,G,B,A) per pixel.
            ``len(data) == width * height * 4``. Alpha is premultiplied (PDF
            spec §11 transparency model).
        width (int): Image width in pixels.
        height (int): Image height in pixels.
    """

    data: bytes
    width: int
    height: int


class SeparationPlate(NamedTuple):
    """A single ink separation plate rendered as grayscale.

    Pixel intensity (0-255) represents the tint percentage of one ink at
    each point. Used in prepress workflows, ink coverage analysis, and ML
    pipelines that process packaging/label PDFs.

    The pixel convention is ML/QC-friendly: ``value == ink coverage``.
    0 means no ink, 255 means full tint coverage. To display the plate as
    black ink on white paper (prepress viewer convention), invert before
    showing: ``display = 255 - value``.

    Attributes:
        ink_name (str): Ink name (e.g., "Cyan", "PANTONE 185 C", "Dieline").
        data (bytes): Grayscale pixels, row-major, top-left origin.
            ``len(data) == width * height``. 0 = no ink, 255 = full tint.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
    """

    ink_name: str
    data: bytes
    width: int
    height: int


from ._async import (  # noqa: E402
    AsyncOfficeConverter,
    AsyncPdf,
    AsyncPdfDocument,
)
from .pdf_oxide import (  # noqa: E402
    VERSION,
    # v0.3.39 tables + primitives
    Align,
    # Page Templates
    Artifact,
    ArtifactStyle,
    BlendMode,
    # Digital signatures / PAdES (issue #235)
    Certificate,
    # Advanced Graphics
    Color,
    Column,
    # Write-side fluent API
    DocumentBuilder,
    Dss,
    EmbeddedFont,
    ExtGState,
    # Extraction
    ExtractionProfile,
    FluentPageBuilder,
    Footer,
    Header,
    LayoutParams,
    LinearGradient,
    LineCap,
    LineJoin,
    # OCR (always available as stub if feature is off)
    OcrConfig,
    OcrEngine,
    # Office (always available as stub if feature is off)
    OfficeConverter,
    PadesLevel,
    PageTemplate,
    PatternPresets,
    # PDF Creation
    Pdf,
    PdfDocument,
    RadialGradient,
    RevocationMaterial,
    Signature,
    StreamingTable,
    Table,
    TextSpan,
    crypto_active_provider,
    crypto_available_providers,
    crypto_cbom,
    crypto_inventory,
    crypto_policy,
    crypto_set_policy,
    crypto_use_fips,
    disable_logging,
    generate_barcode_svg,
    generate_qr_svg,
    get_log_level,
    has_document_timestamp,
    plan_split_by_bookmarks,
    set_log_level,
    setup_logging,
    sign_pdf_bytes,
    sign_pdf_bytes_pades,
    split_by_bookmarks,
)


__all__ = [
    "RenderedPixmap",
    "SeparationPlate",
    "PdfDocument",
    "AsyncPdfDocument",
    "AsyncPdf",
    "AsyncOfficeConverter",
    "VERSION",
    # Write-side fluent API
    "DocumentBuilder",
    "FluentPageBuilder",
    "EmbeddedFont",
    # v0.3.39 tables + primitives (#393)
    "Align",
    "Column",
    "Table",
    "StreamingTable",
    # PDF Creation
    "Pdf",
    # Advanced Graphics
    "Color",
    "BlendMode",
    "ExtGState",
    "LinearGradient",
    "RadialGradient",
    "LineCap",
    "LineJoin",
    "PatternPresets",
    # Page Templates
    "ArtifactStyle",
    "Artifact",
    "Header",
    "Footer",
    "PageTemplate",
    # Extraction
    "ExtractionProfile",
    "LayoutParams",
    "TextSpan",
    # OCR
    "OcrEngine",
    "OcrConfig",
    # Office
    "OfficeConverter",
    # Barcodes (#421)
    "generate_barcode_svg",
    "generate_qr_svg",
    # Logging
    "setup_logging",
    "set_log_level",
    "get_log_level",
    "disable_logging",
    # FIPS crypto-provider surface (issue #236)
    "crypto_active_provider",
    "crypto_available_providers",
    "crypto_use_fips",
    # Runtime crypto-governance policy surface (issue #230)
    "crypto_cbom",
    "crypto_inventory",
    "crypto_policy",
    "crypto_set_policy",
    # Digital signatures / PAdES surface (issue #235)
    "Certificate",
    "Signature",
    "PadesLevel",
    "RevocationMaterial",
    "Dss",
    "sign_pdf_bytes",
    "sign_pdf_bytes_pades",
    "has_document_timestamp",
    # Split-by-bookmarks surface (issue #482)
    "plan_split_by_bookmarks",
    "split_by_bookmarks",
]
__version__ = VERSION
