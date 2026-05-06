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
    # Advanced Graphics
    Color,
    Column,
    # Write-side fluent API
    DocumentBuilder,
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
    PageTemplate,
    PatternPresets,
    # PDF Creation
    Pdf,
    PdfDocument,
    RadialGradient,
    StreamingTable,
    Table,
    TextSpan,
    crypto_active_provider,
    crypto_available_providers,
    crypto_use_fips,
    disable_logging,
    generate_barcode_svg,
    generate_qr_svg,
    get_log_level,
    set_log_level,
    setup_logging,
)


__all__ = [
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
]
__version__ = VERSION
