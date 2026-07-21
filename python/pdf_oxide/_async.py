"""Async wrappers for pdf_oxide.

Provides:

- ``AsyncPdfDocument`` -- async interface for reading/editing existing PDFs
- ``AsyncPdf`` -- async interface for creating new PDFs (mirrors ``Pdf``)
- ``AsyncOfficeConverter`` -- async interface for Office conversion

Each wrapper runs all PDF operations in a background thread via
``run_in_executor`` so the event loop is never blocked.

All public methods from the sync classes are auto-generated as async
wrappers, so they stay in sync automatically when the Rust bindings
add new methods.

Example::

    import asyncio
    from pdf_oxide import AsyncPdfDocument, AsyncPdf

    async def main():
        # Read
        doc = await AsyncPdfDocument.open("report.pdf")
        text = await doc.extract_text(0)

        # Create
        pdf = await AsyncPdf.from_markdown("# Hello")
        await pdf.save("hello.pdf")

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

from .pdf_oxide import OfficeConverter, Pdf, PdfDocument


# ---------------------------------------------------------------------------
#  Auto-generation helpers
# ---------------------------------------------------------------------------


def _make_async_method(method_name: str):
    """Create an async method that delegates to ``self._doc.<method_name>``
    via ``run_in_executor`` on ``self._executor``."""

    @functools.wraps(getattr(PdfDocument, method_name, None) or (lambda: None))
    async def method(self, *args, **kwargs):
        loop = asyncio.get_running_loop()
        fn = getattr(self._doc, method_name)
        if kwargs:
            return await loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))
        return await loop.run_in_executor(self._executor, fn, *args)

    method.__name__ = method_name
    method.__qualname__ = f"AsyncPdfDocument.{method_name}"
    return method


def _make_async_static(sync_cls, method_name: str, wrap_result=None):
    """Create a static async method that delegates to
    ``sync_cls.<method_name>`` via the default executor.

    If *wrap_result* is given, the sync return value is wrapped with it.
    """

    @staticmethod
    async def method(*args, **kwargs):
        loop = asyncio.get_running_loop()
        fn = getattr(sync_cls, method_name)
        if kwargs:
            result = await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
        else:
            result = await loop.run_in_executor(None, fn, *args)
        return wrap_result(result) if wrap_result else result

    method.__name__ = method_name
    return method


# ---------------------------------------------------------------------------
#  AsyncPdfDocument
# ---------------------------------------------------------------------------

# Methods that need custom handling (constructors, properties, special logic)
_ASYNC_DOC_SKIP = frozenset(
    {
        "from_bytes",  # static constructor -- handled manually
    }
)


class AsyncPdfDocument:
    """Async wrapper around :class:`PdfDocument`.

    Each instance owns a single-worker thread pool that offloads PDF
    operations so the event loop is never blocked.  The underlying
    ``PdfDocument`` is ``Send + Sync``, so it is safe to use from any
    thread.

    All public methods from the sync ``PdfDocument`` are available as
    ``async`` methods with identical names and signatures.

    Supports async context manager for deterministic cleanup::

        async with await AsyncPdfDocument.open("doc.pdf") as doc:
            text = await doc.extract_text(0)
    """

    __slots__ = ("_executor", "_doc")

    def __init__(self, doc: PdfDocument) -> None:
        self._doc = doc
        self._executor = ThreadPoolExecutor(max_workers=1)

    # -- Construction (hand-written) ----------------------------------------

    @staticmethod
    async def open(path: str, password: str | None = None) -> AsyncPdfDocument:
        """Open a PDF file.  The document is created on the background thread."""
        loop = asyncio.get_running_loop()
        inst = AsyncPdfDocument.__new__(AsyncPdfDocument)
        inst._executor = ThreadPoolExecutor(max_workers=1)

        def _open():
            return PdfDocument(path, password) if password else PdfDocument(path)

        inst._doc = await loop.run_in_executor(inst._executor, _open)
        return inst

    @staticmethod
    async def from_bytes(data: bytes, password: str | None = None) -> AsyncPdfDocument:
        """Open a PDF from bytes."""
        loop = asyncio.get_running_loop()
        inst = AsyncPdfDocument.__new__(AsyncPdfDocument)
        inst._executor = ThreadPoolExecutor(max_workers=1)

        def _open():
            return (
                PdfDocument.from_bytes(data, password) if password else PdfDocument.from_bytes(data)
            )

        inst._doc = await loop.run_in_executor(inst._executor, _open)
        return inst

    # -- Sync access --------------------------------------------------------

    @property
    def doc(self) -> PdfDocument:
        """Access the underlying sync ``PdfDocument``."""
        return self._doc

    async def close(self) -> None:
        """Shut down the background thread pool."""
        self._executor.shutdown(wait=False)

    async def __aenter__(self) -> AsyncPdfDocument:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    def __del__(self):
        self._executor.shutdown(wait=False)


# Auto-populate every public method from PdfDocument
def _populate_async_pdf_document() -> None:
    for name in dir(PdfDocument):
        if name.startswith("_") or name in _ASYNC_DOC_SKIP:
            continue
        attr = getattr(PdfDocument, name, None)
        if callable(attr) and not hasattr(AsyncPdfDocument, name):
            setattr(AsyncPdfDocument, name, _make_async_method(name))


_populate_async_pdf_document()


# ---------------------------------------------------------------------------
#  AsyncPdf  --  mirrors the sync ``Pdf`` creation API
# ---------------------------------------------------------------------------


class AsyncPdf:
    """Async wrapper around :class:`Pdf` for PDF creation.

    All ``from_*`` static methods return an ``AsyncPdf`` instance.
    Call :meth:`save` or :meth:`to_bytes` to get the result.

    Example::

        pdf = await AsyncPdf.from_markdown("# Title")
        await pdf.save("out.pdf")
        raw = await pdf.to_bytes()
    """

    __slots__ = ("_pdf",)

    def __init__(self, pdf: Pdf) -> None:
        self._pdf = pdf

    # -- Instance methods (hand-written) ------------------------------------

    async def save(self, path: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._pdf.save, path)

    async def to_bytes(self) -> bytes:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._pdf.to_bytes)

    @property
    def pdf(self) -> Pdf:
        """Access the underlying sync ``Pdf``."""
        return self._pdf

    def __repr__(self) -> str:
        return f"AsyncPdf({self._pdf!r})"


# Auto-populate static factory methods from Pdf (wrapped in a function so the
# iteration variables don't leak into module scope and so basedpyright can see
# that any `del` is unnecessary).
def _populate_async_pdf() -> None:
    for name in dir(Pdf):
        if name.startswith("_") or name in ("save", "to_bytes"):
            continue
        attr = getattr(Pdf, name, None)
        if callable(attr) and not hasattr(AsyncPdf, name):
            setattr(AsyncPdf, name, _make_async_static(Pdf, name, wrap_result=AsyncPdf))


_populate_async_pdf()


# ---------------------------------------------------------------------------
#  AsyncOfficeConverter  --  mirrors the sync ``OfficeConverter`` API
# ---------------------------------------------------------------------------


class AsyncOfficeConverter:
    """Async wrapper around :class:`OfficeConverter`.

    All methods are static and return an :class:`AsyncPdf`.

    Example::

        pdf = await AsyncOfficeConverter.from_docx("report.docx")
        await pdf.save("report.pdf")
    """


def _populate_async_office_converter() -> None:
    for name in dir(OfficeConverter):
        if name.startswith("_"):
            continue
        attr = getattr(OfficeConverter, name, None)
        if callable(attr):
            setattr(
                AsyncOfficeConverter,
                name,
                _make_async_static(OfficeConverter, name, wrap_result=AsyncPdf),
            )


_populate_async_office_converter()
