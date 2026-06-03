<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide;

use FFI\CData;
use PdfOxide\Exceptions\InvalidStateException;
use PdfOxide\Exceptions\IoException;
use PdfOxide\FFI\FunctionBindings;

/**
 * The primary read-only entry point to a PDF.
 *
 * Mirrors `fyi.oxide.pdf.PdfDocument` from the Java binding. Owns a
 * native handle that MUST be released via {@see close()}. PHP doesn't
 * have try-with-resources; rely on `__destruct()` cleanup or call
 * `close()` explicitly.
 *
 * Lifecycle:
 *   - `__destruct()` is best-effort — explicit `close()` is preferred.
 *   - `close()` is idempotent; calling more than once is a no-op.
 *
 * Thread safety: instances are NOT safe to share across threads. PHP
 * is single-threaded by default; this matters only if you're using
 * pthreads/parallel.
 *
 * Convenience helpers `extractTextOnce()` open + extract + close in a
 * single call.
 */
final class PdfDocument
{
    private ?CData $handle = null;

    private readonly FunctionBindings $bindings;

    /** @var string|null absolute path or `null` when opened from bytes */
    private readonly ?string $sourcePath;

    /**
     * Internal: use one of the static factories.
     */
    private function __construct(CData $handle, ?string $sourcePath)
    {
        $this->bindings = new FunctionBindings();
        $this->handle = $handle;
        $this->sourcePath = $sourcePath;
    }

    // ────────────────────── factories ──────────────────────

    /**
     * Open a PDF from a filesystem path.
     *
     * @throws IoException when the file is missing
     * @throws \PdfOxide\Exceptions\ParseException on malformed PDF bytes
     * @throws \PdfOxide\Exceptions\EncryptionException on a password-required PDF
     */
    public static function open(string $path): self
    {
        if (! is_file($path)) {
            throw new IoException("PDF file not found: {$path}", ['file' => $path]);
        }
        $bindings = new FunctionBindings();
        $handle = $bindings->pdfDocumentOpen($path);
        if ($handle === null) {
            throw new IoException("Failed to open PDF: {$path}", ['file' => $path]);
        }
        return new self($handle, $path);
    }

    /**
     * Open a PDF from an in-memory byte string. Writes to a temp file
     * because pdf_oxide's PHP FFI surface only exposes a path-based
     * opener.
     *
     * @throws IoException when the temp file can't be written
     * @throws \PdfOxide\Exceptions\ParseException on malformed PDF bytes
     */
    public static function openBytes(string $bytes): self
    {
        $tmp = tempnam(sys_get_temp_dir(), 'pdf_oxide_');
        if ($tmp === false) {
            throw new IoException('Failed to allocate temp file for in-memory PDF');
        }
        if (file_put_contents($tmp, $bytes) === false) {
            @unlink($tmp);
            throw new IoException('Failed to write in-memory PDF to temp file');
        }
        try {
            $doc = self::open($tmp);
        } catch (\Throwable $e) {
            @unlink($tmp);
            throw $e;
        }
        // Keep the temp file for the document's lifetime — it'll be unlinked on close.
        $doc->ownedTempPath = $tmp;
        return $doc;
    }

    /** @internal temp file the document owns (deleted in close()) */
    private ?string $ownedTempPath = null;

    // ─────────── static one-shot convenience ───────────────

    /**
     * Convenience: open + `extractText(0)` + close in a single call.
     */
    public static function extractTextOnce(string $path): string
    {
        $doc = self::open($path);
        try {
            return $doc->extractText(0);
        } finally {
            $doc->close();
        }
    }

    // ─────────────────────── instance ──────────────────────

    /** @return int the number of pages in the document */
    public function pageCount(): int
    {
        return $this->bindings->pdfDocumentGetPageCount($this->requireHandle());
    }

    /**
     * Extract plain text from a single page.
     *
     * @throws \OutOfBoundsException when `$pageIndex` is out of range
     * @throws InvalidStateException when the document has been closed
     */
    public function extractText(int $pageIndex): string
    {
        return $this->bindings->pdfDocumentExtractText($this->requireHandle(), $pageIndex);
    }

    /**
     * Extract a structured layout view of a single page (#536).
     *
     * Returns the deserialized `StructuredPage` as an associative array:
     * `['page_index' => int, 'page_width' => float, 'page_height' => float,
     *   'regions' => [['kind' => string, 'text' => string, 'bbox' => [...],
     *                  'spans' => [...], 'column_index' => int], ...]]`.
     *
     * @return array<string, mixed>
     *
     * @throws \OutOfBoundsException when `$page` is out of range
     * @throws InvalidStateException when the document has been closed
     */
    public function extractStructured(int $page): array
    {
        $json = $this->bindings->pdfDocumentExtractStructuredToJson($this->requireHandle(), $page);
        return json_decode($json, true);
    }

    /**
     * Auto-routed extraction (v0.3.51 #517). Returns native text when
     * present, OCR text for scanned regions when the `ocr` feature is
     * available, and gracefully falls back to native text + a logged
     * warning when OCR is unavailable — NEVER throws on the
     * graceful-fallback path.
     */
    public function extractTextAuto(int $pageIndex): string
    {
        return $this->bindings->pdfDocumentExtractTextAuto($this->requireHandle(), $pageIndex);
    }

    /** PDF version as `['major' => int, 'minor' => int]`. */
    public function version(): array
    {
        return $this->bindings->pdfDocumentGetVersion($this->requireHandle());
    }

    /** Whether the document carries a logical structure tree (PDF/UA prereq). */
    public function hasStructureTree(): bool
    {
        return $this->bindings->pdfDocumentHasStructureTree($this->requireHandle());
    }

    /**
     * Whether the document has any AcroForm/XFA form fields. The
     * C ABI doesn't expose a direct `has_form_fields` predicate; this
     * counts via `pdf_document_get_form_fields` and reads the list
     * length (frees the list before returning).
     */
    public function hasFormFields(): bool
    {
        $ffi = \PdfOxide\FFI\NativeLibrary::getInstance();
        $errorCode = $ffi->new('int32_t');
        $list = $ffi->pdf_document_get_form_fields($this->requireHandle(), \FFI::addr($errorCode));
        if ($list === null) {
            return false;
        }
        try {
            return ((int) $ffi->pdf_oxide_form_field_count($list)) > 0;
        } finally {
            $ffi->pdf_oxide_form_field_list_free($list);
        }
    }

    /** Whether the document has any embedded digital signatures. */
    public function hasSignatures(): bool
    {
        $ffi = \PdfOxide\FFI\NativeLibrary::getInstance();
        $errorCode = $ffi->new('int32_t');
        $count = (int) $ffi->pdf_document_get_signature_count($this->requireHandle(), \FFI::addr($errorCode));
        return $count > 0;
    }

    // ──────────────── markdown / html shortcuts ────────────

    /** Per-page Markdown conversion. Equivalent to {@see MarkdownConverter::toMarkdown()}. */
    public function toMarkdown(int $pageIndex = 0): string
    {
        return MarkdownConverter::toMarkdown($this, $pageIndex);
    }

    /** Whole-document Markdown conversion. */
    public function toMarkdownAll(): string
    {
        return MarkdownConverter::toMarkdownAll($this);
    }

    /** Per-page HTML conversion. */
    public function toHtml(int $pageIndex = 0): string
    {
        return MarkdownConverter::toHtml($this, $pageIndex);
    }

    // ───────────────────── page iteration ──────────────────

    /**
     * Get a lightweight view of a single page. The {@see PdfPage} is
     * invalidated when the parent document is closed.
     *
     * @throws \OutOfRangeException when `$index` is out of range
     */
    public function page(int $index): PdfPage
    {
        $n = $this->pageCount();
        if ($index < 0 || $index >= $n) {
            throw new \OutOfRangeException("page index {$index} out of range [0, {$n})");
        }
        return new PdfPage($this, $index);
    }

    /**
     * @return array<int, PdfPage> all pages (eager)
     */
    public function pages(): array
    {
        $n = $this->pageCount();
        $out = [];
        for ($i = 0; $i < $n; ++$i) {
            $out[] = new PdfPage($this, $i);
        }
        return $out;
    }

    /**
     * Lazy generator over pages. Prefer for large documents.
     *
     * @return \Generator<int, PdfPage>
     */
    public function pagesIter(): \Generator
    {
        $n = $this->pageCount();
        for ($i = 0; $i < $n; ++$i) {
            yield $i => new PdfPage($this, $i);
        }
    }

    // ────────────────────── lifecycle ──────────────────────

    /** @return bool true if the native handle is still live */
    public function isOpen(): bool
    {
        return $this->handle !== null;
    }

    /**
     * Free the native handle. Idempotent — calling more than once is
     * a no-op.
     */
    public function close(): void
    {
        if ($this->handle !== null) {
            $this->bindings->pdfDocumentFree($this->handle);
            $this->handle = null;
        }
        if ($this->ownedTempPath !== null) {
            @unlink($this->ownedTempPath);
            $this->ownedTempPath = null;
        }
    }

    public function __destruct()
    {
        $this->close();
    }

    /**
     * @internal Accessor used by sibling classes (AutoExtractor,
     *           MarkdownConverter, PdfValidator, PdfPage) that need
     *           the raw handle to pass to their own FFI calls. Same
     *           precondition as the private `requireHandle()`.
     */
    public function getHandle(): CData
    {
        return $this->requireHandle();
    }

    /** @internal */
    public function getSourcePath(): ?string
    {
        return $this->sourcePath;
    }

    private function requireHandle(): CData
    {
        if ($this->handle === null) {
            throw new InvalidStateException('PdfDocument has been closed');
        }
        return $this->handle;
    }
}
