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
use PdfOxide\Exceptions\PdfException;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\FFI\NativeLibrary;
use PdfOxide\FFI\StringMarshaller;

/**
 * Write-side counterpart to {@see PdfDocument}: form-fill, destructive
 * redaction (v0.3.50 #231), metadata scrubbing, and save.
 *
 * Mirrors `fyi.oxide.pdf.DocumentEditor` from the Java binding. Owns a
 * native `DocumentEditor*` handle. {@see close()} is idempotent; rely
 * on `__destruct()` for best-effort cleanup.
 *
 * NOT thread-safe; one editor per worker.
 */
final class DocumentEditor
{
    private ?CData $handle = null;

    private readonly FunctionBindings $bindings;

    private function __construct(CData $handle)
    {
        $this->bindings = new FunctionBindings();
        $this->handle = $handle;
    }

    /**
     * Open a PDF for editing.
     *
     * @throws IoException when the file is missing
     */
    public static function open(string $path): self
    {
        if (! is_file($path)) {
            throw new IoException("PDF file not found: {$path}", ['file' => $path]);
        }
        $bindings = new FunctionBindings();
        // Use the corrected wrapper (`documentEditorOpen`, NOT
        // `pdfDocumentEditorOpen` which targets a nonexistent symbol).
        $handle = $bindings->documentEditorOpen($path);
        if ($handle === null) {
            throw new IoException("Failed to open PDF for editing: {$path}");
        }
        return new self($handle);
    }

    // ─────────────── destructive redaction ─────────────────

    /**
     * Queue a redaction rectangle on a page. The redaction is not
     * applied until {@see applyRedactionsDestructive()} runs.
     *
     * Coordinates are PDF user-space (points).
     */
    public function addRedaction(int $pageIndex, float $x1, float $y1, float $x2, float $y2): self
    {
        $this->bindings->pdfRedactionAdd($this->requireHandle(), $pageIndex, $x1, $y1, $x2, $y2);
        return $this;
    }

    /** Number of pending redactions queued for `$pageIndex`. */
    public function redactionCount(int $pageIndex): int
    {
        return $this->bindings->pdfRedactionCount($this->requireHandle(), $pageIndex);
    }

    /**
     * Apply all queued redactions destructively (v0.3.50 #231). Scrubs
     * document metadata by default; the Rust core fail-closes on
     * composite / Type0 / unknown-font pages.
     *
     * @return int the count of regions applied
     */
    public function applyRedactionsDestructive(bool $scrubMetadata = true): int
    {
        return $this->bindings->pdfRedactionApply($this->requireHandle(), $scrubMetadata);
    }

    /** Scrub document metadata (Info dict, XMP, PieceInfo). */
    public function scrubMetadata(): self
    {
        $this->bindings->pdfRedactionScrubMetadata($this->requireHandle());
        return $this;
    }

    // ─────────────── metadata accessors ────────────────────

    /** @return string the `/Producer` Info-dict entry (empty when absent). */
    public function getProducer(): string
    {
        $ffi = NativeLibrary::getInstance();
        $errorCode = \FFI::new('int32_t');
        $cStr = $ffi->document_editor_get_producer($this->requireHandle(), \FFI::addr($errorCode));
        if ($cStr === null || (int) $errorCode->cdata !== 0) {
            return '';
        }
        $out = \FFI::string($cStr);
        $ffi->free_string($cStr);
        return $out;
    }

    /** Set the `/Producer` Info-dict entry. */
    public function setProducer(string $producer): self
    {
        $ffi = NativeLibrary::getInstance();
        $cStr = StringMarshaller::toCString($producer);
        $errorCode = \FFI::new('int32_t');
        try {
            $rc = (int) $ffi->document_editor_set_producer($this->requireHandle(), $cStr, \FFI::addr($errorCode));
            if ($rc !== 0 || (int) $errorCode->cdata !== 0) {
                throw new PdfException('Failed to set producer', 'EDITOR_SET_PRODUCER_FAILED');
            }
        } finally {
            unset($cStr);
        }
        return $this;
    }

    /** @return array{major:int, minor:int} the PDF version. */
    public function version(): array
    {
        $ffi = NativeLibrary::getInstance();
        $major = \FFI::new('uint8_t');
        $minor = \FFI::new('uint8_t');
        $ffi->document_editor_get_version($this->requireHandle(), \FFI::addr($major), \FFI::addr($minor));
        return ['major' => (int) $major->cdata, 'minor' => (int) $minor->cdata];
    }

    public function pageCount(): int
    {
        $ffi = NativeLibrary::getInstance();
        $errorCode = \FFI::new('int32_t');
        return (int) $ffi->document_editor_get_page_count($this->requireHandle(), \FFI::addr($errorCode));
    }

    public function isModified(): bool
    {
        $ffi = NativeLibrary::getInstance();
        return (bool) $ffi->document_editor_is_modified($this->requireHandle());
    }

    /** @return string the path the editor was opened with. */
    public function sourcePath(): string
    {
        $ffi = NativeLibrary::getInstance();
        $errorCode = \FFI::new('int32_t');
        $cStr = $ffi->document_editor_get_source_path($this->requireHandle(), \FFI::addr($errorCode));
        if ($cStr === null) {
            return '';
        }
        $out = \FFI::string($cStr);
        $ffi->free_string($cStr);
        return $out;
    }

    // ─────────────────────── save ──────────────────────────

    /** Save the edited PDF to `$path`. */
    public function saveTo(string $path): void
    {
        $ffi = NativeLibrary::getInstance();
        $cPath = StringMarshaller::toCString($path);
        $errorCode = \FFI::new('int32_t');
        try {
            $rc = (int) $ffi->document_editor_save($this->requireHandle(), $cPath, \FFI::addr($errorCode));
            if ($rc !== 0 || (int) $errorCode->cdata !== 0) {
                throw new IoException("Failed to save editor to {$path}");
            }
        } finally {
            unset($cPath);
        }
    }

    /** @return string the edited PDF as bytes. */
    public function save(): string
    {
        $ffi = NativeLibrary::getInstance();
        // C signature: document_editor_save_to_bytes(handle, uint64_t* len, int32_t* err)
        $dataLen = \FFI::new('uint64_t');
        $errorCode = \FFI::new('int32_t');
        $ptr = $ffi->document_editor_save_to_bytes(
            $this->requireHandle(),
            \FFI::addr($dataLen),
            \FFI::addr($errorCode),
        );
        if ((int) $errorCode->cdata !== 0 || $ptr === null) {
            throw new PdfException('document_editor_save_to_bytes failed', 'EDITOR_SAVE_FAILED');
        }
        $bytes = \FFI::string($ptr, (int) $dataLen->cdata);
        $ffi->free_bytes($ptr);
        return $bytes;
    }

    // ─────────────────── lifecycle ─────────────────────────

    public function isOpen(): bool
    {
        return $this->handle !== null;
    }

    public function close(): void
    {
        if ($this->handle !== null) {
            $this->bindings->documentEditorFree($this->handle);
            $this->handle = null;
        }
    }

    public function __destruct()
    {
        $this->close();
    }

    /** @internal */
    public function getHandle(): CData
    {
        return $this->requireHandle();
    }

    private function requireHandle(): CData
    {
        if ($this->handle === null) {
            throw new InvalidStateException('DocumentEditor has been closed');
        }
        return $this->handle;
    }
}
