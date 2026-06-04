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
use PdfOxide\Exceptions\ParseException;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\FFI\NativeLibrary;

/**
 * PDF creation / transformation factory.
 *
 * Mirrors `fyi.oxide.pdf.Pdf` from the Java binding. Read-side
 * concerns live on {@see PdfDocument}; mutate concerns on
 * {@see DocumentEditor}; creation + transformation (markdown→PDF,
 * html→PDF) lives here.
 *
 * Lifecycle: idempotent {@see close()}; rely on `__destruct()` for
 * best-effort cleanup. Not thread-safe.
 */
final class Pdf
{
    private ?CData $handle = null;

    private readonly FunctionBindings $bindings;

    private function __construct(CData $handle)
    {
        $this->bindings = new FunctionBindings();
        $this->handle = $handle;
    }

    // ────────────────────── factories ──────────────────────

    /**
     * Create a PDF from a Markdown source. Heading levels,
     * bold/italic, monospace code, lists, links, and inline images
     * (data: URIs) are rendered per pdf_oxide's markdown pipeline
     * (v0.3.52 markdown→PDF styling restored, #525).
     *
     * @throws ParseException when markdown cannot be parsed
     */
    public static function fromMarkdown(string $markdown): self
    {
        $bindings = new FunctionBindings();
        $handle = $bindings->pdfFromMarkdown($markdown);
        if ($handle === null) {
            throw new ParseException('Failed to create PDF from Markdown');
        }
        return new self($handle);
    }

    /**
     * Create a PDF from an HTML source.
     *
     * @throws ParseException when HTML cannot be parsed
     */
    public static function fromHtml(string $html): self
    {
        $bindings = new FunctionBindings();
        $handle = $bindings->pdfFromHtml($html);
        if ($handle === null) {
            throw new ParseException('Failed to create PDF from HTML');
        }
        return new self($handle);
    }

    /**
     * Create a PDF from a plain-text source.
     *
     * @throws ParseException when text cannot be converted
     */
    public static function fromText(string $text): self
    {
        $bindings = new FunctionBindings();
        $handle = $bindings->pdfFromText($text);
        if ($handle === null) {
            throw new ParseException('Failed to create PDF from text');
        }
        return new self($handle);
    }

    // ─────────────────────── output ────────────────────────

    /**
     * @return string the generated PDF bytes
     * @throws InvalidStateException when this object has been closed
     */
    public function save(): string
    {
        // Direct FFI call — the C signature is
        //   uint8_t *pdf_save_to_bytes(Pdf*, int32_t *data_len, int32_t *error_code)
        // (the FunctionBindings wrapper targets a different signature
        // and isn't usable for the Pdf* handle path).
        $ffi = NativeLibrary::getInstance();
        $dataLen = $ffi->new('int32_t');
        $errorCode = $ffi->new('int32_t');
        $ptr = $ffi->pdf_save_to_bytes($this->requireHandle(), \FFI::addr($dataLen), \FFI::addr($errorCode));
        if ((int) $errorCode->cdata !== 0 || $ptr === null) {
            throw new \PdfOxide\Exceptions\PdfException(
                'pdf_save_to_bytes failed',
                'PDF_SAVE_FAILED',
                ['error_code' => (int) $errorCode->cdata]
            );
        }
        $len = (int) $dataLen->cdata;
        $bytes = \FFI::string($ptr, $len);
        // pdf_oxide hands ownership to the caller — release the buffer.
        $ffi->free_bytes($ptr);
        return $bytes;
    }

    /**
     * Write the generated PDF to a path.
     *
     * @throws IoException on filesystem failures
     * @throws InvalidStateException when this object has been closed
     */
    public function saveTo(string $path): void
    {
        $bytes = $this->save();
        if (file_put_contents($path, $bytes) === false) {
            throw new IoException("Failed to write PDF to {$path}");
        }
    }

    // ─────────────────── library helpers ───────────────────

    /**
     * @return string the pdf_oxide library version (e.g. "0.3.55").
     *
     * Sourced from {@see VERSION} (release tooling keeps this constant
     * in sync with `Cargo.toml`'s `version` field). The cdylib doesn't
     * yet export a runtime `pdf_oxide_version()` symbol; until it does,
     * this is the canonical accessor.
     */
    public static function version(): string
    {
        return self::VERSION;
    }

    /**
     * pdf_oxide library version. Kept in sync with `Cargo.toml` by the
     * release tooling (see `docs/releases/RELEASE_PROCESS.md`).
     */
    public const VERSION = '0.3.60';

    /** Whether OCR-model prefetch + cache are available on this build. */
    public static function prefetchAvailable(): bool
    {
        $bindings = new FunctionBindings();
        return $bindings->pdfOxidePrefetchAvailable();
    }

    /**
     * Prefetch OCR models for the supplied IETF language tags
     * (e.g. `['eng', 'rus']`). Returns the cache directory; empty
     * string when OCR is not available on this build.
     *
     * @param array<int,string> $languages
     */
    public static function prefetchModels(array $languages): string
    {
        $bindings = new FunctionBindings();
        return $bindings->pdfOxidePrefetchModels(implode(',', $languages));
    }

    // ─────────────────────── lifecycle ─────────────────────

    public function isOpen(): bool
    {
        return $this->handle !== null;
    }

    public function close(): void
    {
        if ($this->handle !== null) {
            // Pdf* has its own freer (NOT pdf_document_free, which takes
            // a PdfDocument*).
            NativeLibrary::getInstance()->pdf_free($this->handle);
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
            throw new InvalidStateException('Pdf has been closed');
        }
        return $this->handle;
    }
}
