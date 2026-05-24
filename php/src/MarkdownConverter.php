<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide;

use PdfOxide\FFI\FunctionBindings;

/**
 * Static converters from a {@see PdfDocument} to Markdown or HTML.
 *
 * Mirrors `fyi.oxide.pdf.MarkdownConverter`. Stateless façade — the
 * methods are static; the underlying Rust call takes a borrowed
 * `&PdfDocument` and is single-threaded against the same document
 * (see `docs/architecture/00-common-foundation.md` §2.7).
 *
 * v0.3.55 surface ships per-page and whole-document converters with
 * default conversion options. Tunable options (table-extraction
 * toggle, image-embedding mode, heading inference) come in a follow-up.
 */
final class MarkdownConverter
{
    /** Static-only. */
    private function __construct() {}

    /** Convert a single page to Markdown. */
    public static function toMarkdown(PdfDocument $doc, int $pageIndex): string
    {
        $bindings = new FunctionBindings();
        return $bindings->pdfDocumentToMarkdown($doc->getHandle(), $pageIndex);
    }

    /** Convert the entire document to Markdown. */
    public static function toMarkdownAll(PdfDocument $doc): string
    {
        $bindings = new FunctionBindings();
        return $bindings->pdfDocumentToMarkdownAll($doc->getHandle());
    }

    /** Convert a single page to HTML. */
    public static function toHtml(PdfDocument $doc, int $pageIndex): string
    {
        $bindings = new FunctionBindings();
        return $bindings->pdfDocumentToHtml($doc->getHandle(), $pageIndex);
    }

    /**
     * Convert a single page to plain text (no formatting). Distinct
     * from {@see PdfDocument::extractText()} which returns the raw
     * reading-order text — `toPlainText()` honors structure-tree
     * paragraph boundaries when present.
     */
    public static function toPlainText(PdfDocument $doc, int $pageIndex): string
    {
        $bindings = new FunctionBindings();
        return $bindings->pdfDocumentToPlainText($doc->getHandle(), $pageIndex);
    }
}
