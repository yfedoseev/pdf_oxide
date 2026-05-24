<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide;

use PdfOxide\FFI\FunctionBindings;

/**
 * A page within a {@see PdfDocument}, identified by its 0-based page
 * index.
 *
 * Mirrors `fyi.oxide.pdf.PdfPage`. {@code PdfPage} is a lightweight
 * view — it holds no native handle of its own; it borrows from its
 * parent. Calls on a closed parent document throw
 * {@see \PdfOxide\Exceptions\InvalidStateException}.
 *
 * Construct via {@see PdfDocument::page()} or {@see PdfDocument::pages()}.
 */
final class PdfPage
{
    private readonly FunctionBindings $bindings;

    /**
     * @internal package-private — call via {@see PdfDocument::page()}.
     */
    public function __construct(
        private readonly PdfDocument $parent,
        private readonly int $index,
    ) {
        $this->bindings = new FunctionBindings();
    }

    public function parent(): PdfDocument
    {
        return $this->parent;
    }

    public function index(): int
    {
        return $this->index;
    }

    /**
     * Extract this page's text. Equivalent to
     * {@see PdfDocument::extractText()} on the parent.
     */
    public function text(): string
    {
        return $this->parent->extractText($this->index);
    }

    /**
     * Auto-routed extraction for this page. Equivalent to
     * {@see PdfDocument::extractTextAuto()} on the parent.
     */
    public function textAuto(): string
    {
        return $this->parent->extractTextAuto($this->index);
    }

    /** Per-page Markdown conversion. */
    public function toMarkdown(): string
    {
        return MarkdownConverter::toMarkdown($this->parent, $this->index);
    }

    /** Per-page HTML conversion. */
    public function toHtml(): string
    {
        return MarkdownConverter::toHtml($this->parent, $this->index);
    }

    /**
     * v0.3.55 limitation: per-page MediaBox / CropBox accessors are
     * NOT YET exposed on the read-only {@see PdfDocument} C ABI. The
     * {@see DocumentEditor} surface exposes them (via
     * `document_editor_get_page_media_box`); to read a box, open the
     * PDF with {@see DocumentEditor::open()} instead. Mirrors the
     * Java binding's v0.3.53 `cropBox() → mediaBox()` follow-up note.
     *
     * Tracked: a future minor release will add `pdf_document_get_page_
     * media_box` to the C ABI.
     */
    public function mediaBoxNotYetSupported(): void
    {
        throw new \BadMethodCallException(
            'PdfPage::mediaBox() — pdf_oxide v0.3.55 has no read-side C ABI for page boxes; use DocumentEditor instead.'
        );
    }

    public function __toString(): string
    {
        return "PdfPage[index={$this->index}]";
    }
}
