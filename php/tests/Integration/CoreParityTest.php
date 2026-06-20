<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\Exceptions\IoException;
use PdfOxide\Pdf;
use PdfOxide\PdfDocument;

/**
 * Core functional test-parity suite (PHP) — mirrors the shared cross-language
 * spec (docs/releases/plans/v0.3.61/core-test-parity-spec.md) with the
 * idiomatic PHP API. Every binding asserts the same behaviors. (Search has no
 * single PHP-core method, matching the Rust reference, so it is omitted here.)
 */
final class CoreParityTest extends IntegrationTestCase
{
    public function testOpenAndPageCount(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $this->assertSame(1, $doc->pageCount());
        } finally {
            $doc->close();
        }
    }

    public function testExtractTextReturnsString(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $this->assertIsString($doc->extractText(0));
        } finally {
            $doc->close();
        }
    }

    public function testConvertMarkdownAndHtmlReturnStrings(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $this->assertIsString($doc->toMarkdown(0));
            $this->assertIsString($doc->toHtml(0));
        } finally {
            $doc->close();
        }
    }

    public function testStructuredExtraction(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $this->assertIsArray($doc->extractStructured(0));
        } finally {
            $doc->close();
        }
    }

    public function testCreatePdfFromText(): void
    {
        $pdf = Pdf::fromText('Core parity across all bindings.');
        try {
            $bytes = $pdf->save();
            $this->assertSame('%PDF-', substr($bytes, 0, 5));
        } finally {
            $pdf->close();
        }
    }

    public function testOpenFromBytes(): void
    {
        $bytes = (string) file_get_contents($this->fixture('simple.pdf'));
        $doc = PdfDocument::openBytes($bytes);
        try {
            $this->assertSame(1, $doc->pageCount());
        } finally {
            $doc->close();
        }
    }

    public function testOpeningMissingPathThrows(): void
    {
        $this->expectException(IoException::class);
        PdfDocument::open('/no/such/file/does/not/exist.pdf');
    }

    public function testVersionConstant(): void
    {
        $this->assertSame('0.3.67', Pdf::VERSION);
    }
}
