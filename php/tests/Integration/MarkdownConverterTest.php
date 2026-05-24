<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\MarkdownConverter;
use PdfOxide\PdfDocument;

/**
 * Smoke tests for {@see MarkdownConverter}, mirroring
 * `java/src/test/java/fyi/oxide/pdf/MarkdownConverterTest.java`.
 */
final class MarkdownConverterTest extends IntegrationTestCase
{
    public function testToMarkdownReturnsString(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $md = MarkdownConverter::toMarkdown($doc, 0);
            $this->assertIsString($md);
        } finally {
            $doc->close();
        }
    }

    public function testToMarkdownAllReturnsString(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $md = MarkdownConverter::toMarkdownAll($doc);
            $this->assertIsString($md);
        } finally {
            $doc->close();
        }
    }

    public function testToHtmlReturnsString(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $html = MarkdownConverter::toHtml($doc, 0);
            $this->assertIsString($html);
        } finally {
            $doc->close();
        }
    }

    public function testDocConvenienceMatchesStaticConverter(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $this->assertSame(MarkdownConverter::toMarkdown($doc, 0), $doc->toMarkdown(0));
            $this->assertSame(MarkdownConverter::toHtml($doc, 0), $doc->toHtml(0));
        } finally {
            $doc->close();
        }
    }
}
