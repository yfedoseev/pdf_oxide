<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\Exceptions\InvalidStateException;
use PdfOxide\Exceptions\IoException;
use PdfOxide\PdfDocument;

/**
 * Smoke tests for {@see PdfDocument}, mirroring
 * `java/src/test/java/fyi/oxide/pdf/PdfDocumentTest.java`.
 */
final class PdfDocumentTest extends IntegrationTestCase
{
    public function testOpenAndCloseSimplePdf(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $this->assertTrue($doc->isOpen());
            $this->assertGreaterThan(0, $doc->pageCount());
        } finally {
            $doc->close();
        }
    }

    public function testCloseIsIdempotent(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        $this->assertTrue($doc->isOpen());
        $doc->close();
        $this->assertFalse($doc->isOpen());
        // Second + third close: no exception, no crash.
        $doc->close();
        $doc->close();
        $this->assertFalse($doc->isOpen());
    }

    public function testOperationsOnClosedHandleThrowInvalidState(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        $doc->close();
        $this->expectException(InvalidStateException::class);
        $doc->pageCount();
    }

    public function testNonexistentFileThrowsIoException(): void
    {
        $this->expectException(IoException::class);
        PdfDocument::open('/tmp/__pdf_oxide_does_not_exist__.pdf');
    }

    public function testExtractTextReturnsString(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $text = $doc->extractText(0);
            $this->assertIsString($text);
        } finally {
            $doc->close();
        }
    }

    public function testExtractTextAutoGracefulFallback(): void
    {
        // v0.3.51 contract: extractTextAuto must NEVER throw on the
        // graceful-fallback path; returns whatever text is recoverable.
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $text = $doc->extractTextAuto(0);
            $this->assertIsString($text);
        } finally {
            $doc->close();
        }
    }

    public function testVersionReturnsMajorMinor(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $v = $doc->version();
            $this->assertArrayHasKey('major', $v);
            $this->assertArrayHasKey('minor', $v);
        } finally {
            $doc->close();
        }
    }

    public function testPageIteration(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $pages = $doc->pages();
            $this->assertSameSize(range(0, $doc->pageCount() - 1), $pages);
            $i = 0;
            foreach ($doc->pagesIter() as $idx => $p) {
                $this->assertSame($i, $idx);
                $this->assertSame($i, $p->index());
                ++$i;
            }
        } finally {
            $doc->close();
        }
    }

    public function testStaticExtractTextOnce(): void
    {
        $text = PdfDocument::extractTextOnce($this->fixture('simple.pdf'));
        $this->assertIsString($text);
    }
}
