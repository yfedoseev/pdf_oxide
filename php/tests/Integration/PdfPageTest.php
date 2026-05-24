<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\PdfDocument;

/**
 * Smoke tests for {@see \PdfOxide\PdfPage}, mirroring
 * `java/src/test/java/fyi/oxide/pdf/PdfPageTest.java`.
 */
final class PdfPageTest extends IntegrationTestCase
{
    public function testPageAccessAndIndex(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $page = $doc->page(0);
            $this->assertSame(0, $page->index());
            $this->assertSame($doc, $page->parent());
        } finally {
            $doc->close();
        }
    }

    public function testMediaBoxThrowsUntilCAbiLands(): void
    {
        // v0.3.55 limitation: no read-side C ABI for page boxes —
        // `mediaBoxNotYetSupported()` throws BadMethodCall to make the
        // limitation explicit. Tracked for a follow-up.
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $page = $doc->page(0);
            $this->expectException(\BadMethodCallException::class);
            $page->mediaBoxNotYetSupported();
        } finally {
            $doc->close();
        }
    }

    public function testPageTextDelegatesToDocument(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $page = $doc->page(0);
            $this->assertSame($doc->extractText(0), $page->text());
        } finally {
            $doc->close();
        }
    }

    public function testOutOfRangeIndexThrows(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $this->expectException(\OutOfRangeException::class);
            $doc->page(9999);
        } finally {
            $doc->close();
        }
    }
}
