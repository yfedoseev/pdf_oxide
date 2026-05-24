<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\Exceptions\InvalidStateException;
use PdfOxide\Pdf;

/**
 * Smoke tests for the {@see Pdf} factory, mirroring
 * `java/src/test/java/fyi/oxide/pdf/PdfCreationTest.java`.
 */
final class PdfTest extends IntegrationTestCase
{
    public function testFromMarkdownProducesValidPdf(): void
    {
        $md = "# Hello\n\nThis is **bold** text and *italic* text.\n";
        $pdf = Pdf::fromMarkdown($md);
        try {
            $bytes = $pdf->save();
            $this->assertNotEmpty($bytes);
            // Every valid PDF starts with %PDF-
            $this->assertSame('%PDF-', substr($bytes, 0, 5));
        } finally {
            $pdf->close();
        }
    }

    public function testFromHtmlProducesValidPdf(): void
    {
        $html = '<html><body><h1>Hi</h1><p>HTML content</p></body></html>';
        $pdf = Pdf::fromHtml($html);
        try {
            $bytes = $pdf->save();
            $this->assertNotEmpty($bytes);
            $this->assertSame('%PDF-', substr($bytes, 0, 5));
        } finally {
            $pdf->close();
        }
    }

    public function testSaveToWritesFile(): void
    {
        $tmp = tempnam(sys_get_temp_dir(), 'pdf_oxide_save_');
        $this->assertNotFalse($tmp);
        try {
            $pdf = Pdf::fromMarkdown("# T\n\nContent.\n");
            try {
                $pdf->saveTo($tmp);
            } finally {
                $pdf->close();
            }
            $this->assertGreaterThan(0, filesize($tmp));
            $header = file_get_contents($tmp);
            $this->assertSame('%PDF-', substr($header, 0, 5));
        } finally {
            @unlink($tmp);
        }
    }

    public function testSaveAfterCloseThrows(): void
    {
        $pdf = Pdf::fromMarkdown('# X');
        $pdf->close();
        $this->assertFalse($pdf->isOpen());
        $this->expectException(InvalidStateException::class);
        $pdf->save();
    }

    public function testCloseIsIdempotent(): void
    {
        $pdf = Pdf::fromMarkdown('# X');
        $pdf->close();
        $pdf->close();
        $pdf->close();
        $this->assertFalse($pdf->isOpen());
    }

    public function testVersionStaticReturnsSemver(): void
    {
        $v = Pdf::version();
        $this->assertNotEmpty($v);
    }

    public function testPrefetchAvailableReturnsBool(): void
    {
        // Just exercises the static — doesn't assert true (depends on
        // whether the cdylib was built with the `ocr` feature).
        $this->assertIsBool(Pdf::prefetchAvailable());
    }
}
