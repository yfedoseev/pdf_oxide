<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\PdfDocument;
use PHPUnit\Framework\TestCase;

/**
 * Open / extract / close round-trip — the load-bearing smoke test for
 * the whole binding. If this fails everything else is moot.
 *
 * @requires extension ffi
 */
final class PdfDocumentTest extends TestCase
{
    protected function setUp(): void
    {
        if (! extension_loaded('ffi')) {
            $this->markTestSkipped('ext-ffi not loaded');
        }
        if (PDF_OXIDE_NATIVE_LIB === null) {
            $this->markTestSkipped('libpdf_oxide cdylib not found.');
        }
        if (PDF_OXIDE_SAMPLE_PDF === null) {
            $this->markTestSkipped('No sample PDF available.');
        }
    }

    public function testOpenAndGetPageCount(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $this->assertTrue($doc->isOpen());
        $count = $doc->getPageCount();
        $this->assertIsInt($count);
        $this->assertGreaterThan(0, $count, 'tiny.pdf must have at least one page.');
    }

    public function testExtractTextProducesNonEmptyOutput(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $text = $doc->extractText(0);
        $this->assertIsString($text);
        // The tiny fixture is the canonical "hello world" — we don't
        // assert the exact text content here (other tests in the Rust
        // suite cover that) but assert non-empty so an extraction
        // regression in `extract_text` would surface.
        $this->assertNotSame('', trim($text), 'extractText should produce non-empty text for tiny.pdf.');
    }

    public function testToMarkdownReturnsString(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $md = $doc->toMarkdown(0);
        $this->assertIsString($md);
        $this->assertNotSame('', trim($md));
    }

    public function testCloseIsIdempotent(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $this->assertTrue($doc->isOpen());
        $doc->close();
        $this->assertFalse($doc->isOpen());
        // Second close should not raise.
        $doc->close();
        $this->assertFalse($doc->isOpen());
    }

    public function testGetFilePathRoundTrip(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $this->assertSame(PDF_OXIDE_SAMPLE_PDF, $doc->getFilePath());
    }
}
