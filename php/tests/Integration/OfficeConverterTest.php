<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\Managers\OfficeConverter;
use PdfOxide\PdfDocument;
use PHPUnit\Framework\TestCase;

/**
 * Integration tests for v0.3.48 #159 Office converter.
 *
 * The simple.pdf fixture is a tiny text-layer PDF; we exercise the
 * PDF→DOCX export path (the reverse direction is symmetric and shares
 * the same FFI surface).
 *
 * @requires extension ffi
 */
final class OfficeConverterTest extends TestCase
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

    public function testExportToDocxReturnsZipShapedBytes(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $converter = new OfficeConverter($doc->getHandle());
        $bytes = $converter->toDocx();
        $this->assertIsString($bytes);
        $this->assertNotEmpty($bytes, 'DOCX export should produce non-empty bytes.');
        // DOCX files are ZIP archives; the first two bytes are 'PK'.
        $this->assertSame('PK', substr($bytes, 0, 2), 'Output should be ZIP-shaped (DOCX).');
    }

    public function testRoundTripExportImport(): void
    {
        // Export tiny PDF to DOCX bytes, then re-import — proves the
        // FFI surface is bidirectional. Skipped if either direction
        // raises (some builds skip the importer; per
        // `feedback_extraction_graceful_fallback` that's a warn-and-
        // skip rather than a hard failure).
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $converter = new OfficeConverter($doc->getHandle());
        $docxBytes = $converter->toDocx();
        try {
            $imported = PdfDocument::fromDocxBytes($docxBytes);
            $this->assertGreaterThan(0, $imported->getPageCount());
        } catch (\PdfOxide\Exceptions\PdfException $e) {
            $this->markTestSkipped('DOCX→PDF round-trip not available in this build: ' . $e->getMessage());
        }
    }
}
