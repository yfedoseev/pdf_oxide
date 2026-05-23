<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\Managers\OfficeConverter;
use PdfOxide\PdfDocument;
use PHPUnit\Framework\TestCase;

/**
 * Smoke test for v0.3.48 #159 Office converter.
 *
 * We do NOT verify a full DOCX import (the fixture set doesn't include
 * one); we DO verify the export-to-DOCX path on the sample PDF, which
 * exercises the same FFI surface in reverse.
 */
final class OfficeConverterTest extends TestCase
{
    protected function setUp(): void
    {
        if (PDF_OXIDE_NATIVE_LIB === null) {
            $this->markTestSkipped('libpdf_oxide not built.');
        }
        if (PDF_OXIDE_SAMPLE_PDF === null) {
            $this->markTestSkipped('No sample PDF available.');
        }
    }

    public function testExportToDocxReturnsBytes(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $converter = new OfficeConverter($doc->getHandle());
        $this->assertInstanceOf(OfficeConverter::class, $converter);

        $bytes = $converter->toDocx();
        $this->assertIsString($bytes);
        $this->assertNotEmpty($bytes, 'DOCX export should produce non-empty bytes.');
        // DOCX files are ZIP archives; first two bytes are 'PK'.
        $this->assertSame('PK', substr($bytes, 0, 2), 'Output should be a ZIP-shaped (DOCX) byte stream.');
    }
}
