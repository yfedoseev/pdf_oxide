<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\AutoExtractor;
use PdfOxide\AutoExtractResult;
use PdfOxide\Enums\ExtractReason;
use PdfOxide\Enums\PageKind;
use PdfOxide\PdfDocument;
use PHPUnit\Framework\TestCase;

/**
 * Smoke test for {@see AutoExtractor} (v0.3.51 #519).
 *
 * Skipped automatically when the native library isn't present. Asserts
 * that the typed-reason envelope round-trips successfully and that
 * `prefetchAvailable()` returns a boolean (the build-feature
 * indicator). No assumption is made about whether OCR is compiled in
 * (the build matrix exercises both).
 */
final class AutoExtractorTest extends TestCase
{
    protected function setUp(): void
    {
        if (PDF_OXIDE_NATIVE_LIB === null) {
            $this->markTestSkipped('libpdf_oxide not built; run `cargo build --release` first.');
        }
        if (PDF_OXIDE_SAMPLE_PDF === null) {
            $this->markTestSkipped('No sample PDF available for integration tests.');
        }
    }

    public function testEnumsCarryCanonicalWireTokens(): void
    {
        // These values are FROZEN per the v0.3.51 cross-binding contract.
        $this->assertSame('ok', ExtractReason::Ok->value);
        $this->assertSame('ocr_requested_but_unavailable', ExtractReason::OcrRequestedButUnavailable->value);
        $this->assertSame('text_layer', PageKind::TextLayer->value);

        // fromWire is tolerant of unknown values.
        $this->assertSame(ExtractReason::Ok, ExtractReason::fromWire('not_a_real_reason'));
        $this->assertSame(ExtractReason::Ok, ExtractReason::fromWire(null));
    }

    public function testClassifyPageReturnsResult(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $auto = $doc->auto();
        $this->assertInstanceOf(AutoExtractor::class, $auto);

        $result = $auto->classifyPage($doc, 0);
        $this->assertInstanceOf(AutoExtractResult::class, $result);
        $this->assertInstanceOf(PageKind::class, $result->kind);
        $this->assertInstanceOf(ExtractReason::class, $result->reason);
    }

    public function testExtractTextAutoRoundTrips(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $result = $doc->auto()->extractText($doc, 0);
        $this->assertInstanceOf(AutoExtractResult::class, $result);
        // Text may be empty if the fixture is image-only, but it must be a string.
        $this->assertIsString($result->text);
    }

    public function testPrefetchAvailableReturnsBoolean(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $auto = $doc->auto();
        $this->assertIsBool($auto->prefetchAvailable());
    }
}
