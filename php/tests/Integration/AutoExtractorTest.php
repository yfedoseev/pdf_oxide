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
 * Integration tests for {@see AutoExtractor} (v0.3.51 #519).
 *
 * Each test exercises the real C ABI via FFI. The whole class
 * self-skips when:
 *  - the `ffi` PHP extension isn't loaded (`@requires extension ffi`)
 *  - the cdylib isn't found (`PDF_OXIDE_NATIVE_LIB === null`)
 *  - the tiny test PDF isn't available
 *
 * @requires extension ffi
 */
final class AutoExtractorTest extends TestCase
{
    protected function setUp(): void
    {
        if (! extension_loaded('ffi')) {
            $this->markTestSkipped('ext-ffi not loaded');
        }
        if (PDF_OXIDE_NATIVE_LIB === null) {
            $this->markTestSkipped('libpdf_oxide cdylib not found; build with `cargo build --release`.');
        }
        if (PDF_OXIDE_SAMPLE_PDF === null) {
            $this->markTestSkipped('No sample PDF available for integration tests.');
        }
    }

    public function testEnumsCarryCanonicalWireTokens(): void
    {
        // FROZEN per the v0.3.51 cross-binding contract.
        $this->assertSame('ok', ExtractReason::Ok->value);
        $this->assertSame('ocr_requested_but_unavailable', ExtractReason::OcrRequestedButUnavailable->value);
        $this->assertSame('text_layer', PageKind::TextLayer->value);

        // fromWire is tolerant of unknown values.
        $this->assertSame(ExtractReason::Ok, ExtractReason::fromWire('not_a_real_reason'));
        $this->assertSame(ExtractReason::Ok, ExtractReason::fromWire(null));
    }

    public function testClassifyPageReturnsTypedResult(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $auto = $doc->auto();
        $this->assertInstanceOf(AutoExtractor::class, $auto);

        $result = $auto->classifyPage($doc, 0);
        $this->assertInstanceOf(AutoExtractResult::class, $result);
        $this->assertInstanceOf(PageKind::class, $result->kind);
        $this->assertInstanceOf(ExtractReason::class, $result->reason);
        // The tiny "Hello, world" fixture is a text-layer PDF — assert
        // the classifier doesn't falsely flag it as scanned.
        $this->assertNotSame(PageKind::Scanned, $result->kind);
    }

    public function testExtractTextAutoReturnsString(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $result = $doc->auto()->extractText($doc, 0);
        $this->assertInstanceOf(AutoExtractResult::class, $result);
        $this->assertIsString($result->text);
        // The simple.pdf fixture has a non-empty text layer; if this
        // ever fails we want to know about the regression.
        $this->assertNotSame('', trim($result->text), 'AutoExtractor should return non-empty text for the tiny fixture.');
    }

    public function testPrefetchAvailableReturnsBoolean(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $auto = $doc->auto();
        // Whether or not the build has OCR, this MUST be a bool —
        // callers branch on it to decide graceful fallback paths.
        $this->assertIsBool($auto->prefetchAvailable());
    }
}
