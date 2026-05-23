<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Unit;

use PdfOxide\AutoExtractResult;
use PdfOxide\Enums\ExtractReason;
use PdfOxide\Enums\PageKind;
use PHPUnit\Framework\TestCase;

/**
 * Unit tests for the {@see AutoExtractResult} value object.
 *
 * No FFI / cdylib dependency — runs in the Unit testsuite on every
 * matrix cell regardless of whether the native library is present.
 */
final class AutoExtractResultTest extends TestCase
{
    public function testConstructorAssignsAllProperties(): void
    {
        $r = new AutoExtractResult(
            text: 'hello',
            reason: ExtractReason::Ok,
            kind: PageKind::TextLayer,
            confidence: 0.98,
            classification: ['kind' => 'text_layer'],
        );

        $this->assertSame('hello', $r->text);
        $this->assertSame(ExtractReason::Ok, $r->reason);
        $this->assertSame(PageKind::TextLayer, $r->kind);
        $this->assertSame(0.98, $r->confidence);
        $this->assertSame(['kind' => 'text_layer'], $r->classification);
    }

    public function testIsOkAcceptsBothCanonicalOkReasons(): void
    {
        $r1 = new AutoExtractResult('', ExtractReason::Ok, PageKind::TextLayer, 1.0, null);
        $r2 = new AutoExtractResult('', ExtractReason::NativeTextHighConfidence, PageKind::TextLayer, 1.0, null);
        $this->assertTrue($r1->isOk());
        $this->assertTrue($r2->isOk());
    }

    public function testIsOkFalseForDegradedReasons(): void
    {
        $r = new AutoExtractResult('', ExtractReason::Empty, PageKind::Empty, 0.0, null);
        $this->assertFalse($r->isOk());
    }

    public function testIsOcrFallbackTrueForBothFallbackReasons(): void
    {
        $r1 = new AutoExtractResult('', ExtractReason::OcrRequestedButUnavailable, PageKind::Scanned, 0.0, null);
        $r2 = new AutoExtractResult('', ExtractReason::OcrLowConfidenceFallback, PageKind::Scanned, 0.3, null);
        $this->assertTrue($r1->isOcrFallback());
        $this->assertTrue($r2->isOcrFallback());
    }

    public function testIsOcrFallbackFalseForNonFallbackReasons(): void
    {
        $r = new AutoExtractResult('', ExtractReason::Ok, PageKind::TextLayer, 1.0, null);
        $this->assertFalse($r->isOcrFallback());
    }
}
