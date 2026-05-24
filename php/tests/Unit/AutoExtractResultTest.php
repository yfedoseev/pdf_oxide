<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide\Tests\Unit;

use PdfOxide\AutoExtractResult;
use PHPUnit\Framework\TestCase;

/**
 * Pure-PHP value-object tests for AutoExtractResult — no native lib
 * needed, run in the Unit suite.
 */
final class AutoExtractResultTest extends TestCase
{
    public function testDefaultsAreOkAndConfidentAndNotOcr(): void
    {
        $r = new AutoExtractResult(text: 'hello');
        $this->assertSame('hello', $r->text);
        $this->assertSame(AutoExtractResult::REASON_OK, $r->reason);
        $this->assertSame(1.0, $r->confidence);
        $this->assertFalse($r->ocrUsed);
        $this->assertSame([], $r->regions);
        $this->assertSame([], $r->pagesNeedingOcr);
    }

    public function testIsOkForOkAndHighConfidence(): void
    {
        $okR = new AutoExtractResult(text: '', reason: AutoExtractResult::REASON_OK);
        $hi = new AutoExtractResult(
            text: '',
            reason: AutoExtractResult::REASON_NATIVE_TEXT_HIGH_CONFIDENCE
        );
        $this->assertTrue($okR->isOk());
        $this->assertTrue($hi->isOk());
    }

    public function testIsOcrFallbackForBothFallbackReasons(): void
    {
        $unavail = new AutoExtractResult(
            text: '',
            reason: AutoExtractResult::REASON_OCR_REQUESTED_BUT_UNAVAILABLE
        );
        $lowconf = new AutoExtractResult(
            text: '',
            reason: AutoExtractResult::REASON_OCR_LOW_CONFIDENCE_FALLBACK
        );
        $this->assertTrue($unavail->isOcrFallback());
        $this->assertTrue($lowconf->isOcrFallback());
    }

    public function testEmptyIsNotOkAndNotFallback(): void
    {
        $empty = new AutoExtractResult(text: '', reason: AutoExtractResult::REASON_EMPTY);
        $this->assertFalse($empty->isOk());
        $this->assertFalse($empty->isOcrFallback());
    }

    public function testKindAndReasonConstantsAreSnakeCaseStrings(): void
    {
        // Wire tokens are frozen — guard rename regressions.
        $this->assertSame('ok', AutoExtractResult::REASON_OK);
        $this->assertSame('ocr_requested_but_unavailable', AutoExtractResult::REASON_OCR_REQUESTED_BUT_UNAVAILABLE);
        $this->assertSame('text_layer', AutoExtractResult::KIND_TEXT_LAYER);
        $this->assertSame('scanned', AutoExtractResult::KIND_SCANNED);
    }
}
