<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\AutoExtractor;
use PdfOxide\AutoExtractResult;
use PdfOxide\PdfDocument;

/**
 * Smoke tests for {@see AutoExtractor}.
 */
final class AutoExtractorTest extends IntegrationTestCase
{
    public function testExtractTextConcatenatesPages(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $ex = AutoExtractor::of($doc);
            $all = $ex->extractText();
            $this->assertIsString($all);
        } finally {
            $doc->close();
        }
    }

    public function testExtractTextForPageReturnsString(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $ex = AutoExtractor::of($doc);
            $this->assertIsString($ex->extractTextForPage(0));
        } finally {
            $doc->close();
        }
    }

    public function testExtractAutoPageReturnsAutoExtractResult(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $ex = AutoExtractor::of($doc);
            $r = $ex->extractAutoPage(0);
            $this->assertInstanceOf(AutoExtractResult::class, $r);
            $this->assertSame(AutoExtractResult::REASON_OK, $r->reason);
        } finally {
            $doc->close();
        }
    }

    public function testExtractAutoDocumentReturnsResult(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $ex = AutoExtractor::of($doc);
            $r = $ex->extractAutoDocument();
            $this->assertInstanceOf(AutoExtractResult::class, $r);
        } finally {
            $doc->close();
        }
    }

    public function testClassifyPageKindReturnsToken(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $ex = AutoExtractor::of($doc);
            $kind = $ex->classifyPageKind(0);
            $this->assertIsString($kind);
            $this->assertContains($kind, [
                AutoExtractResult::KIND_TEXT_LAYER,
                AutoExtractResult::KIND_SCANNED,
                AutoExtractResult::KIND_IMAGE_TEXT,
                AutoExtractResult::KIND_MIXED,
                AutoExtractResult::KIND_EMPTY,
            ]);
        } finally {
            $doc->close();
        }
    }

    public function testOutOfRangePageThrows(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $ex = AutoExtractor::of($doc);
            $this->expectException(\OutOfRangeException::class);
            $ex->extractTextForPage(9999);
        } finally {
            $doc->close();
        }
    }

    public function testPresetFactories(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $this->assertSame(AutoExtractor::MODE_TEXT_ONLY, AutoExtractor::fast($doc)->mode());
            $this->assertSame(AutoExtractor::MODE_AUTO, AutoExtractor::balanced($doc)->mode());
            $this->assertSame(AutoExtractor::MODE_FORCE_OCR, AutoExtractor::highFidelity($doc)->mode());
        } finally {
            $doc->close();
        }
    }
}
