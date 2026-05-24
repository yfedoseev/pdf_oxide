<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\PdfDocument;
use PdfOxide\PdfValidator;

/**
 * Smoke tests for {@see PdfValidator}, mirroring
 * `java/src/test/java/fyi/oxide/pdf/PdfValidatorTest.java`.
 */
final class PdfValidatorTest extends IntegrationTestCase
{
    public function testIsPdfAReturnsBoolean(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $this->assertIsBool(PdfValidator::isPdfA($doc, PdfValidator::PDFA_1B));
        } finally {
            $doc->close();
        }
    }

    public function testIsPdfUaReturnsBoolean(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $this->assertIsBool(PdfValidator::isPdfUa($doc));
        } finally {
            $doc->close();
        }
    }

    public function testValidatePdfAReturnsStructured(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $r = PdfValidator::validatePdfA($doc, PdfValidator::PDFA_1B);
            $this->assertArrayHasKey('compliant', $r);
            $this->assertArrayHasKey('errorCount', $r);
            $this->assertArrayHasKey('warningCount', $r);
            $this->assertIsBool($r['compliant']);
            $this->assertIsInt($r['errorCount']);
        } finally {
            $doc->close();
        }
    }

    public function testIsPdfXThrowsUntilCAbiLands(): void
    {
        $doc = PdfDocument::open($this->fixture('simple.pdf'));
        try {
            $this->expectException(\BadMethodCallException::class);
            PdfValidator::isPdfX($doc);
        } finally {
            $doc->close();
        }
    }
}
