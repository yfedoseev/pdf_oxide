<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide\Tests\Unit;

use PdfOxide\PdfValidator;
use PHPUnit\Framework\TestCase;

/**
 * Locks in the PDF/A and PDF/UA wire-format integer mapping against
 * the cdylib's documented C ABI ({@code src/ffi.rs:1225} +
 * {@code src/ffi.rs:5538}). Every pdf_oxide binding (Java, Ruby, PHP,
 * C#, Go) must send the SAME integer for the SAME PDF/A level — any
 * future re-numbering surfaces here as a hard test failure rather
 * than as a silently-wrong validation verdict.
 *
 * Pre-v0.3.55 PHP had {@code PDFA_1A = 0; PDFA_1B = 1; ...} —
 * alphabetical-natural, semantically reversed against what the
 * cdylib actually does ({@code 0 = A1b, 1 = A1a, ...}). Users asking
 * for "validate as A-1a" got A-1b validation back.
 */
final class PdfValidatorLevelMappingTest extends TestCase
{
    public function testPdfALevelOrdinals(): void
    {
        // B before A within each level (1, 2, 3) — matches src/ffi.rs:1225.
        $this->assertSame(0, PdfValidator::PDFA_1B);
        $this->assertSame(1, PdfValidator::PDFA_1A);
        $this->assertSame(2, PdfValidator::PDFA_2B);
        $this->assertSame(3, PdfValidator::PDFA_2A);
        $this->assertSame(4, PdfValidator::PDFA_2U);
        $this->assertSame(5, PdfValidator::PDFA_3B);
        $this->assertSame(6, PdfValidator::PDFA_3A);
        $this->assertSame(7, PdfValidator::PDFA_3U);
    }

    public function testPdfUaLevelOrdinals(): void
    {
        // 1-indexed — cdylib treats `level == 2` as UA-2, else UA-1.
        // Mirrors C#'s `Ua1 = 1, Ua2 = 2`.
        $this->assertSame(1, PdfValidator::PDFUA_1);
        $this->assertSame(2, PdfValidator::PDFUA_2);
    }
}
