<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide;

use PdfOxide\FFI\NativeLibrary;

/**
 * Static façade for PDF/A · PDF/UA compliance validation (v0.3.50).
 *
 * Mirrors `fyi.oxide.pdf.PdfValidator` from the Java binding.
 *
 * v0.3.55 surface ships the simplified boolean variants for PDF/A and
 * PDF/UA. PDF/X validation is not exposed in the C ABI yet — the
 * matching wrapper throws {@see \BadMethodCallException} until the
 * pdf_oxide Rust side adds the public symbol (tracked).
 */
final class PdfValidator
{
    // ─────────────── PDF/A level ordinals ──────────────────
    // Frozen by the Rust ABI (`PdfALevel` in pdf_oxide core).
    // Wire format: int32_t. Ordinals match the Rust enum order.

    public const PDFA_1A = 0;

    public const PDFA_1B = 1;

    public const PDFA_2A = 2;

    public const PDFA_2B = 3;

    public const PDFA_2U = 4;

    public const PDFA_3A = 5;

    public const PDFA_3B = 6;

    public const PDFA_3U = 7;

    // ─────────────── PDF/UA level ordinals ─────────────────

    public const PDFUA_1 = 0;

    public const PDFUA_2 = 1;

    /** Static-only. */
    private function __construct()
    {
    }

    /**
     * Quick PDF/A compliance check. `$level` is one of the
     * `PDFA_*` constants (see also Java's `PdfALevel` ordinal table).
     */
    public static function isPdfA(PdfDocument $doc, int $level = self::PDFA_1B): bool
    {
        $ffi = NativeLibrary::getInstance();
        $errorCode = \FFI::new('int32_t');
        $results = $ffi->pdf_validate_pdf_a_level($doc->getHandle(), $level, \FFI::addr($errorCode));
        if ((int) $errorCode->cdata !== 0 || $results === null) {
            return false;
        }
        try {
            $compliant = (bool) $ffi->pdf_pdf_a_is_compliant($results, \FFI::addr($errorCode));
            return (int) $errorCode->cdata === 0 ? $compliant : false;
        } finally {
            $ffi->pdf_pdf_a_results_free($results);
        }
    }

    /** Quick PDF/UA accessibility check. */
    public static function isPdfUa(PdfDocument $doc, int $level = self::PDFUA_1): bool
    {
        $ffi = NativeLibrary::getInstance();
        $errorCode = \FFI::new('int32_t');
        $results = $ffi->pdf_validate_pdf_ua($doc->getHandle(), $level, \FFI::addr($errorCode));
        if ((int) $errorCode->cdata !== 0 || $results === null) {
            return false;
        }
        try {
            $accessible = (bool) $ffi->pdf_pdf_ua_is_accessible($results, \FFI::addr($errorCode));
            return (int) $errorCode->cdata === 0 ? $accessible : false;
        } finally {
            $ffi->pdf_pdf_ua_results_free($results);
        }
    }

    /**
     * v0.3.55: PDF/X validation is not exposed in the pdf_oxide C ABI.
     * Mirrors Java's `validatePdfX` follow-up note.
     */
    public static function isPdfX(PdfDocument $doc): bool
    {
        throw new \BadMethodCallException(
            'PdfValidator::isPdfX — pdf_oxide v0.3.55 has no public PDF/X validator (Phase 4 T16 follow-up).'
        );
    }

    /**
     * @return array{compliant:bool, errorCount:int, warningCount:int}
     *         a richer PDF/A validation result.
     */
    public static function validatePdfA(PdfDocument $doc, int $level = self::PDFA_1B): array
    {
        $ffi = NativeLibrary::getInstance();
        $errorCode = \FFI::new('int32_t');
        $results = $ffi->pdf_validate_pdf_a_level($doc->getHandle(), $level, \FFI::addr($errorCode));
        if ((int) $errorCode->cdata !== 0 || $results === null) {
            return ['compliant' => false, 'errorCount' => 0, 'warningCount' => 0];
        }
        try {
            $compliant = (bool) $ffi->pdf_pdf_a_is_compliant($results, \FFI::addr($errorCode));
            $errors = (int) $ffi->pdf_pdf_a_error_count($results);
            $warnings = (int) $ffi->pdf_pdf_a_warning_count($results);
            return [
                'compliant' => $compliant,
                'errorCount' => $errors,
                'warningCount' => $warnings,
            ];
        } finally {
            $ffi->pdf_pdf_a_results_free($results);
        }
    }
}
