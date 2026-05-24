<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide;

/**
 * Result of an {@see AutoExtractor::extractAutoPage()} or
 * {@see AutoExtractor::extractAutoDocument()} call.
 *
 * Mirrors `fyi.oxide.pdf.auto.AutoResult` from the Java binding.
 *
 * Carries the extracted text + a typed reason naming the quality /
 * fallback state. Region detail (per-region bbox + confidence) lands
 * in a follow-up via the v0.3.51 JSON-envelope wire format; reach for
 * {@see AutoExtractor::extractPageJson()} when you need it.
 */
final readonly class AutoExtractResult
{
    // ─────────────── Reason constants ──────────────────────
    // Snake-case wire tokens from Rust's `ReasonCode` enum
    // (src/extractors/auto.rs). Frozen for cross-binding parity.

    public const REASON_OK = 'ok';

    public const REASON_NATIVE_TEXT_HIGH_CONFIDENCE = 'native_text_high_confidence';

    public const REASON_NO_TEXT_LAYER_PRESENT = 'no_text_layer_present';

    public const REASON_OCR_REQUESTED_BUT_UNAVAILABLE = 'ocr_requested_but_unavailable';

    public const REASON_OCR_LOW_CONFIDENCE_FALLBACK = 'ocr_low_confidence_fallback';

    public const REASON_IMAGE_TABLE_RECONSTRUCTED = 'image_table_reconstructed';

    public const REASON_EMPTY = 'empty';

    // ─────────────── PageKind constants ────────────────────

    public const KIND_TEXT_LAYER = 'text_layer';

    public const KIND_SCANNED = 'scanned';

    public const KIND_IMAGE_TEXT = 'image_text';

    public const KIND_MIXED = 'mixed';

    public const KIND_EMPTY = 'empty';

    /**
     * @param string                $text             extracted text
     * @param string                $reason           one of `REASON_*`
     * @param float                 $confidence       in `[0.0, 1.0]`
     * @param bool                  $ocrUsed          whether OCR ran for this result
     * @param array<int, array<string,mixed>> $regions         per-region rich detail
     *                                                  (currently empty in simplified surface)
     * @param array<int, int>       $pagesNeedingOcr  pages that still need OCR
     */
    public function __construct(
        public string $text,
        public string $reason = self::REASON_OK,
        public float $confidence = 1.0,
        public bool $ocrUsed = false,
        public array $regions = [],
        public array $pagesNeedingOcr = [],
    ) {}

    /** Whether the extraction succeeded with no degradation. */
    public function isOk(): bool
    {
        return $this->reason === self::REASON_OK
            || $this->reason === self::REASON_NATIVE_TEXT_HIGH_CONFIDENCE;
    }

    /** Whether the OCR-unavailable graceful-fallback path engaged. */
    public function isOcrFallback(): bool
    {
        return $this->reason === self::REASON_OCR_REQUESTED_BUT_UNAVAILABLE
            || $this->reason === self::REASON_OCR_LOW_CONFIDENCE_FALLBACK;
    }
}
