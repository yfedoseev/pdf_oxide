/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.auto;

/**
 * Typed reason explaining why an {@link AutoResult} or
 * {@link RegionResult} is in a particular state. The v0.3.51
 * "tell me why" feature ({@code docs/releases/plans/v0.3.51/00-common-foundation.md} §3)
 * — the #1 user-pain fix vs every other PDF library, which return
 * opaque empty strings on failure.
 *
 * <p>{@link #OK} is the only non-degraded outcome. Anything else
 * must name why.
 */
public enum ExtractReason {
    /** Result is good — no degradation. */
    OK,
    /** Page has no text layer; OCR ran (if available) or wasn't requested. */
    SCANNED_NO_TEXT_LAYER,
    /** Native text exists but the font lacks a usable {@code /ToUnicode} mapping — output is garbled. */
    GLYPH_MAPPING_MISSING,
    /** PDF encrypted with a {@code /P} bit denying extraction permission. */
    ENCRYPTED_NO_EXTRACT_PERMISSION,
    /** OCR detected an image-table but the spatial detector couldn't recover rows/cols. */
    IMAGE_TABLE_NO_STRUCTURE,
    /** Chart / figure detected; pdf_oxide does NOT transcribe charts (an honest non-goal). */
    CHART_NOT_TRANSCRIBED,
    /** OCR was requested ({@link ExtractMode#AUTO}/{@link ExtractMode#FORCE_OCR}) but the {@code ocr} feature is not compiled in OR no models are available. */
    OCR_REQUESTED_BUT_UNAVAILABLE,
    /** OCR ran but the average per-region confidence is below threshold. */
    OCR_LOW_CONFIDENCE,
    /** Region produced no output (empty image or pure whitespace). */
    EMPTY,
    /** OCR was attempted but failed at runtime; native text-layer is used as fallback. */
    FALLBACK_FROM_OCR
}
