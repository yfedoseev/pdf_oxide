/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.auto;

/**
 * Classification of a PDF page from the v0.3.51 AutoExtractor
 * classifier. Drives the {@code pages_needing_ocr} list and the
 * routing decision in {@link ExtractMode#AUTO}.
 *
 * <p>Mirrors the Rust {@code pdf_oxide::extractors::auto::PageKind}
 * variants. Chart / encrypted-permission-denied states surface
 * through {@link ExtractReason} (not {@code PageClass}) — see
 * {@link ExtractReason#CHART_NOT_TRANSCRIBED} and
 * {@link ExtractReason#ENCRYPTED_NO_EXTRACT_PERMISSION}.
 *
 * <p>Ordinals cross the JNI boundary, so the order here is locked
 * to the Rust mapping in {@code pdf_oxide_jni/src/auto_extractor.rs}.
 */
public enum PageClass {
    /** Native text-layer is good — no OCR needed. */
    TEXT_LAYER,
    /** Image-only page (scanned) — OCR required for any text. */
    SCANNED,
    /** Native text plus image regions with embedded text. */
    MIXED,
    /** No text and no images — blank or whitespace-only page. */
    EMPTY
}
