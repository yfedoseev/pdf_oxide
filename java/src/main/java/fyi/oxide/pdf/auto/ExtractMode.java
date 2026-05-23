/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.auto;

/**
 * The single-mode enum that drives {@link AutoExtractor}.
 *
 * <p>From v0.3.51's design ({@code docs/releases/plans/v0.3.51/api-design.md}):
 * one enum, not boolean soup (which is the Docling / PyMuPDF4LLM
 * anti-pattern that produced silent-no-op bugs like Docling #2312).
 * Default is {@link #AUTO}.
 */
public enum ExtractMode {
    /** Text-layer only — never invoke OCR even on scanned pages. */
    TEXT_ONLY,
    /** Default: native text-layer where present, OCR for scanned regions. */
    AUTO,
    /** Always OCR every page, ignoring any native text layer. */
    FORCE_OCR
}
