/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.compliance;

/**
 * PDF/A conformance levels per ISO 19005. Mirrors pdf_oxide's
 * compliance validator output.
 */
public enum PdfALevel {
    /** PDF/A-1a (Level A, accessible — tagged structure required). */
    A_1A,
    /** PDF/A-1b (Level B, visually reliable — no tagging required). */
    A_1B,
    /** PDF/A-2a (Level A, ISO 32000-1 base; tagged). */
    A_2A,
    /** PDF/A-2b (Level B, ISO 32000-1 base). */
    A_2B,
    /** PDF/A-2u (Level U, with Unicode mapping). */
    A_2U,
    /** PDF/A-3a, 3b, 3u — same as 2x but allow attached files of any type. */
    A_3A,
    A_3B,
    A_3U,
    /** PDF/A-4 (ISO 19005-4) and sub-levels. */
    A_4,
    A_4E,
    A_4F
}
