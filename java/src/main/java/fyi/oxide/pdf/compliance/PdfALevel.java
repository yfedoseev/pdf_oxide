/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.compliance;

/**
 * PDF/A conformance levels per ISO 19005.
 *
 * <p>Declaration order is chosen so {@link #ordinal()} matches the
 * cdylib's wire-format integer encoding documented at
 * {@code src/ffi.rs:1225} ({@code 0=A1b 1=A1a 2=A2b 3=A2a 4=A2u
 * 5=A3b 6=A3a 7=A3u}). C# uses explicit-coded enums for the same
 * wire format; Java keeps {@code .ordinal()} idiomatic by reordering
 * the declarations so every binding (Java, Ruby, PHP, C#, Go) sends
 * the same integer for the same PDF/A level.
 *
 * <p>The "B before A" intra-level order may look unusual to readers
 * expecting alphabetical; it matches the cdylib contract, which is
 * the canonical source of truth.
 */
public enum PdfALevel {
    /** PDF/A-1b (Level B, visually reliable — no tagging required). */
    A_1B,
    /** PDF/A-1a (Level A, accessible — tagged structure required). */
    A_1A,
    /** PDF/A-2b (Level B, ISO 32000-1 base). */
    A_2B,
    /** PDF/A-2a (Level A, ISO 32000-1 base; tagged). */
    A_2A,
    /** PDF/A-2u (Level U, with Unicode mapping). */
    A_2U,
    /** PDF/A-3b (Level B with attached files of any type). */
    A_3B,
    /** PDF/A-3a (Level A with attached files of any type). */
    A_3A,
    /** PDF/A-3u (Level U with attached files of any type). */
    A_3U,
    /** PDF/A-4 (ISO 19005-4) and sub-levels — not yet supported by cdylib. */
    A_4,
    A_4E,
    A_4F
}
