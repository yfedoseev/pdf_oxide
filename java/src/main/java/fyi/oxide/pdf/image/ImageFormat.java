/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.image;

/**
 * Format of an {@link ExtractedImage}. Mirrors the Rust core's
 * supported image stream filters (PDF 32000-1 §7.4).
 */
public enum ImageFormat {
    /** JPEG (DCTDecode in PDF). */
    JPEG,
    /** PNG (FlateDecode + per-row predictor, lossless). */
    PNG,
    /** JBIG2 (bilevel image compression; PDF 32000-1 §7.4.7). */
    JBIG2,
    /** JPEG2000 (JPXDecode). */
    JPEG2000,
    /** CCITTFax (G3/G4 facsimile). */
    CCITT,
    /** Raw bitmap (uncompressed or zlib-compressed). */
    RAW,
    /** Other / not yet classified. */
    OTHER
}
