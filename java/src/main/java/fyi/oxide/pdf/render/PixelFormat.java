/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.render;

/**
 * Output pixel format for {@link fyi.oxide.pdf.PdfDocument} page
 * rendering.
 */
public enum PixelFormat {
    /** 8-bit per channel RGBA. */
    RGBA_8888,
    /** 8-bit per channel RGB (no alpha). */
    RGB_888,
    /** 8-bit grayscale. */
    GRAY_8,
    /** PNG-encoded byte stream. */
    PNG
}
