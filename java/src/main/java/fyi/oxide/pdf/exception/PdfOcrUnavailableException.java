/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.exception;

/**
 * Pinned {@link PdfErrorKind#OCR_UNAVAILABLE} subclass of {@link PdfException}.
 * See {@link PdfErrorKind#OCR_UNAVAILABLE} for the semantic definition.
 */
public final class PdfOcrUnavailableException extends PdfException {

    private static final long serialVersionUID = 1L;

    /** @see PdfException#PdfException(PdfErrorKind, String) */
    public PdfOcrUnavailableException(String message) {
        super(PdfErrorKind.OCR_UNAVAILABLE, message);
    }

    /** @see PdfException#PdfException(PdfErrorKind, String, Throwable) */
    public PdfOcrUnavailableException(String message, Throwable cause) {
        super(PdfErrorKind.OCR_UNAVAILABLE, message, cause);
    }
}
