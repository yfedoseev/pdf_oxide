/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.exception;

/**
 * Pinned {@link PdfErrorKind#INVALID_STATE} subclass of {@link PdfException}.
 * See {@link PdfErrorKind#INVALID_STATE} for the semantic definition.
 */
public final class PdfInvalidStateException extends PdfException {

    private static final long serialVersionUID = 1L;

    /** @see PdfException#PdfException(PdfErrorKind, String) */
    public PdfInvalidStateException(String message) {
        super(PdfErrorKind.INVALID_STATE, message);
    }

    /** @see PdfException#PdfException(PdfErrorKind, String, Throwable) */
    public PdfInvalidStateException(String message, Throwable cause) {
        super(PdfErrorKind.INVALID_STATE, message, cause);
    }
}
