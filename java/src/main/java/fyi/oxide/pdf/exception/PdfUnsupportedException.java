/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.exception;

/**
 * Pinned {@link PdfErrorKind#UNSUPPORTED} subclass of {@link PdfException}.
 * See {@link PdfErrorKind#UNSUPPORTED} for the semantic definition.
 */
public final class PdfUnsupportedException extends PdfException {

    private static final long serialVersionUID = 1L;

    /** @see PdfException#PdfException(PdfErrorKind, String) */
    public PdfUnsupportedException(String message) {
        super(PdfErrorKind.UNSUPPORTED, message);
    }

    /** @see PdfException#PdfException(PdfErrorKind, String, Throwable) */
    public PdfUnsupportedException(String message, Throwable cause) {
        super(PdfErrorKind.UNSUPPORTED, message, cause);
    }
}
