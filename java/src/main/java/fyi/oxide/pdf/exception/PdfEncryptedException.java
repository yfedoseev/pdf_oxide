/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.exception;

/**
 * Pinned {@link PdfErrorKind#ENCRYPTED} subclass of {@link PdfException}.
 * See {@link PdfErrorKind#ENCRYPTED} for the semantic definition.
 */
public final class PdfEncryptedException extends PdfException {

    private static final long serialVersionUID = 1L;

    /** @see PdfException#PdfException(PdfErrorKind, String) */
    public PdfEncryptedException(String message) {
        super(PdfErrorKind.ENCRYPTED, message);
    }

    /** @see PdfException#PdfException(PdfErrorKind, String, Throwable) */
    public PdfEncryptedException(String message, Throwable cause) {
        super(PdfErrorKind.ENCRYPTED, message, cause);
    }
}
