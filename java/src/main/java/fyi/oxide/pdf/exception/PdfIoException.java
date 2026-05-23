/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.exception;

/**
 * Pinned {@link PdfErrorKind#IO} subclass of {@link PdfException}.
 * See {@link PdfErrorKind#IO} for the semantic definition.
 */
public final class PdfIoException extends PdfException {

    private static final long serialVersionUID = 1L;

    /** @see PdfException#PdfException(PdfErrorKind, String) */
    public PdfIoException(String message) {
        super(PdfErrorKind.IO, message);
    }

    /** @see PdfException#PdfException(PdfErrorKind, String, Throwable) */
    public PdfIoException(String message, Throwable cause) {
        super(PdfErrorKind.IO, message, cause);
    }
}
