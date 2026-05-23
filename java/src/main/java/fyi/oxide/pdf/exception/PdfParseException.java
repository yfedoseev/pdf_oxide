/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.exception;

/**
 * Pinned {@link PdfErrorKind#PARSE} subclass of {@link PdfException}.
 * See {@link PdfErrorKind#PARSE} for the semantic definition.
 */
public final class PdfParseException extends PdfException {

    private static final long serialVersionUID = 1L;

    /** @see PdfException#PdfException(PdfErrorKind, String) */
    public PdfParseException(String message) {
        super(PdfErrorKind.PARSE, message);
    }

    /** @see PdfException#PdfException(PdfErrorKind, String, Throwable) */
    public PdfParseException(String message, Throwable cause) {
        super(PdfErrorKind.PARSE, message, cause);
    }
}
