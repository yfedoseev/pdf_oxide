/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.exception;

/**
 * Pinned {@link PdfErrorKind#SIGNATURE} subclass of {@link PdfException}.
 * See {@link PdfErrorKind#SIGNATURE} for the semantic definition.
 */
public final class PdfSignatureException extends PdfException {

    private static final long serialVersionUID = 1L;

    /** @see PdfException#PdfException(PdfErrorKind, String) */
    public PdfSignatureException(String message) {
        super(PdfErrorKind.SIGNATURE, message);
    }

    /** @see PdfException#PdfException(PdfErrorKind, String, Throwable) */
    public PdfSignatureException(String message, Throwable cause) {
        super(PdfErrorKind.SIGNATURE, message, cause);
    }
}
