/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.exception;

/**
 * Pinned {@link PdfErrorKind#PERMISSION} subclass of {@link PdfException}.
 * See {@link PdfErrorKind#PERMISSION} for the semantic definition.
 */
public final class PdfPermissionException extends PdfException {

    private static final long serialVersionUID = 1L;

    /** @see PdfException#PdfException(PdfErrorKind, String) */
    public PdfPermissionException(String message) {
        super(PdfErrorKind.PERMISSION, message);
    }

    /** @see PdfException#PdfException(PdfErrorKind, String, Throwable) */
    public PdfPermissionException(String message, Throwable cause) {
        super(PdfErrorKind.PERMISSION, message, cause);
    }
}
