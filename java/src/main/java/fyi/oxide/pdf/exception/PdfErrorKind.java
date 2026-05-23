/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.exception;

/**
 * The canonical taxonomy of pdf_oxide errors as seen from Java.
 *
 * <p>Each {@link PdfException} carries a {@code PdfErrorKind} via
 * {@link PdfException#kind()}. Call sites can either catch the
 * specific subclass (when the recovery path is type-specific) or
 * {@code switch} on the kind (when generic dispatch is enough).
 *
 * <p>Mapping from the Rust {@code PdfError} variants is one-to-one
 * and centralised in {@code pdf_oxide_jni/src/error.rs}. CI enforces
 * that every Rust variant maps to exactly one {@code PdfErrorKind};
 * an unmapped variant fails the build.
 *
 * <p>See {@code docs/releases/plans/v0.3.53/00-common-foundation.md}
 * §5 for the exception-taxonomy contract.
 */
public enum PdfErrorKind {

    /** Malformed PDF (xref, header, syntax). Subclass: {@link PdfParseException}. */
    PARSE,

    /** PDF is encrypted and no usable password was supplied. Subclass: {@link PdfEncryptedException}. */
    ENCRYPTED,

    /** PDF permissions block the requested operation. Subclass: {@link PdfPermissionException}. */
    PERMISSION,

    /** Underlying I/O error (file system, network, stream). Subclass: {@link PdfIoException}. */
    IO,

    /** OCR was requested but unavailable (feature off, no models). Subclass: {@link PdfOcrUnavailableException}. */
    OCR_UNAVAILABLE,

    /** Digital-signature operation failed (PAdES B-B/B-T/B-LT). Subclass: {@link PdfSignatureException}. */
    SIGNATURE,

    /** Handle was closed, null, or otherwise invalid. Subclass: {@link PdfInvalidStateException}. */
    INVALID_STATE,

    /** The requested operation is not implemented for the input. Subclass: {@link PdfUnsupportedException}. */
    UNSUPPORTED,

    /** Fallback bucket; includes panics caught at the JNI boundary. Subclass: {@link PdfException} directly. */
    OTHER
}
