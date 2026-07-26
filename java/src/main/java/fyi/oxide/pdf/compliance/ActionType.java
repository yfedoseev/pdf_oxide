/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.compliance;

/**
 * Kind of remediation a {@link fyi.oxide.pdf.PdfAConverter} run performed on a document.
 *
 * <p>Declaration order matches the cdylib's {@code ActionType} enum
 * (see {@code src/compliance/converter.rs}) so the JNI shim can build
 * these directly from the Rust variant's discriminant.
 */
public enum ActionType {
    /** Added XMP metadata where none was present. */
    ADDED_XMP_METADATA,
    /** Added PDF/A identification (part/conformance) to the XMP packet. */
    ADDED_PDFA_IDENTIFICATION,
    /** Embedded a font that was previously referenced but not embedded. */
    EMBEDDED_FONT,
    /** Added an output intent with an ICC profile. */
    ADDED_OUTPUT_INTENT,
    /** Removed JavaScript actions. */
    REMOVED_JAVASCRIPT,
    /** Removed encryption. */
    REMOVED_ENCRYPTION,
    /** Flattened transparency (required for PDF/A-1). */
    FLATTENED_TRANSPARENCY,
    /** Removed embedded files (required for PDF/A-1 and PDF/A-2). */
    REMOVED_EMBEDDED_FILES,
    /** Added a structure tree. */
    ADDED_STRUCTURE,
    /** Fixed an annotation missing its appearance stream. */
    FIXED_ANNOTATION,
    /** Added a document-level language specification. */
    ADDED_LANGUAGE
}
