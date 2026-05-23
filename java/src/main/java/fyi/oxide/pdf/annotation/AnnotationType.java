/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.annotation;

/**
 * PDF annotation subtype enum per ISO 32000-1 §12.5. v0.3.53 ships
 * the most common types; the {@link #OTHER} bucket holds any
 * subtype pdf_oxide recognises but Java hasn't subclassed.
 */
public enum AnnotationType {
    /** Highlight annotation (text underlay, semi-transparent). */
    HIGHLIGHT,
    /** Sticky-note / text annotation (pop-up comment). */
    TEXT,
    /** Hyperlink (URI / GoTo destination). */
    LINK,
    /** Stamp (image overlay; e.g. "Approved"). */
    STAMP,
    /** Underline. */
    UNDERLINE,
    /** Strike-out. */
    STRIKEOUT,
    /** Squiggly underline (spell-check). */
    SQUIGGLY,
    /** Free text (annotation drawn directly on the page). */
    FREE_TEXT,
    /** Line annotation. */
    LINE,
    /** Square. */
    SQUARE,
    /** Circle. */
    CIRCLE,
    /** File attachment. */
    FILE_ATTACHMENT,
    /** Other / not yet classified. */
    OTHER
}
