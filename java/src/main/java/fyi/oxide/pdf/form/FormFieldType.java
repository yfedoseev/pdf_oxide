/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.form;

/**
 * The five PDF AcroForm field types per PDF 32000-1 §12.7. XFA-only
 * fields are not exposed in v0.3.53 — they collapse to {@link #TEXT}
 * for read purposes and refuse writes (the Rust core's
 * `set_form_field_value` returns an unsupported error).
 */
public enum FormFieldType {
    /** Single- or multi-line text input. */
    TEXT,
    /** Two-state checkbox. */
    CHECKBOX,
    /** Mutually-exclusive radio button group. */
    RADIO,
    /** Single- or multi-select choice list / combo box. */
    CHOICE,
    /** Digital signature field (PAdES / CMS). */
    SIGNATURE
}
