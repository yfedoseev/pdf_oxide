/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.form;

import fyi.oxide.pdf.geometry.BBox;
import java.util.Objects;
import java.util.Optional;
import org.jspecify.annotations.Nullable;

/**
 * A PDF AcroForm field as read from a document. Mutation is performed
 * via {@link fyi.oxide.pdf.DocumentEditor#setFormField} (Java side
 * holds no mutable state on the field).
 */
public final class FormField {
    private final String name;
    private final FormFieldType type;
    private final @Nullable String value;
    private final @Nullable BBox bbox;
    private final int pageIndex;

    public FormField(String name, FormFieldType type, @Nullable String value, @Nullable BBox bbox, int pageIndex) {
        this.name = Objects.requireNonNull(name, "name");
        this.type = Objects.requireNonNull(type, "type");
        this.value = value;
        this.bbox = bbox;
        this.pageIndex = pageIndex;
    }

    /** @return field name (the dot-separated AcroForm full name). */
    public String name() {
        return name;
    }

    public FormFieldType type() {
        return type;
    }

    /** @return the field's value, or {@code Optional.empty()} if unset. */
    public Optional<String> value() {
        return Optional.ofNullable(value);
    }

    /** @return the field's on-page widget bbox, or {@code Optional.empty()} if no visible widget. */
    public Optional<BBox> bbox() {
        return Optional.ofNullable(bbox);
    }

    /** @return 0-based page index where the widget is placed. */
    public int pageIndex() {
        return pageIndex;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof FormField)) return false;
        FormField f = (FormField) o;
        return pageIndex == f.pageIndex
                && name.equals(f.name)
                && type == f.type
                && Objects.equals(value, f.value)
                && Objects.equals(bbox, f.bbox);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, type, value, bbox, pageIndex);
    }

    @Override
    public String toString() {
        return "FormField[" + type + " name=" + name
                + (value == null ? "" : " value=" + value)
                + " page=" + pageIndex + "]";
    }
}
