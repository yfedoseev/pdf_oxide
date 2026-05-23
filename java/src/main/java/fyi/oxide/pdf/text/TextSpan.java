/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.text;

import fyi.oxide.pdf.geometry.BBox;
import java.util.Objects;

/**
 * A run of text with uniform style (font, size, color, weight).
 * Multiple spans typically compose a {@link TextLine}.
 */
public final class TextSpan {
    private final String text;
    private final BBox bbox;
    private final TextStyle style;

    public TextSpan(String text, BBox bbox, TextStyle style) {
        this.text = Objects.requireNonNull(text, "text");
        this.bbox = Objects.requireNonNull(bbox, "bbox");
        this.style = Objects.requireNonNull(style, "style");
    }

    public String text() {
        return text;
    }

    public BBox bbox() {
        return bbox;
    }

    public TextStyle style() {
        return style;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TextSpan)) return false;
        TextSpan s = (TextSpan) o;
        return text.equals(s.text) && bbox.equals(s.bbox) && style.equals(s.style);
    }

    @Override
    public int hashCode() {
        return Objects.hash(text, bbox, style);
    }

    @Override
    public String toString() {
        return "TextSpan[text=" + text + ", bbox=" + bbox + ", style=" + style + "]";
    }
}
