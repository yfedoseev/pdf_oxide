/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.text;

import fyi.oxide.pdf.geometry.Color;
import java.util.Objects;
import org.jspecify.annotations.Nullable;

/**
 * Visual style metadata for a {@link TextSpan}. Font name may be
 * absent on encrypted PDFs with restricted permission or on
 * synthetic OCR spans.
 */
public final class TextStyle {

    private final @Nullable String font;
    private final double size;
    private final Color color;
    private final boolean bold;
    private final boolean italic;

    public TextStyle(@Nullable String font, double size, Color color, boolean bold, boolean italic) {
        this.font = font;
        this.size = size;
        this.color = Objects.requireNonNull(color, "color");
        this.bold = bold;
        this.italic = italic;
    }

    /** @return PostScript font name (e.g. {@code "Helvetica-Bold"}), or null if unavailable. */
    public @Nullable String font() {
        return font;
    }
    /** @return font size in PDF user-space units (typically points). */
    public double size() {
        return size;
    }
    /** @return fill color. */
    public Color color() {
        return color;
    }
    /** @return true if the span is rendered in bold style. */
    public boolean bold() {
        return bold;
    }
    /** @return true if the span is rendered in italic style. */
    public boolean italic() {
        return italic;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TextStyle)) return false;
        TextStyle s = (TextStyle) o;
        return Double.compare(s.size, size) == 0
                && bold == s.bold
                && italic == s.italic
                && Objects.equals(font, s.font)
                && color.equals(s.color);
    }

    @Override
    public int hashCode() {
        return Objects.hash(font, size, color, bold, italic);
    }

    @Override
    public String toString() {
        return "TextStyle[font=" + font + ", size=" + size + ", color=" + color + ", bold=" + bold + ", italic="
                + italic + "]";
    }
}
