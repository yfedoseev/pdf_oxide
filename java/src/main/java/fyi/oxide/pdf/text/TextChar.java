/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.text;

import fyi.oxide.pdf.geometry.BBox;
import java.util.Objects;

/**
 * A single character (Unicode codepoint) extracted from a PDF page.
 *
 * <p>{@link #codepoint()} returns a full Unicode codepoint (may be
 * &gt; 0xFFFF in the supplementary plane). The character can be
 * converted to a Java string via {@link String#valueOf(int[], int, int)}
 * or {@link Character#toChars(int)}.
 */
public final class TextChar {
    private final int codepoint;
    private final BBox bbox;
    private final float confidence;

    public TextChar(int codepoint, BBox bbox, float confidence) {
        if (codepoint < 0) {
            throw new IllegalArgumentException("codepoint must be non-negative, got " + codepoint);
        }
        this.codepoint = codepoint;
        this.bbox = Objects.requireNonNull(bbox, "bbox");
        this.confidence = confidence;
    }

    /** @return the Unicode codepoint (NOT a UTF-16 char). */
    public int codepoint() {
        return codepoint;
    }

    public BBox bbox() {
        return bbox;
    }

    public float confidence() {
        return confidence;
    }

    /** @return the codepoint as a Java string (handles supplementary plane). */
    public String asString() {
        return new String(Character.toChars(codepoint));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TextChar)) return false;
        TextChar c = (TextChar) o;
        return codepoint == c.codepoint && Float.compare(c.confidence, confidence) == 0 && bbox.equals(c.bbox);
    }

    @Override
    public int hashCode() {
        return Objects.hash(codepoint, bbox, confidence);
    }

    @Override
    public String toString() {
        return "TextChar[codepoint=" + codepoint + " ('" + asString() + "')" + ", bbox=" + bbox + ", confidence="
                + confidence + "]";
    }
}
