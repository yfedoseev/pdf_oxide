/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.text;

import fyi.oxide.pdf.geometry.BBox;
import java.util.Objects;

/**
 * A single word extracted from a PDF page, with its bounding box
 * and (if from OCR) a confidence score in {@code [0, 1]}.
 *
 * <p>For native text-layer extraction (no OCR), {@link #confidence()}
 * is always {@code 1.0f}. For OCR-derived words it reflects the
 * recognizer's per-token confidence.
 */
public final class TextWord {
    private final String text;
    private final BBox bbox;
    private final float confidence;
    private final long sequence;
    private final float rotationDegrees;

    public TextWord(String text, BBox bbox, float confidence, long sequence, float rotationDegrees) {
        this.text = Objects.requireNonNull(text, "text");
        this.bbox = Objects.requireNonNull(bbox, "bbox");
        this.confidence = confidence;
        this.sequence = sequence;
        this.rotationDegrees = rotationDegrees;
    }

    public String text() {
        return text;
    }

    public BBox bbox() {
        return bbox;
    }

    public float confidence() {
        return confidence;
    }

    /**
     * The content-stream emission order of the span this word originated
     * from. Words drawn consecutively in the page's content stream have
     * adjacent sequence values, which distinguishes genuinely consecutive
     * draws from words that are merely spatially close. Independent of
     * reading order.
     */
    public long sequence() {
        return sequence;
    }

    /**
     * Rotation of the word's glyph run in degrees, snapped to a quadrant
     * ({@code 0} / {@code 90} / {@code 180} / {@code -90}). {@code 90}
     * means the text reads bottom-to-top on an unrotated page — e.g. a
     * landscape table typeset on a portrait page. {@code 0} for ordinary
     * horizontal text.
     */
    public float rotationDegrees() {
        return rotationDegrees;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TextWord)) return false;
        TextWord w = (TextWord) o;
        return Float.compare(w.confidence, confidence) == 0
                && sequence == w.sequence
                && Float.compare(w.rotationDegrees, rotationDegrees) == 0
                && text.equals(w.text)
                && bbox.equals(w.bbox);
    }

    @Override
    public int hashCode() {
        return Objects.hash(text, bbox, confidence, sequence, rotationDegrees);
    }

    @Override
    public String toString() {
        return "TextWord[text=" + text + ", bbox=" + bbox + ", confidence=" + confidence + ", sequence=" + sequence
                + ", rotationDegrees=" + rotationDegrees + "]";
    }
}
