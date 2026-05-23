/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.text;

import fyi.oxide.pdf.geometry.BBox;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * A horizontal line of text composed of {@link TextWord}s in
 * reading order.
 */
public final class TextLine {
    private final String text;
    private final BBox bbox;
    private final List<TextWord> words;

    public TextLine(String text, BBox bbox, List<TextWord> words) {
        this.text = Objects.requireNonNull(text, "text");
        this.bbox = Objects.requireNonNull(bbox, "bbox");
        // Defensive copy + unmodifiable view — the list is part of the
        // value, must not mutate after construction.
        this.words = Collections.unmodifiableList(new java.util.ArrayList<>(Objects.requireNonNull(words, "words")));
    }

    public String text() {
        return text;
    }

    public BBox bbox() {
        return bbox;
    }
    /** @return unmodifiable view of the words on this line, in reading order. */
    public List<TextWord> words() {
        return words;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TextLine)) return false;
        TextLine l = (TextLine) o;
        return text.equals(l.text) && bbox.equals(l.bbox) && words.equals(l.words);
    }

    @Override
    public int hashCode() {
        return Objects.hash(text, bbox, words);
    }

    @Override
    public String toString() {
        return "TextLine[text=" + text + ", bbox=" + bbox + ", words=" + words.size() + "]";
    }
}
