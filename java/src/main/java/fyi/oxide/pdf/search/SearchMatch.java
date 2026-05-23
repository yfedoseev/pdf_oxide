/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.search;

import fyi.oxide.pdf.geometry.BBox;
import java.util.Objects;

/**
 * A single match in a {@link SearchResult}. Carries the matched text,
 * the page index where it was found, and its bounding box on the page.
 */
public final class SearchMatch {
    private final int pageIndex;
    private final BBox bbox;
    private final String text;

    public SearchMatch(int pageIndex, BBox bbox, String text) {
        this.pageIndex = pageIndex;
        this.bbox = Objects.requireNonNull(bbox, "bbox");
        this.text = Objects.requireNonNull(text, "text");
    }

    public int pageIndex() {
        return pageIndex;
    }

    public BBox bbox() {
        return bbox;
    }

    public String text() {
        return text;
    }

    @Override
    public String toString() {
        return "SearchMatch[page=" + pageIndex + " bbox=" + bbox + " text=" + text + "]";
    }
}
