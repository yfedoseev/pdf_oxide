/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.auto;

import fyi.oxide.pdf.geometry.BBox;
import fyi.oxide.pdf.table.Table;
import java.util.Objects;
import java.util.Optional;
import org.jspecify.annotations.Nullable;

/**
 * Per-region extraction result inside an {@link AutoResult}. Each
 * region corresponds to a contiguous chunk on a page (a text block,
 * an image-as-text, an image-table). v0.3.51 §3 guarantee:
 * {@code bbox} is always present even if {@code text} is empty —
 * reading order is never silently corrupted.
 */
public final class RegionResult {
    private final int pageIndex;
    private final BBox bbox;
    private final String text;
    private final ExtractReason reason;
    private final double confidence;
    private final boolean ocrUsed;
    private final @Nullable Table table;

    public RegionResult(
            int pageIndex,
            BBox bbox,
            String text,
            ExtractReason reason,
            double confidence,
            boolean ocrUsed,
            @Nullable Table table) {
        this.pageIndex = pageIndex;
        this.bbox = Objects.requireNonNull(bbox, "bbox");
        this.text = Objects.requireNonNull(text, "text");
        this.reason = Objects.requireNonNull(reason, "reason");
        this.confidence = confidence;
        this.ocrUsed = ocrUsed;
        this.table = table;
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

    public ExtractReason reason() {
        return reason;
    }

    public double confidence() {
        return confidence;
    }

    public boolean ocrUsed() {
        return ocrUsed;
    }

    /** @return reconstructed table, or empty if this region is not an image-table. */
    public Optional<Table> table() {
        return Optional.ofNullable(table);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof RegionResult)) return false;
        RegionResult r = (RegionResult) o;
        return pageIndex == r.pageIndex
                && Double.compare(r.confidence, confidence) == 0
                && ocrUsed == r.ocrUsed
                && bbox.equals(r.bbox)
                && text.equals(r.text)
                && reason == r.reason
                && Objects.equals(table, r.table);
    }

    @Override
    public int hashCode() {
        return Objects.hash(pageIndex, bbox, text, reason, confidence, ocrUsed, table);
    }

    @Override
    public String toString() {
        return "RegionResult[page=" + pageIndex + " reason=" + reason
                + " ocrUsed=" + ocrUsed + " conf=" + confidence
                + " bbox=" + bbox + " text=" + (text.length() > 40 ? text.substring(0, 37) + "..." : text)
                + "]";
    }
}
