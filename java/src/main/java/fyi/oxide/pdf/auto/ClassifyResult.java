/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.auto;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Result of {@link fyi.oxide.pdf.AutoExtractor#classifyDocument()} —
 * the cheap preflight that decides which pages need OCR / which need
 * image-table reconstruction, before the heavy extraction pass.
 *
 * <p>The cost model: classification is &lt; 5% of a plain text
 * extract on born-digital pages, per v0.3.51 performance budget
 * ({@code 00-common-foundation.md} §6).
 */
public final class ClassifyResult {
    private final List<PageClass> pages;
    private final List<Integer> pagesNeedingOcr;
    private final List<Integer> pagesWithChart;
    private final List<Integer> pagesEncrypted;

    public ClassifyResult(
            List<PageClass> pages,
            List<Integer> pagesNeedingOcr,
            List<Integer> pagesWithChart,
            List<Integer> pagesEncrypted) {
        this.pages = Collections.unmodifiableList(new java.util.ArrayList<>(Objects.requireNonNull(pages, "pages")));
        this.pagesNeedingOcr = Collections.unmodifiableList(
                new java.util.ArrayList<>(Objects.requireNonNull(pagesNeedingOcr, "pagesNeedingOcr")));
        this.pagesWithChart = Collections.unmodifiableList(
                new java.util.ArrayList<>(Objects.requireNonNull(pagesWithChart, "pagesWithChart")));
        this.pagesEncrypted = Collections.unmodifiableList(
                new java.util.ArrayList<>(Objects.requireNonNull(pagesEncrypted, "pagesEncrypted")));
    }

    /** @return per-page classification (size == pageCount). */
    public List<PageClass> pages() {
        return pages;
    }
    /** @return 0-based page indices the classifier flagged for OCR routing. */
    public List<Integer> pagesNeedingOcr() {
        return pagesNeedingOcr;
    }
    /** @return 0-based page indices the classifier flagged as containing charts (not transcribed). */
    public List<Integer> pagesWithChart() {
        return pagesWithChart;
    }
    /** @return 0-based page indices where extraction is permission-denied. */
    public List<Integer> pagesEncrypted() {
        return pagesEncrypted;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ClassifyResult)) return false;
        ClassifyResult r = (ClassifyResult) o;
        return pages.equals(r.pages)
                && pagesNeedingOcr.equals(r.pagesNeedingOcr)
                && pagesWithChart.equals(r.pagesWithChart)
                && pagesEncrypted.equals(r.pagesEncrypted);
    }

    @Override
    public int hashCode() {
        return Objects.hash(pages, pagesNeedingOcr, pagesWithChart, pagesEncrypted);
    }

    @Override
    public String toString() {
        return "ClassifyResult[" + pages.size() + " pages, "
                + pagesNeedingOcr.size() + " need OCR, "
                + pagesWithChart.size() + " with chart, "
                + pagesEncrypted.size() + " encrypted]";
    }
}
