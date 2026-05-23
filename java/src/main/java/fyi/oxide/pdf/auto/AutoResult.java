/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.auto;

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import org.jspecify.annotations.Nullable;

/**
 * Result of an {@link fyi.oxide.pdf.AutoExtractor} extraction.
 *
 * <p>The v0.3.51 graceful-fallback contract: this object is
 * <b>never</b> null and is always populated with the best-effort
 * native text — even when OCR is unavailable. Check
 * {@link #reason()} to discover degradation; see the
 * "feedback_extraction_graceful_fallback" project memory.
 */
public final class AutoResult {
    private final String text;
    private final @Nullable String markdown;
    private final @Nullable String html;
    private final ExtractReason reason;
    private final double confidence;
    private final boolean ocrUsed;
    private final List<RegionResult> regions;
    private final List<Integer> pagesNeedingOcr;

    public AutoResult(
            String text,
            @Nullable String markdown,
            @Nullable String html,
            ExtractReason reason,
            double confidence,
            boolean ocrUsed,
            List<RegionResult> regions,
            List<Integer> pagesNeedingOcr) {
        this.text = Objects.requireNonNull(text, "text");
        this.markdown = markdown;
        this.html = html;
        this.reason = Objects.requireNonNull(reason, "reason");
        this.confidence = confidence;
        this.ocrUsed = ocrUsed;
        this.regions =
                Collections.unmodifiableList(new java.util.ArrayList<>(Objects.requireNonNull(regions, "regions")));
        this.pagesNeedingOcr = Collections.unmodifiableList(
                new java.util.ArrayList<>(Objects.requireNonNull(pagesNeedingOcr, "pagesNeedingOcr")));
    }

    public String text() {
        return text;
    }
    /** @return markdown rendering of the same content, if requested. */
    public Optional<String> markdown() {
        return Optional.ofNullable(markdown);
    }
    /** @return HTML rendering, if requested. */
    public Optional<String> html() {
        return Optional.ofNullable(html);
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
    /** @return per-region results in document order. */
    public List<RegionResult> regions() {
        return regions;
    }
    /** @return list of 0-based page indices the classifier flagged as needing OCR. */
    public List<Integer> pagesNeedingOcr() {
        return pagesNeedingOcr;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof AutoResult)) return false;
        AutoResult r = (AutoResult) o;
        return Double.compare(r.confidence, confidence) == 0
                && ocrUsed == r.ocrUsed
                && text.equals(r.text)
                && Objects.equals(markdown, r.markdown)
                && Objects.equals(html, r.html)
                && reason == r.reason
                && regions.equals(r.regions)
                && pagesNeedingOcr.equals(r.pagesNeedingOcr);
    }

    @Override
    public int hashCode() {
        return Objects.hash(text, markdown, html, reason, confidence, ocrUsed, regions, pagesNeedingOcr);
    }

    @Override
    public String toString() {
        return "AutoResult[reason=" + reason
                + " ocrUsed=" + ocrUsed
                + " confidence=" + confidence
                + " regions=" + regions.size()
                + " textLen=" + text.length() + "]";
    }
}
