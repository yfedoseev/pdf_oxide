/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.annotation;

import fyi.oxide.pdf.geometry.BBox;
import java.util.Objects;
import java.util.Optional;
import org.jspecify.annotations.Nullable;

/**
 * A PDF annotation as read from a page. Carries the subtype, on-page
 * placement bbox, optional contents (the popup text or label), and
 * optional URI for {@link AnnotationType#LINK} subtype.
 */
public final class Annotation {
    private final AnnotationType type;
    private final int pageIndex;
    private final BBox bbox;
    private final @Nullable String contents;
    private final @Nullable String uri;

    public Annotation(AnnotationType type, int pageIndex, BBox bbox, @Nullable String contents, @Nullable String uri) {
        this.type = Objects.requireNonNull(type, "type");
        this.pageIndex = pageIndex;
        this.bbox = Objects.requireNonNull(bbox, "bbox");
        this.contents = contents;
        this.uri = uri;
    }

    public AnnotationType type() {
        return type;
    }

    public int pageIndex() {
        return pageIndex;
    }

    public BBox bbox() {
        return bbox;
    }
    /** @return annotation contents (popup text, label, etc.). */
    public Optional<String> contents() {
        return Optional.ofNullable(contents);
    }
    /** @return URI for {@link AnnotationType#LINK} annotations. */
    public Optional<String> uri() {
        return Optional.ofNullable(uri);
    }

    @Override
    public String toString() {
        return "Annotation[" + type + " page=" + pageIndex + " bbox=" + bbox
                + (contents == null ? "" : " contents=" + contents)
                + (uri == null ? "" : " uri=" + uri) + "]";
    }
}
