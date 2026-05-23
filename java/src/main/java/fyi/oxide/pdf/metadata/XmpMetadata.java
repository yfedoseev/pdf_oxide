/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.metadata;

import java.util.Objects;

/**
 * Raw XMP metadata stream from a PDF (XML-RDF). Consumers parse
 * via their own XMP/XML library — the binding doesn't impose a
 * particular dependency.
 */
public final class XmpMetadata {

    /** Empty XMP — returned when no XMP stream is present. */
    public static final XmpMetadata EMPTY = new XmpMetadata("");

    private final String xml;

    public XmpMetadata(String xml) {
        this.xml = Objects.requireNonNull(xml, "xml");
    }

    /** @return raw XMP XML (may be empty). */
    public String xml() {
        return xml;
    }

    public boolean isEmpty() {
        return xml.isEmpty();
    }
}
