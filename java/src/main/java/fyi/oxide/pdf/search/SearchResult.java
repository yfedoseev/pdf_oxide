/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.search;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Result of a text search across a {@link fyi.oxide.pdf.PdfDocument}.
 */
public final class SearchResult {

    private final List<SearchMatch> matches;
    private final String query;

    public SearchResult(String query, List<SearchMatch> matches) {
        this.query = Objects.requireNonNull(query, "query");
        this.matches =
                Collections.unmodifiableList(new java.util.ArrayList<>(Objects.requireNonNull(matches, "matches")));
    }

    public String query() {
        return query;
    }

    public List<SearchMatch> matches() {
        return matches;
    }

    public int count() {
        return matches.size();
    }

    public boolean isEmpty() {
        return matches.isEmpty();
    }
}
