/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.search;

import org.jspecify.annotations.Nullable;

/**
 * Configuration for a {@link fyi.oxide.pdf.PdfDocument} text search.
 * Builder-driven.
 */
public final class SearchOptions {

    public static final SearchOptions DEFAULT = builder().build();

    private final boolean caseSensitive;
    private final boolean wholeWord;
    private final boolean regex;
    private final @Nullable Integer maxResults;

    private SearchOptions(Builder b) {
        this.caseSensitive = b.caseSensitive;
        this.wholeWord = b.wholeWord;
        this.regex = b.regex;
        this.maxResults = b.maxResults;
    }

    public boolean caseSensitive() {
        return caseSensitive;
    }

    public boolean wholeWord() {
        return wholeWord;
    }

    public boolean regex() {
        return regex;
    }

    public java.util.Optional<Integer> maxResults() {
        return java.util.Optional.ofNullable(maxResults);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private boolean caseSensitive = false;
        private boolean wholeWord = false;
        private boolean regex = false;
        private @Nullable Integer maxResults;

        public Builder withCaseSensitive(boolean b) {
            this.caseSensitive = b;
            return this;
        }

        public Builder withWholeWord(boolean b) {
            this.wholeWord = b;
            return this;
        }

        public Builder withRegex(boolean b) {
            this.regex = b;
            return this;
        }

        public Builder withMaxResults(@Nullable Integer m) {
            this.maxResults = m;
            return this;
        }

        public Builder withMaxResults(int m) {
            this.maxResults = m;
            return this;
        }

        public SearchOptions build() {
            return new SearchOptions(this);
        }
    }
}
