/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.split;

import java.util.Optional;
import org.jspecify.annotations.Nullable;

/**
 * Configuration for {@link fyi.oxide.pdf.Pdf#splitByBookmarks} per
 * v0.3.50 #482.
 */
public final class SplitByBookmarksOptions {

    private final int level;
    private final @Nullable String filenamePrefix;

    private SplitByBookmarksOptions(Builder b) {
        this.level = b.level;
        this.filenamePrefix = b.filenamePrefix;
    }

    /** @return bookmark level to split at (1 = top-level only, 2 = next level, …). */
    public int level() {
        return level;
    }

    public Optional<String> filenamePrefix() {
        return Optional.ofNullable(filenamePrefix);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private int level = 1;
        private @Nullable String filenamePrefix;

        public Builder withLevel(int l) {
            this.level = l;
            return this;
        }

        public Builder withFilenamePrefix(@Nullable String p) {
            this.filenamePrefix = p;
            return this;
        }

        public SplitByBookmarksOptions build() {
            return new SplitByBookmarksOptions(this);
        }
    }
}
