/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.split;

import java.util.Objects;

/**
 * One segment of a split plan: the bookmark title + the (inclusive)
 * page range to extract. The output file name is
 * {@code "{prefix}_{title-slug}.pdf"} when a prefix is configured;
 * otherwise {@code "{title-slug}.pdf"}.
 */
public final class BookmarkSegment {

    private final String title;
    private final int firstPage;
    private final int lastPage;
    private final String filename;

    public BookmarkSegment(String title, int firstPage, int lastPage, String filename) {
        this.title = Objects.requireNonNull(title, "title");
        this.firstPage = firstPage;
        this.lastPage = lastPage;
        this.filename = Objects.requireNonNull(filename, "filename");
    }

    public String title() {
        return title;
    }
    /** @return 0-based first page index (inclusive). */
    public int firstPage() {
        return firstPage;
    }
    /** @return 0-based last page index (inclusive). */
    public int lastPage() {
        return lastPage;
    }

    public String filename() {
        return filename;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof BookmarkSegment)) return false;
        BookmarkSegment s = (BookmarkSegment) o;
        return firstPage == s.firstPage
                && lastPage == s.lastPage
                && title.equals(s.title)
                && filename.equals(s.filename);
    }

    @Override
    public int hashCode() {
        return Objects.hash(title, firstPage, lastPage, filename);
    }

    @Override
    public String toString() {
        return "BookmarkSegment[title=" + title
                + " pages=[" + firstPage + "," + lastPage + "]"
                + " filename=" + filename + "]";
    }
}
