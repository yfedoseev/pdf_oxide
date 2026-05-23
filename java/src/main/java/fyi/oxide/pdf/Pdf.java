/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import fyi.oxide.pdf.exception.PdfInvalidStateException;
import fyi.oxide.pdf.internal.NativeLoader;
import fyi.oxide.pdf.split.BookmarkSegment;
import fyi.oxide.pdf.split.SplitByBookmarksOptions;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Create / edit / save PDFs. Read-side concerns live on
 * {@link PdfDocument}; mutate concerns on {@link DocumentEditor};
 * creation + transformation (markdown→PDF, html→PDF, split) live
 * here.
 *
 * <p>{@code AutoCloseable} + idempotent close. Not thread-safe.
 *
 * <p><b>Status (v0.3.53)</b>: API surface complete; native bindings
 * stub until Phase 3 T12/T13.
 */
public final class Pdf implements AutoCloseable {

    static {
        NativeLoader.ensureLoaded();
    }

    private final AtomicLong handleState;

    private Pdf(long handle) {
        this.handleState = new AtomicLong(handle);
    }

    // ────────────────────── factories ──────────────────────

    /**
     * Create a PDF from a Markdown source. The generated PDF has
     * pdf_oxide's default page size and margins; heading levels,
     * bold/italic, monospace code, lists, links, and inline images
     * (data: URIs supported) are rendered per pdf_oxide's markdown
     * pipeline (v0.3.52 markdown→PDF styling restored, #525).
     */
    public static Pdf fromMarkdown(String markdown) {
        Objects.requireNonNull(markdown, "markdown");
        long h = nativeFromMarkdown(markdown);
        return new Pdf(h);
    }

    /** Create a PDF from an HTML source. CSS is honored per pdf_oxide's html_css pipeline. */
    public static Pdf fromHtml(String html) {
        Objects.requireNonNull(html, "html");
        long h = nativeFromHtml(html);
        return new Pdf(h);
    }

    /**
     * Build a multi-page PDF from a list of JPEG/PNG image byte
     * arrays. Each image becomes a separate page. Format is
     * auto-detected from the magic bytes.
     *
     * @throws IllegalArgumentException if the list is empty.
     * @throws fyi.oxide.pdf.exception.PdfParseException if any
     *         image's bytes can't be decoded (unsupported format,
     *         malformed JPEG/PNG).
     */
    public static Pdf fromImages(List<byte[]> images) {
        Objects.requireNonNull(images, "images");
        if (images.isEmpty()) {
            throw new IllegalArgumentException("at least one image is required");
        }
        byte[][] arr = images.toArray(new byte[0][]);
        long h = nativeFromImages(arr);
        return new Pdf(h);
    }

    // ────────────────────── transforms ─────────────────────

    /**
     * Compute the split plan (page ranges) without producing the
     * output bytes. Useful for previewing the split decisions.
     *
     * <p><b>v0.3.53 limitation</b>: returns an empty
     * {@link BookmarkSegment} list because the full segment-with-
     * metadata marshaller lands in a follow-up; for now use
     * {@link #planSplitByBookmarksCount(byte[], int)} for the count.
     */
    public List<BookmarkSegment> planSplitByBookmarks(SplitByBookmarksOptions opts) {
        Objects.requireNonNull(opts, "opts");
        throw new UnsupportedOperationException(
                "Pdf.planSplitByBookmarks(SplitByBookmarksOptions): Phase 3 T12 — segment marshaller TBD; use planSplitByBookmarksCount for the count");
    }

    /** Execute the split, returning one byte[] per output document. */
    public List<byte[]> splitByBookmarks(SplitByBookmarksOptions opts) {
        Objects.requireNonNull(opts, "opts");
        throw new UnsupportedOperationException(
                "Pdf.splitByBookmarks(SplitByBookmarksOptions): Phase 3 T12 — instance API needs source-PDF retention; use static splitByBookmarksFromBytes for now");
    }

    /**
     * Static convenience — count the bookmark-split segments that
     * would result, without producing the output PDFs.
     *
     * @param sourcePdf the PDF bytes to plan-split.
     * @param level     bookmark depth level (1 = top-level only,
     *                  2 = top + first sub-level, etc.; 0 = all).
     * @return the number of segments the split would produce.
     */
    public static int planSplitByBookmarksCount(byte[] sourcePdf, int level) {
        Objects.requireNonNull(sourcePdf, "sourcePdf");
        return nativePlanSplitCount(sourcePdf, level);
    }

    /**
     * Static convenience — split a PDF at bookmark boundaries.
     *
     * @param sourcePdf the PDF bytes to split.
     * @param level     bookmark depth level (1 = top-level only).
     * @return a {@code byte[][]} with one element per output
     *         segment, in document order. Source is not modified.
     */
    public static byte[][] splitByBookmarksFromBytes(byte[] sourcePdf, int level) {
        Objects.requireNonNull(sourcePdf, "sourcePdf");
        return nativeSplitBytes(sourcePdf, level);
    }

    // ─────────────────────── output ────────────────────────

    /** @return a fresh {@code byte[]} containing the generated PDF. */
    public byte[] save() {
        return nativeSaveBytes(checkHandle());
    }

    /** Write the generated PDF bytes to the given path. */
    public void saveTo(Path out) {
        Objects.requireNonNull(out, "out");
        try {
            java.nio.file.Files.write(out, save());
        } catch (java.io.IOException e) {
            throw new fyi.oxide.pdf.exception.PdfIoException("saveTo: " + out + ": " + e.getMessage(), e);
        }
    }

    // ─────────────────────── lifecycle ─────────────────────

    public boolean isOpen() {
        return handleState.get() != 0L;
    }

    @Override
    public void close() {
        final long h = handleState.getAndSet(0L);
        if (h != 0L) {
            nativeClose(h);
        }
    }

    private long checkHandle() {
        final long h = handleState.get();
        if (h == 0L) {
            throw new PdfInvalidStateException("Pdf has been closed");
        }
        return h;
    }

    private static native long nativeFromMarkdown(String markdown);

    private static native long nativeFromHtml(String html);

    private static native long nativeFromImages(byte[][] images);

    private static native byte[] nativeSaveBytes(long handle);

    private static native void nativeClose(long handle);

    private static native int nativePlanSplitCount(byte[] sourcePdf, int level);

    private static native byte[][] nativeSplitBytes(byte[] sourcePdf, int level);
}
