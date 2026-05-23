/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import fyi.oxide.pdf.internal.NativeLoader;
import java.util.Objects;

/**
 * Static converters from a {@link PdfDocument} to Markdown or HTML.
 *
 * <p>Thread-safe (the methods are stateless static; the underlying
 * Rust call takes a borrowed {@code &PdfDocument}, and per
 * {@code 00-common-foundation.md} §2.7 a {@code PdfDocument} handle
 * is single-threaded — so a caller must not invoke a converter
 * concurrently against the same document, but two threads each with
 * their own document are fine).
 *
 * <p>v0.3.53 ships the per-page and whole-document converters with
 * default conversion options. Tunable options (table extraction
 * toggle, image-embedding mode, heading inference) come in a follow-
 * up issue (see {@code api-design.md} §7).
 */
public final class MarkdownConverter {

    static {
        NativeLoader.ensureLoaded();
    }

    private MarkdownConverter() {
        // Static-only.
    }

    /**
     * Convert a single page to Markdown.
     *
     * @param doc       open {@link PdfDocument} (must not be closed).
     * @param pageIndex 0-based page index.
     * @return Markdown representation of the page.
     */
    public static String toMarkdown(PdfDocument doc, int pageIndex) {
        Objects.requireNonNull(doc, "doc");
        return nativeToMarkdownPage(doc.requireHandleForCallers(), pageIndex);
    }

    /**
     * Convert the entire document to Markdown.
     *
     * @param doc open {@link PdfDocument} (must not be closed).
     * @return Markdown representation of the whole document.
     */
    public static String toMarkdown(PdfDocument doc) {
        Objects.requireNonNull(doc, "doc");
        return nativeToMarkdownAll(doc.requireHandleForCallers());
    }

    /**
     * Convert a single page to HTML.
     *
     * @param doc       open {@link PdfDocument} (must not be closed).
     * @param pageIndex 0-based page index.
     * @return HTML representation of the page.
     */
    public static String toHtml(PdfDocument doc, int pageIndex) {
        Objects.requireNonNull(doc, "doc");
        return nativeToHtmlPage(doc.requireHandleForCallers(), pageIndex);
    }

    /**
     * Convert the entire document to HTML.
     *
     * @param doc open {@link PdfDocument} (must not be closed).
     * @return HTML representation of the whole document.
     */
    public static String toHtml(PdfDocument doc) {
        Objects.requireNonNull(doc, "doc");
        return nativeToHtmlAll(doc.requireHandleForCallers());
    }

    // ─────────────────────── native ────────────────────────

    private static native String nativeToMarkdownPage(long handle, int pageIndex);

    private static native String nativeToMarkdownAll(long handle);

    private static native String nativeToHtmlPage(long handle, int pageIndex);

    private static native String nativeToHtmlAll(long handle);
}
