/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import fyi.oxide.pdf.geometry.BBox;
import fyi.oxide.pdf.internal.NativeLoader;
import fyi.oxide.pdf.text.TextChar;
import fyi.oxide.pdf.text.TextLine;
import fyi.oxide.pdf.text.TextWord;
import java.util.List;
import java.util.Objects;

/**
 * A page within a {@link PdfDocument}, identified by its 0-based
 * page index.
 *
 * <p>{@code PdfPage} is a lightweight view — it holds no native
 * handle of its own; it borrows from its parent {@link PdfDocument}.
 * Calls on a {@code PdfPage} after the parent document's
 * {@link PdfDocument#close()} throw
 * {@link fyi.oxide.pdf.exception.PdfInvalidStateException}.
 *
 * <p>Construction is package-private: obtain a {@code PdfPage} via
 * {@link PdfDocument#page(int)} or by iterating
 * {@link PdfDocument#pages()}.
 */
public final class PdfPage {

    static {
        NativeLoader.ensureLoaded();
    }

    private final PdfDocument parent;
    private final int index;

    PdfPage(PdfDocument parent, int index) {
        this.parent = Objects.requireNonNull(parent, "parent");
        this.index = index;
    }

    /** @return owning document; useful for re-acquiring shared state. */
    public PdfDocument parent() {
        return parent;
    }

    /** @return 0-based page index. */
    public int index() {
        return index;
    }

    /** @return the {@code /MediaBox} entry in PDF user-space coordinates. */
    public BBox mediaBox() {
        return readBBox(true);
    }

    /**
     * @return the {@code /CropBox}, or {@link #mediaBox()} if absent.
     *         v0.3.53: returns {@link #mediaBox()} unconditionally —
     *         dedicated crop-box access is a follow-up
     *         (pdf_oxide core's {@code get_page_crop_box} not yet
     *         public; tracked in a future v0.3.54 issue).
     */
    public BBox cropBox() {
        return mediaBox();
    }

    /** @return page width in PDF user-space units. */
    public double width() {
        BBox m = mediaBox();
        return m.width();
    }

    /** @return page height in PDF user-space units. */
    public double height() {
        BBox m = mediaBox();
        return m.height();
    }

    /** @return clockwise page rotation in degrees (0, 90, 180, 270). */
    public int rotation() {
        return nativeRotation(parent.requireHandleForCallers(), index);
    }

    /**
     * @return extracted text for this page (same as
     *         {@link PdfDocument#extractText(int)}).
     */
    public String text() {
        return parent.extractText(index);
    }

    /**
     * Extract text within a region of this page (PDF user-space
     * coordinates; y grows upward).
     *
     * @param region the rectangular region in PDF user-space.
     * @return text contained in the region.
     */
    public String text(BBox region) {
        java.util.Objects.requireNonNull(region, "region");
        return nativeTextInRect(
                parent.requireHandleForCallers(), index, region.x0(), region.y0(), region.x1(), region.y1());
    }

    /** @return list of words on this page, in reading order. */
    public List<TextWord> words() {
        return nativeWords(parent.requireHandleForCallers(), index);
    }

    /** @return list of text lines on this page, in reading order. */
    public List<TextLine> lines() {
        return nativeLines(parent.requireHandleForCallers(), index);
    }

    /** @return list of characters on this page, in reading order. */
    public List<TextChar> chars() {
        return nativeChars(parent.requireHandleForCallers(), index);
    }

    /**
     * @return list of raster images embedded in this page. Each
     *         {@link fyi.oxide.pdf.image.ExtractedImage} carries the
     *         encoded bytes (JPEG or raw pixels per {@link
     *         fyi.oxide.pdf.image.ImageFormat}), pixel dimensions,
     *         and on-page placement bbox (zero-rect if unknown).
     */
    public List<fyi.oxide.pdf.image.ExtractedImage> images() {
        return nativeImages(parent.requireHandleForCallers(), index);
    }

    /**
     * @return list of tables on this page. Each
     *         {@link fyi.oxide.pdf.table.Table} carries a flat
     *         list of cells with explicit row/column indices and
     *         spans.
     */
    public List<fyi.oxide.pdf.table.Table> tables() {
        return nativeTables(parent.requireHandleForCallers(), index);
    }

    /**
     * @return list of annotations on this page (highlights, text
     *         notes, links, stamps, etc.). Annotations with subtypes
     *         not yet exposed by the binding bucket as
     *         {@link fyi.oxide.pdf.annotation.AnnotationType#OTHER}.
     */
    public List<fyi.oxide.pdf.annotation.Annotation> annotations() {
        return nativeAnnotations(parent.requireHandleForCallers(), index);
    }

    @Override
    public String toString() {
        return "PdfPage[index=" + index + "]";
    }

    /**
     * Helper: read the {@code /MediaBox} or {@code /CropBox} via JNI.
     * The native side returns 4 doubles via a fresh {@code double[4]}
     * to keep the FFI surface tight (no need for a {@link BBox}
     * Java object to be constructible from JNI).
     */
    private BBox readBBox(boolean media) {
        double[] xy = nativeReadBBox(parent.requireHandleForCallers(), index, media);
        return new BBox(xy[0], xy[1], xy[2], xy[3]);
    }

    // ─────────────────────── native ────────────────────────

    /** Returns {@code double[]{x0, y0, x1, y1}} for the requested box. */
    private static native double[] nativeReadBBox(long handle, int pageIndex, boolean media);

    private static native int nativeRotation(long handle, int pageIndex);

    private static native String nativeTextInRect(
            long handle, int pageIndex, double x0, double y0, double x1, double y1);

    private static native List<TextWord> nativeWords(long handle, int pageIndex);

    private static native List<TextLine> nativeLines(long handle, int pageIndex);

    private static native List<TextChar> nativeChars(long handle, int pageIndex);

    private static native List<fyi.oxide.pdf.image.ExtractedImage> nativeImages(long handle, int pageIndex);

    private static native List<fyi.oxide.pdf.table.Table> nativeTables(long handle, int pageIndex);

    private static native List<fyi.oxide.pdf.annotation.Annotation> nativeAnnotations(long handle, int pageIndex);
}
