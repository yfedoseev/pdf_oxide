/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.image;

import fyi.oxide.pdf.geometry.BBox;
import java.util.Objects;

/**
 * An image extracted from a PDF page. Carries the raw bytes in the
 * native PDF stream format (no decoding is performed Rust-side) and
 * the on-page placement bbox in PDF user-space coordinates.
 *
 * <p>Decode to a {@link java.awt.image.BufferedImage} on the Java
 * side with the format-appropriate ImageIO reader. JPEG and PNG
 * decode out of the box; JBIG2 / JPEG2000 / CCITT need an
 * additional reader plugin.
 */
public final class ExtractedImage {
    private final byte[] bytes;
    private final ImageFormat format;
    private final BBox bbox;
    private final int width;
    private final int height;

    public ExtractedImage(byte[] bytes, ImageFormat format, BBox bbox, int width, int height) {
        Objects.requireNonNull(bytes, "bytes");
        this.bytes = bytes.clone(); // defensive copy
        this.format = Objects.requireNonNull(format, "format");
        this.bbox = Objects.requireNonNull(bbox, "bbox");
        this.width = width;
        this.height = height;
    }

    /** @return defensive copy of the encoded image bytes. */
    public byte[] bytes() {
        return bytes.clone();
    }

    public ImageFormat format() {
        return format;
    }
    /** @return on-page placement in PDF user-space coordinates. */
    public BBox bbox() {
        return bbox;
    }
    /** @return image pixel width. */
    public int width() {
        return width;
    }
    /** @return image pixel height. */
    public int height() {
        return height;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ExtractedImage)) return false;
        ExtractedImage img = (ExtractedImage) o;
        return width == img.width
                && height == img.height
                && format == img.format
                && bbox.equals(img.bbox)
                && java.util.Arrays.equals(bytes, img.bytes);
    }

    @Override
    public int hashCode() {
        int h = Objects.hash(format, bbox, width, height);
        return 31 * h + java.util.Arrays.hashCode(bytes);
    }

    @Override
    public String toString() {
        return "ExtractedImage[" + format + " " + width + "x" + height + " " + bytes.length + " bytes, bbox=" + bbox
                + "]";
    }
}
