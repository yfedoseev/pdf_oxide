/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.geometry;

import java.util.Objects;

/**
 * Rectangle in {@code (x, y, width, height)} form. Differs from
 * {@link BBox} (which uses {@code (x0, y0, x1, y1)}) for callers that
 * prefer the graphics-style {@code x/y/w/h} convention.
 *
 * <p>Y grows upward (PDF spec). See {@link BBox} for the convention.
 */
public final class Rect {
    private final double x;
    private final double y;
    private final double width;
    private final double height;

    public Rect(double x, double y, double width, double height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    public double x() {
        return x;
    }

    public double y() {
        return y;
    }

    public double width() {
        return width;
    }

    public double height() {
        return height;
    }

    /** @return equivalent {@link BBox} with {@code (x0=x, y0=y, x1=x+w, y1=y+h)}. */
    public BBox toBBox() {
        return new BBox(x, y, x + width, y + height);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Rect)) return false;
        Rect r = (Rect) o;
        return Double.compare(r.x, x) == 0
                && Double.compare(r.y, y) == 0
                && Double.compare(r.width, width) == 0
                && Double.compare(r.height, height) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y, width, height);
    }

    @Override
    public String toString() {
        return "Rect[x=" + x + ", y=" + y + ", w=" + width + ", h=" + height + "]";
    }
}
