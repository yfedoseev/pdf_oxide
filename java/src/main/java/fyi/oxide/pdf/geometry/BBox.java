/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.geometry;

import java.util.Objects;

/**
 * Axis-aligned bounding box in PDF user-space coordinates.
 *
 * <p>Uses the PDF-spec coordinate convention: {@code (x0, y0)} is the
 * bottom-left, {@code (x1, y1)} is the top-right; the y-axis grows
 * upward. This matches the Rust core's {@code pdf_oxide_core::BBox},
 * NOT the screen / image convention where y grows downward.
 *
 * <p><b>Note on the JDK 11 floor:</b> this class is a {@code final
 * class} with record-shaped accessors. When the JDK floor moves to
 * 16+, the entire declaration can be replaced by
 * {@code public record BBox(double x0, double y0, double x1, double y1) {}}
 * without breaking ABI — every accessor method here has the same name
 * as the synthesised record accessor.
 */
public final class BBox {

    private final double x0;
    private final double y0;
    private final double x1;
    private final double y1;

    public BBox(double x0, double y0, double x1, double y1) {
        this.x0 = x0;
        this.y0 = y0;
        this.x1 = x1;
        this.y1 = y1;
    }

    /** @return left edge in PDF user space (typically &le; {@link #x1()}). */
    public double x0() {
        return x0;
    }
    /** @return bottom edge in PDF user space (typically &le; {@link #y1()}). */
    public double y0() {
        return y0;
    }
    /** @return right edge in PDF user space. */
    public double x1() {
        return x1;
    }
    /** @return top edge in PDF user space. */
    public double y1() {
        return y1;
    }

    /** @return width of the box ({@code x1 - x0}); negative if degenerate. */
    public double width() {
        return x1 - x0;
    }
    /** @return height of the box ({@code y1 - y0}); negative if degenerate. */
    public double height() {
        return y1 - y0;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof BBox)) return false;
        BBox b = (BBox) o;
        return Double.compare(b.x0, x0) == 0
                && Double.compare(b.y0, y0) == 0
                && Double.compare(b.x1, x1) == 0
                && Double.compare(b.y1, y1) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(x0, y0, x1, y1);
    }

    @Override
    public String toString() {
        return "BBox[x0=" + x0 + ", y0=" + y0 + ", x1=" + x1 + ", y1=" + y1 + "]";
    }
}
