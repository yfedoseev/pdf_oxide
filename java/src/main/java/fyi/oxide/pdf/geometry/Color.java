/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.geometry;

import java.util.Objects;

/**
 * 8-bit-per-channel RGBA color. Values are clamped at construction to
 * {@code [0, 255]} — callers passing out-of-range ints get a
 * {@link IllegalArgumentException}, matching {@link java.awt.Color}'s
 * convention.
 */
public final class Color {

    /** Pure black ({@code 0, 0, 0, 255}). */
    public static final Color BLACK = new Color(0, 0, 0, 255);
    /** Pure white ({@code 255, 255, 255, 255}). */
    public static final Color WHITE = new Color(255, 255, 255, 255);
    /** Fully transparent ({@code 0, 0, 0, 0}). */
    public static final Color TRANSPARENT = new Color(0, 0, 0, 0);

    private final int r;
    private final int g;
    private final int b;
    private final int a;

    /**
     * @param r red channel, 0-255 inclusive
     * @param g green channel, 0-255 inclusive
     * @param b blue channel, 0-255 inclusive
     * @param a alpha channel, 0-255 inclusive (0 = transparent, 255 = opaque)
     * @throws IllegalArgumentException if any channel is outside [0, 255]
     */
    public Color(int r, int g, int b, int a) {
        check(r, "r");
        check(g, "g");
        check(b, "b");
        check(a, "a");
        this.r = r;
        this.g = g;
        this.b = b;
        this.a = a;
    }

    /** Construct an opaque RGB color (alpha = 255). */
    public Color(int r, int g, int b) {
        this(r, g, b, 255);
    }

    private static void check(int v, String name) {
        if (v < 0 || v > 255) {
            throw new IllegalArgumentException(name + " must be in [0, 255], got " + v);
        }
    }

    public int r() {
        return r;
    }

    public int g() {
        return g;
    }

    public int b() {
        return b;
    }

    public int a() {
        return a;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Color)) return false;
        Color c = (Color) o;
        return r == c.r && g == c.g && b == c.b && a == c.a;
    }

    @Override
    public int hashCode() {
        return Objects.hash(r, g, b, a);
    }

    @Override
    public String toString() {
        if (a == 255) {
            return "Color[r=" + r + ", g=" + g + ", b=" + b + "]";
        }
        return "Color[r=" + r + ", g=" + g + ", b=" + b + ", a=" + a + "]";
    }
}
