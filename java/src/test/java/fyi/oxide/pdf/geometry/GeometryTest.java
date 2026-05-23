/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.geometry;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

/**
 * Pure-Java tests for the geometry value types. No native code.
 */
class GeometryTest {

    @Test
    void bboxComputesWidthAndHeight() {
        BBox b = new BBox(10, 20, 100, 200);
        assertThat(b.width()).isEqualTo(90.0);
        assertThat(b.height()).isEqualTo(180.0);
        assertThat(b.x0()).isEqualTo(10.0);
        assertThat(b.x1()).isEqualTo(100.0);
    }

    @Test
    void bboxEqualsAndHashCode() {
        BBox a = new BBox(1, 2, 3, 4);
        BBox b = new BBox(1, 2, 3, 4);
        BBox c = new BBox(1, 2, 3, 5);
        assertThat(a).isEqualTo(b).hasSameHashCodeAs(b);
        assertThat(a).isNotEqualTo(c);
    }

    @Test
    void pointEquality() {
        assertThat(new Point(1.0, 2.0)).isEqualTo(new Point(1.0, 2.0));
        assertThat(new Point(1.0, 2.0)).isNotEqualTo(new Point(2.0, 1.0));
    }

    @Test
    void rectConvertsToBBox() {
        Rect r = new Rect(10, 20, 30, 40);
        BBox b = r.toBBox();
        assertThat(b.x0()).isEqualTo(10.0);
        assertThat(b.y0()).isEqualTo(20.0);
        assertThat(b.x1()).isEqualTo(40.0); // x + w
        assertThat(b.y1()).isEqualTo(60.0); // y + h
    }

    @Test
    void colorClampsRejectOutOfRange() {
        assertThatThrownBy(() -> new Color(-1, 0, 0)).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new Color(0, 256, 0)).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new Color(0, 0, -10)).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void colorConstants() {
        assertThat(Color.BLACK.r()).isEqualTo(0);
        assertThat(Color.BLACK.a()).isEqualTo(255);
        assertThat(Color.WHITE.r()).isEqualTo(255);
        assertThat(Color.TRANSPARENT.a()).isEqualTo(0);
    }

    @Test
    void colorToStringOmitsAlphaIfOpaque() {
        assertThat(new Color(1, 2, 3).toString()).doesNotContain("a=");
        assertThat(new Color(1, 2, 3, 128).toString()).contains("a=128");
    }
}
