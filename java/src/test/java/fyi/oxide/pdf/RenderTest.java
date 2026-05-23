/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link PdfDocument#render(int)} and
 * {@link PdfDocument#render(int, int)}.
 *
 * <p>Requires the {@code pdf_oxide_jni} library to be built with
 * {@code --features rendering} (or {@code --features full}). The
 * Maven surefire run points at {@code target/release/libpdf_oxide_jni.so},
 * which must be the {@code full}-features build.
 */
class RenderTest {

    private static Path fixturesDir;

    @BeforeAll
    static void resolveFixtures() {
        fixturesDir = Paths.get("..")
                .resolve("tests")
                .resolve("fixtures")
                .toAbsolutePath()
                .normalize();
        org.junit.jupiter.api.Assumptions.assumeTrue(
                Files.isDirectory(fixturesDir), "fixtures dir not present: " + fixturesDir);
    }

    @Test
    void renderProducesPngBytes() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            byte[] png = doc.render(0);
            assertThat(png).isNotEmpty();
            // PNG magic: 89 50 4E 47 0D 0A 1A 0A
            assertThat(png[0] & 0xff).isEqualTo(0x89);
            assertThat(png[1]).isEqualTo((byte) 'P');
            assertThat(png[2]).isEqualTo((byte) 'N');
            assertThat(png[3]).isEqualTo((byte) 'G');
        }
    }

    @Test
    void renderHonorsDpi() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            byte[] low = doc.render(0, 72);
            byte[] high = doc.render(0, 300);
            // Higher DPI → larger PNG (more pixels).
            assertThat(high.length).isGreaterThan(low.length);
        }
    }

    @Test
    void renderRejectsNegativePageIndex() {
        Path simple = fixturesDir.resolve("simple.pdf");
        try (PdfDocument doc = PdfDocument.open(simple)) {
            assertThatThrownBy(() -> doc.render(-1)).isInstanceOf(IndexOutOfBoundsException.class);
        }
    }
}
