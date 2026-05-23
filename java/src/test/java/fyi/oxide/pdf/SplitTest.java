/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import fyi.oxide.pdf.exception.PdfException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Tests {@link Pdf#splitByBookmarksFromBytes(byte[], int)} +
 * {@link Pdf#planSplitByBookmarksCount(byte[], int)} — the v0.3.50
 * #482 split-at-bookmarks feature wired through the byte[][]
 * return path.
 */
class SplitTest {

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
    void splitOnNoOutlineThrows() throws Exception {
        // simple.pdf has no /Outlines; the planner should reject
        // with a PdfException ("document has no bookmarks/outline").
        Path simple = fixturesDir.resolve("simple.pdf");
        byte[] bytes = Files.readAllBytes(simple);
        assertThatThrownBy(() -> Pdf.planSplitByBookmarksCount(bytes, 1)).isInstanceOf(PdfException.class);
        assertThatThrownBy(() -> Pdf.splitByBookmarksFromBytes(bytes, 1)).isInstanceOf(PdfException.class);
    }

    @Test
    void splitOnOutlinedPdfReturnsSegments() throws Exception {
        Path outlined = fixturesDir.resolve("outline.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(outlined), "outline.pdf not present");
        byte[] bytes = Files.readAllBytes(outlined);
        // Plan the count first.
        int count = Pdf.planSplitByBookmarksCount(bytes, 1);
        assertThat(count).isPositive();
        // Now produce the bytes.
        byte[][] segments = Pdf.splitByBookmarksFromBytes(bytes, 1);
        assertThat(segments).isNotNull();
        assertThat(segments.length).isEqualTo(count);
        for (byte[] seg : segments) {
            assertThat(seg).isNotEmpty();
            assertThat(new String(seg, 0, 5)).isEqualTo("%PDF-");
            // Round-trip: each segment should reopen as a valid PDF.
            try (PdfDocument doc = PdfDocument.open(seg)) {
                assertThat(doc.pageCount()).isPositive();
            }
        }
    }
}
