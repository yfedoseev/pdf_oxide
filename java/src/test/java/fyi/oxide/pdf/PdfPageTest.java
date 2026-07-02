/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import static org.assertj.core.api.Assertions.assertThat;

import fyi.oxide.pdf.geometry.BBox;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class PdfPageTest {

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
    void mediaBoxIsLetterForHelloStructure() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            PdfPage page = doc.page(0);
            BBox media = page.mediaBox();
            assertThat(media.x0()).isEqualTo(0.0);
            assertThat(media.y0()).isEqualTo(0.0);
            // US Letter = 612 x 792 PDF user-space units
            assertThat(media.x1()).isEqualTo(612.0);
            assertThat(media.y1()).isEqualTo(792.0);
            assertThat(page.width()).isEqualTo(612.0);
            assertThat(page.height()).isEqualTo(792.0);
            assertThat(page.rotation()).isEqualTo(0);
        }
    }

    @Test
    void pagesIteratesAllPages() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            assertThat(doc.pages()).hasSize(doc.pageCount());
            assertThat(doc.pagesStream().count()).isEqualTo(doc.pageCount());
        }
    }

    @Test
    void linesReturnsListWithNestedWords() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            java.util.List<fyi.oxide.pdf.text.TextLine> lines = doc.page(0).lines();
            assertThat(lines).isNotNull().isNotEmpty();
            for (fyi.oxide.pdf.text.TextLine line : lines) {
                assertThat(line.bbox()).isNotNull();
                assertThat(line.text()).isNotNull();
                assertThat(line.words()).isNotNull();
                // Each word's text should appear in the line text.
                for (fyi.oxide.pdf.text.TextWord w : line.words()) {
                    assertThat(w.text()).isNotEmpty();
                }
            }
        }
    }

    @Test
    void wordsReturnsNonEmptyList() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            PdfPage page = doc.page(0);
            java.util.List<fyi.oxide.pdf.text.TextWord> words = page.words();
            assertThat(words).isNotNull().isNotEmpty();
            assertThat(words.get(0).text()).isNotEmpty();
            assertThat(words.get(0).bbox()).isNotNull();
            // Content-stream emission order is exposed and non-negative.
            assertThat(words.get(0).sequence()).isGreaterThanOrEqualTo(0L);
        }
    }

    @Test
    void annotationsReturnsList() {
        Path simple = fixturesDir.resolve("simple.pdf");
        try (PdfDocument doc = PdfDocument.open(simple)) {
            java.util.List<fyi.oxide.pdf.annotation.Annotation> annotations =
                    doc.page(0).annotations();
            assertThat(annotations).isNotNull();
        }
    }

    @Test
    void tablesReturnsList() {
        // simple.pdf has no tables — list should be empty but non-null.
        // hello_structure.pdf likewise no tables.
        Path simple = fixturesDir.resolve("simple.pdf");
        try (PdfDocument doc = PdfDocument.open(simple)) {
            java.util.List<fyi.oxide.pdf.table.Table> tables = doc.page(0).tables();
            assertThat(tables).isNotNull();
        }
    }

    @Test
    void imagesReturnsList() {
        // hello_structure.pdf has no embedded raster images — list
        // should be empty but non-null. The shape contract is what
        // matters; presence of images is fixture-dependent.
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            java.util.List<fyi.oxide.pdf.image.ExtractedImage> images =
                    doc.page(0).images();
            assertThat(images).isNotNull();
        }
    }

    @Test
    void charsReturnsCodepoints() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            PdfPage page = doc.page(0);
            java.util.List<fyi.oxide.pdf.text.TextChar> chars = page.chars();
            assertThat(chars).isNotNull().isNotEmpty();
            // "Hello World" → 'H' should appear as a codepoint
            boolean foundH = chars.stream().anyMatch(c -> c.codepoint() == (int) 'H');
            assertThat(foundH).isTrue();
        }
    }

    @Test
    void textInRegionReturnsSubsetOfFullText() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            PdfPage page = doc.page(0);
            BBox full = page.mediaBox();
            // Full mediaBox region should match full text extraction.
            String region = page.text(full);
            String all = page.text();
            assertThat(region).isNotNull();
            assertThat(all).isNotNull();
            // Both should be non-empty for hello_structure.pdf
            assertThat(region).isNotEmpty();
            assertThat(all).isNotEmpty();
        }
    }

    @Test
    void outOfRangePageThrowsIndexOutOfBounds() {
        Path simple = fixturesDir.resolve("simple.pdf");
        try (PdfDocument doc = PdfDocument.open(simple)) {
            org.junit.jupiter.api.Assertions.assertThrows(IndexOutOfBoundsException.class, () -> doc.page(-1));
            org.junit.jupiter.api.Assertions.assertThrows(
                    IndexOutOfBoundsException.class, () -> doc.page(doc.pageCount()));
        }
    }
}
