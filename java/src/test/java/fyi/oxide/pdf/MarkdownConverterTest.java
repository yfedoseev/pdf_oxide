/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class MarkdownConverterTest {

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
    void toMarkdownProducesHeading() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            String md = MarkdownConverter.toMarkdown(doc, 0);
            assertThat(md).contains("# "); // tagged heading
            assertThat(md).containsIgnoringCase("hello");
        }
    }

    @Test
    void toHtmlProducesContent() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            String html = MarkdownConverter.toHtml(doc, 0);
            assertThat(html).isNotEmpty();
        }
    }

    @Test
    void docConvenienceMethodsMatchConverterStatics() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            assertThat(doc.toMarkdown(0)).isEqualTo(MarkdownConverter.toMarkdown(doc, 0));
            assertThat(doc.toHtml(0)).isEqualTo(MarkdownConverter.toHtml(doc, 0));
            assertThat(doc.toMarkdown()).isEqualTo(MarkdownConverter.toMarkdown(doc));
            assertThat(doc.toHtml()).isEqualTo(MarkdownConverter.toHtml(doc));
        }
    }
}
