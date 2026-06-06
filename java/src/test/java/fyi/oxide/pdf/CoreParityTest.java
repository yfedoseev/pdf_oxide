/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import fyi.oxide.pdf.exception.PdfIoException;
import org.junit.jupiter.api.Test;

/**
 * Core functional test-parity suite (Java) — mirrors the shared cross-language
 * spec ({@code docs/releases/plans/v0.3.61/core-test-parity-spec.md}) with the
 * idiomatic Java API. Every binding asserts the same behaviors.
 *
 * <p>Each case is self-contained: it builds its own input via
 * {@link Pdf#fromMarkdown(String)} and opens it from bytes, so there is no
 * fixture-file dependency.
 */
class CoreParityTest {

    private static final String MARKDOWN = "# Core Parity\n\nFunctional parity across all language bindings.\n";

    private static byte[] buildBytes() {
        try (Pdf pdf = Pdf.fromMarkdown(MARKDOWN)) {
            return pdf.save();
        }
    }

    private static PdfDocument open() {
        return PdfDocument.open(buildBytes());
    }

    @Test
    void openAndPageCount() {
        try (PdfDocument doc = open()) {
            assertThat(doc.pageCount()).isGreaterThanOrEqualTo(1);
        }
    }

    @Test
    void extractTextReturnsString() {
        try (PdfDocument doc = open()) {
            assertThat(doc.extractText(0)).isNotNull();
        }
    }

    @Test
    void convertMarkdownAndHtmlReturnStrings() {
        try (PdfDocument doc = open()) {
            assertThat(doc.toMarkdown(0)).isNotNull();
            assertThat(doc.toHtml(0)).isNotNull();
        }
    }

    @Test
    void searchReturnsList() {
        try (PdfDocument doc = open()) {
            assertThat(doc.search("parity")).isNotNull();
        }
    }

    @Test
    void structuredExtraction() {
        try (PdfDocument doc = open()) {
            assertThat(doc.extractStructured(0)).isNotNull();
        }
    }

    @Test
    void createPdfFromMarkdown() {
        byte[] bytes = buildBytes();
        assertThat(bytes).hasSizeGreaterThan(4);
        assertThat(new String(bytes, 0, 5, java.nio.charset.StandardCharsets.ISO_8859_1))
                .isEqualTo("%PDF-");
    }

    @Test
    void openFromBytesPageCount() {
        try (PdfDocument doc = PdfDocument.open(buildBytes())) {
            assertThat(doc.pageCount()).isGreaterThanOrEqualTo(1);
        }
    }

    @Test
    void openingMissingPathThrows() {
        assertThatThrownBy(() -> PdfDocument.open("/no/such/file/does/not/exist.pdf"))
                .isInstanceOf(PdfIoException.class);
    }
}
