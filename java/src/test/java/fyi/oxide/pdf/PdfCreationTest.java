/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import fyi.oxide.pdf.exception.PdfInvalidStateException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/**
 * Tests for the Markdown→PDF and HTML→PDF creation surface.
 * Round-trips a small Markdown document → PDF bytes → reopen via
 * {@link PdfDocument#open(byte[])} → confirm at least one page,
 * non-empty text.
 */
class PdfCreationTest {

    @Test
    void fromMarkdownProducesValidPdf() {
        String md = "# Hello\n\nThis is **bold** text and *italic* text.\n";
        try (Pdf pdf = Pdf.fromMarkdown(md)) {
            byte[] bytes = pdf.save();
            assertThat(bytes).isNotEmpty();
            // PDF header magic — every valid PDF starts with %PDF-
            assertThat(new String(bytes, 0, Math.min(5, bytes.length))).isEqualTo("%PDF-");

            // Round-trip: reopen the generated PDF and verify content.
            try (PdfDocument doc = PdfDocument.open(bytes)) {
                assertThat(doc.pageCount()).isGreaterThan(0);
                String extracted = doc.extractText(0);
                assertThat(extracted).containsIgnoringCase("hello");
                assertThat(extracted).containsIgnoringCase("bold");
                assertThat(extracted).containsIgnoringCase("italic");
            }
        }
    }

    @Test
    void fromHtmlProducesValidPdf() {
        String html = "<html><body><h1>Hi</h1><p>HTML content</p></body></html>";
        try (Pdf pdf = Pdf.fromHtml(html)) {
            byte[] bytes = pdf.save();
            assertThat(bytes).isNotEmpty();
            assertThat(new String(bytes, 0, Math.min(5, bytes.length))).isEqualTo("%PDF-");
        }
    }

    @Test
    void saveToWritesFile() throws Exception {
        Path tmp = Files.createTempFile("pdf-oxide-jni-create-", ".pdf");
        try {
            try (Pdf pdf = Pdf.fromMarkdown("# T\n\nContent.\n")) {
                pdf.saveTo(tmp);
            }
            assertThat(Files.size(tmp)).isGreaterThan(0);
            byte[] header = Files.readAllBytes(tmp);
            assertThat(new String(header, 0, 5)).isEqualTo("%PDF-");
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    @Test
    void saveAfterCloseThrowsInvalidState() {
        Pdf pdf = Pdf.fromMarkdown("# X\n");
        pdf.close();
        assertThat(pdf.isOpen()).isFalse();
        assertThatThrownBy(pdf::save).isInstanceOf(PdfInvalidStateException.class);
    }

    @Test
    void fromImagesRoundTrips() {
        // Generate a PDF from markdown, render its page to PNG bytes,
        // then build a NEW PDF from that PNG → confirms fromImages
        // works end-to-end with real image data.
        byte[] pngBytes;
        try (Pdf src = Pdf.fromMarkdown("# Test Page\n\nContent.\n");
                PdfDocument srcDoc = PdfDocument.open(src.save())) {
            pngBytes = srcDoc.render(0);
        }
        assertThat(pngBytes).isNotEmpty();
        // Now feed the PNG to fromImages.
        try (Pdf imgPdf = Pdf.fromImages(java.util.List.of(pngBytes));
                PdfDocument doc = PdfDocument.open(imgPdf.save())) {
            assertThat(doc.pageCount()).isGreaterThan(0);
        }
    }

    @Test
    void fromImagesRejectsEmptyList() {
        assertThatThrownBy(() -> Pdf.fromImages(java.util.List.of())).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void fromImagesRejectsInvalidImage() {
        // Random bytes — not a PNG, JPEG, etc.
        byte[] junk = new byte[] {1, 2, 3, 4, 5, 6, 7, 8};
        assertThatThrownBy(() -> Pdf.fromImages(java.util.List.of(junk)))
                .isInstanceOf(fyi.oxide.pdf.exception.PdfException.class);
    }

    @Test
    void closeIsIdempotent() {
        Pdf pdf = Pdf.fromMarkdown("# X\n");
        pdf.close();
        pdf.close(); // no-op
        pdf.close(); // no-op
    }
}
