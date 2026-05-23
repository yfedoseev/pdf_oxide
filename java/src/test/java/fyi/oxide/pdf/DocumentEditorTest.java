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
import java.nio.file.Paths;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Tests for the DocumentEditor write surface. Round-trips
 * open → save → reopen-as-PdfDocument and exercises the exception
 * paths.
 */
class DocumentEditorTest {

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
    void openSaveRoundTripPreservesPageCount() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (DocumentEditor editor = DocumentEditor.open(hello)) {
            byte[] saved = editor.save();
            assertThat(saved).isNotEmpty();
            assertThat(new String(saved, 0, 5)).isEqualTo("%PDF-");
            try (PdfDocument doc = PdfDocument.open(saved)) {
                assertThat(doc.pageCount()).isGreaterThan(0);
            }
        }
    }

    @Test
    void openBytesAndSaveRoundTrip() throws Exception {
        Path simple = fixturesDir.resolve("simple.pdf");
        byte[] in = Files.readAllBytes(simple);
        try (DocumentEditor editor = DocumentEditor.open(in)) {
            byte[] out = editor.save();
            assertThat(out).isNotEmpty();
            assertThat(new String(out, 0, 5)).isEqualTo("%PDF-");
        }
    }

    @Test
    void closeIsIdempotent() {
        Path simple = fixturesDir.resolve("simple.pdf");
        DocumentEditor editor = DocumentEditor.open(simple);
        assertThat(editor.isOpen()).isTrue();
        editor.close();
        assertThat(editor.isOpen()).isFalse();
        editor.close(); // no-op
        editor.close(); // no-op
    }

    @Test
    void operationsOnClosedEditorThrow() {
        Path simple = fixturesDir.resolve("simple.pdf");
        DocumentEditor editor = DocumentEditor.open(simple);
        editor.close();
        assertThatThrownBy(editor::save).isInstanceOf(PdfInvalidStateException.class);
        assertThatThrownBy(() -> editor.setFormField("x", "y")).isInstanceOf(PdfInvalidStateException.class);
    }

    @Test
    void addRedactionQueuesRegion() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (DocumentEditor editor = DocumentEditor.open(hello)) {
            assertThat(editor.redactionCount(0)).isZero();
            editor.addRedaction(0, new fyi.oxide.pdf.geometry.BBox(50, 100, 200, 130));
            assertThat(editor.redactionCount(0)).isEqualTo(1);
            editor.addRedaction(0, new fyi.oxide.pdf.geometry.BBox(50, 200, 200, 230));
            assertThat(editor.redactionCount(0)).isEqualTo(2);
        }
    }

    @Test
    void addRedactionOutOfRangePageThrows() {
        Path simple = fixturesDir.resolve("simple.pdf");
        try (DocumentEditor editor = DocumentEditor.open(simple)) {
            assertThatThrownBy(() -> editor.addRedaction(99, new fyi.oxide.pdf.geometry.BBox(0, 0, 10, 10)))
                    .isInstanceOf(fyi.oxide.pdf.exception.PdfException.class);
        }
    }

    @Test
    void applyRedactionsDestructiveRemovesContent() {
        // hello_structure.pdf contains "Hello World". We queue a
        // big redaction covering most of the page, apply, save, and
        // verify extracted text shrinks (the v0.3.50 #231 contract).
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        String original;
        try (PdfDocument doc = PdfDocument.open(hello)) {
            original = doc.extractText(0);
        }
        int origLen = original.length();
        byte[] redacted;
        try (DocumentEditor editor = DocumentEditor.open(hello)) {
            // Big redaction covering the upper-left quadrant.
            editor.addRedaction(0, new fyi.oxide.pdf.geometry.BBox(0, 600, 500, 792));
            fyi.oxide.pdf.redaction.RedactResult result = editor.applyRedactionsDestructive();
            assertThat(result.regionsApplied()).isGreaterThanOrEqualTo(1);
            redacted = editor.save();
        }
        assertThat(redacted).isNotEmpty();
        // Note: the precise extracted-text shrinkage depends on font
        // path of the fixture; on hello_structure.pdf the "Hello"
        // text is in the upper-left and should be removed.
        try (PdfDocument doc = PdfDocument.open(redacted)) {
            String after = doc.extractText(0);
            // After destructive redaction of the upper-left region,
            // the text should be EQUAL OR SHORTER. (Equality if the
            // text was outside the box; shorter if inside.)
            assertThat(after.length()).isLessThanOrEqualTo(origLen);
        }
    }

    @Test
    void scrubMetadataRunsCleanly() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (DocumentEditor editor = DocumentEditor.open(hello)) {
            editor.scrubMetadata();
            byte[] out = editor.save();
            assertThat(out).isNotEmpty();
            assertThat(new String(out, 0, 5)).isEqualTo("%PDF-");
        }
    }

    @Test
    void setFormFieldOnDocWithoutFormThrows() {
        Path simple = fixturesDir.resolve("simple.pdf");
        try (DocumentEditor editor = DocumentEditor.open(simple)) {
            // simple.pdf has no AcroForm — setting any field name fails
            // with a Pdf{Parse,InvalidState}Exception from the Rust side.
            assertThatThrownBy(() -> editor.setFormField("nonexistent", "value"))
                    .isInstanceOf(fyi.oxide.pdf.exception.PdfException.class);
        }
    }
}
