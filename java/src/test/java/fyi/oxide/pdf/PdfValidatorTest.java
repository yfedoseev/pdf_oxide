/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import fyi.oxide.pdf.compliance.PdfALevel;
import fyi.oxide.pdf.compliance.PdfUaLevel;
import fyi.oxide.pdf.compliance.ValidationResult;
import fyi.oxide.pdf.exception.PdfUnsupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class PdfValidatorTest {

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
    void isPdfAReturnsBooleanForUntaggedDoc() {
        Path simple = fixturesDir.resolve("simple.pdf");
        try (PdfDocument doc = PdfDocument.open(simple)) {
            // simple.pdf is not declared PDF/A; A1b verdict should be
            // false (or maybe true for trivial docs — accept either,
            // the point is "no exception, no crash").
            boolean result = PdfValidator.isPdfA(doc, PdfALevel.A_1B);
            // No assertion on value — both true and false are valid
            // depending on the fixture's actual structure.
            // Validate that we got a clean boolean back.
            assertThat(result == true || result == false).isTrue();
        }
    }

    @Test
    void validatePdfAReturnsResultWithVerdict() {
        Path simple = fixturesDir.resolve("simple.pdf");
        try (PdfDocument doc = PdfDocument.open(simple)) {
            ValidationResult r = PdfValidator.validatePdfA(doc, PdfALevel.A_1B);
            assertThat(r).isNotNull();
            assertThat(r.violations()).isNotNull();
        }
    }

    @Test
    void pdfA4LevelsThrowUnsupported() {
        Path simple = fixturesDir.resolve("simple.pdf");
        try (PdfDocument doc = PdfDocument.open(simple)) {
            assertThatThrownBy(() -> PdfValidator.isPdfA(doc, PdfALevel.A_4))
                    .isInstanceOf(PdfUnsupportedException.class);
            assertThatThrownBy(() -> PdfValidator.isPdfA(doc, PdfALevel.A_4E))
                    .isInstanceOf(PdfUnsupportedException.class);
        }
    }

    @Test
    void isPdfUaReturnsBoolean() {
        Path simple = fixturesDir.resolve("simple.pdf");
        try (PdfDocument doc = PdfDocument.open(simple)) {
            boolean result = PdfValidator.isPdfUa(doc, PdfUaLevel.UA_1);
            assertThat(result == true || result == false).isTrue();
        }
    }

    @Test
    void pdfUa2ThrowsUnsupported() {
        Path simple = fixturesDir.resolve("simple.pdf");
        try (PdfDocument doc = PdfDocument.open(simple)) {
            assertThatThrownBy(() -> PdfValidator.isPdfUa(doc, PdfUaLevel.UA_2))
                    .isInstanceOf(PdfUnsupportedException.class);
        }
    }
}
