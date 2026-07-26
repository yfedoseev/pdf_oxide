/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import fyi.oxide.pdf.compliance.ConversionResult;
import fyi.oxide.pdf.compliance.PdfALevel;
import fyi.oxide.pdf.exception.PdfUnsupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class PdfAConverterTest {

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

    private static Stream<PdfALevel> supportedLevels() {
        return Stream.of(PdfALevel.A_1B, PdfALevel.A_2B, PdfALevel.A_2U, PdfALevel.A_3B);
    }

    @ParameterizedTest
    @MethodSource("supportedLevels")
    void convertProducesNonEmptyBytesForSupportedLevels(PdfALevel level) throws Exception {
        byte[] source = Files.readAllBytes(fixturesDir.resolve("simple.pdf"));
        ConversionResult result = PdfAConverter.convert(source, level);
        assertThat(result).isNotNull();
        assertThat(result.level()).isEqualTo(level);
        assertThat(result.convertedPdf()).isNotEmpty();
    }

    @Test
    void convertRecordsXmpMetadataAction() throws Exception {
        byte[] source = Files.readAllBytes(fixturesDir.resolve("simple.pdf"));
        ConversionResult result = PdfAConverter.convert(source, PdfALevel.A_1B);
        // simple.pdf has no XMP packet, so conversion to any PDF/A level
        // must add one; the converted bytes carry a /Metadata stream
        // regardless of whether every other rule was satisfiable.
        String convertedText = new String(result.convertedPdf(), java.nio.charset.StandardCharsets.ISO_8859_1);
        assertThat(convertedText).contains("/Metadata");
    }

    @Test
    void convertPreservesPageCount() throws Exception {
        byte[] source = Files.readAllBytes(fixturesDir.resolve("simple.pdf"));
        int originalPageCount;
        try (PdfDocument doc = PdfDocument.open(source)) {
            originalPageCount = doc.pageCount();
        }
        ConversionResult result = PdfAConverter.convert(source, PdfALevel.A_2B);
        try (PdfDocument converted = PdfDocument.open(result.convertedPdf())) {
            assertThat(converted.pageCount()).isEqualTo(originalPageCount);
        }
    }

    @Test
    void convertRejectsPdfA4Levels() throws Exception {
        byte[] source = Files.readAllBytes(fixturesDir.resolve("simple.pdf"));
        assertThatThrownBy(() -> PdfAConverter.convert(source, PdfALevel.A_4))
                .isInstanceOf(PdfUnsupportedException.class);
        assertThatThrownBy(() -> PdfAConverter.convert(source, PdfALevel.A_4E))
                .isInstanceOf(PdfUnsupportedException.class);
        assertThatThrownBy(() -> PdfAConverter.convert(source, PdfALevel.A_4F))
                .isInstanceOf(PdfUnsupportedException.class);
    }
}
