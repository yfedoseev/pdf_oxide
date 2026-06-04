/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import fyi.oxide.pdf.exception.PdfEncryptedException;
import fyi.oxide.pdf.exception.PdfInvalidStateException;
import fyi.oxide.pdf.exception.PdfIoException;
import fyi.oxide.pdf.exception.PdfParseException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Smoke tests for {@link PdfDocument} that validate the native side
 * end-to-end. Requires {@code -Dfyi.oxide.pdf.lib.path=…} pointing at
 * a pre-built {@code libpdf_oxide_jni.so} (the Maven {@code dev}
 * profile produces it via {@code questdb/rust-maven-plugin}; the
 * default Surefire config in {@code pom.xml} points at
 * {@code ../target/release/libpdf_oxide_jni.so}).
 *
 * <p>Fixtures are pdf_oxide's existing {@code tests/fixtures/} from
 * the workspace root; we resolve them relative to the project basedir.
 */
class PdfDocumentTest {

    private static Path fixturesDir;

    @BeforeAll
    static void resolveFixtures() {
        // java/src/test/java/... → java/ → ../tests/fixtures/
        fixturesDir = Paths.get("..")
                .resolve("tests")
                .resolve("fixtures")
                .toAbsolutePath()
                .normalize();
        // Skip the entire class if the fixture path doesn't exist — useful
        // when the tests run from a non-workspace context (Maven Central
        // standalone consumer). Won't happen in our CI.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                Files.isDirectory(fixturesDir),
                "fixtures dir not present (skipping native-bound tests): " + fixturesDir);
    }

    @Test
    void openAndCloseSimplePdf() {
        Path simple = fixturesDir.resolve("simple.pdf");
        try (PdfDocument doc = PdfDocument.open(simple)) {
            assertThat(doc.isOpen()).isTrue();
            assertThat(doc.pageCount()).isGreaterThan(0);
        }
    }

    @Test
    void closeIsIdempotent() {
        Path simple = fixturesDir.resolve("simple.pdf");
        PdfDocument doc = PdfDocument.open(simple);
        try {
            assertThat(doc.isOpen()).isTrue();
            doc.close();
            assertThat(doc.isOpen()).isFalse();
            // Second + third close: no exception, no JVM crash.
            doc.close();
            doc.close();
        } finally {
            // safety net even if asserts above throw
            doc.close();
        }
    }

    @Test
    void operationsOnClosedHandleThrowInvalidState() {
        Path simple = fixturesDir.resolve("simple.pdf");
        PdfDocument doc = PdfDocument.open(simple);
        doc.close();
        assertThatThrownBy(doc::pageCount)
                .isInstanceOf(PdfInvalidStateException.class)
                .hasMessageContaining("closed");
        assertThatThrownBy(() -> doc.extractText(0)).isInstanceOf(PdfInvalidStateException.class);
    }

    @Test
    void extractTextOnHelloStructureReturnsContent() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            String text = doc.extractText(0);
            assertThat(text).isNotEmpty();
            assertThat(text).containsIgnoringCase("hello");
        }
    }

    @Test
    @org.junit.jupiter.api.Tag("legacy-crypto")
    void encryptedPdfExtractsEmptyTextGracefully() {
        Path enc = fixturesDir.resolve("encrypted_needs_password.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(enc), "encrypted fixture not present");
        try (PdfDocument doc = PdfDocument.open(enc)) {
            // open succeeded (it just parsed metadata). As of v0.3.60, content
            // extraction on a PDF that cannot be decrypted with the empty
            // password degrades gracefully to empty text (matching
            // pdftotext / PyMuPDF) rather than throwing.
            assertThat(doc.extractText(0)).isEmpty();
        }
    }

    @Test
    void nonexistentFileThrowsIoException() {
        Path missing = fixturesDir.resolve("__does_not_exist__.pdf");
        assertThatThrownBy(() -> PdfDocument.open(missing)).isInstanceOf(PdfIoException.class);
    }

    @Test
    @org.junit.jupiter.api.Tag("legacy-crypto")
    void authenticateWithWrongPasswordReturnsFalse() {
        Path enc = fixturesDir.resolve("encrypted_needs_password.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(enc), "encrypted fixture not present");
        try (PdfDocument doc = PdfDocument.open(enc)) {
            assertThat(doc.authenticate("totally-wrong-password")).isFalse();
        }
    }

    @Test
    @org.junit.jupiter.api.Tag("legacy-crypto")
    void authenticateWithEmptyPasswordOnNonPasswordedEncryptionReturnsTrue() {
        // encrypted_cid_truetype.pdf is encrypted but with an empty user
        // password — authenticate("") should still return true (the
        // PdfDocument may have already auto-authenticated on open()).
        Path enc = fixturesDir.resolve("encrypted_cid_truetype.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(enc), "encrypted_cid_truetype.pdf not present");
        try (PdfDocument doc = PdfDocument.open(enc)) {
            assertThat(doc.authenticate("")).isTrue();
        }
    }

    @Test
    void authenticateOnUnencryptedDocReturnsTrue() {
        Path simple = fixturesDir.resolve("simple.pdf");
        try (PdfDocument doc = PdfDocument.open(simple)) {
            // Unencrypted PDFs return true regardless of the password.
            assertThat(doc.authenticate("anything")).isTrue();
            assertThat(doc.authenticate(new byte[0])).isTrue();
        }
    }

    @Test
    @org.junit.jupiter.api.Tag("legacy-crypto")
    void openWithWrongPasswordThrowsEncrypted() {
        Path enc = fixturesDir.resolve("encrypted_needs_password.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(enc), "encrypted fixture not present");
        assertThatThrownBy(() -> PdfDocument.open(enc, "wrong"))
                .isInstanceOf(PdfEncryptedException.class)
                .hasMessageContaining("wrong password");
    }

    @Test
    @org.junit.jupiter.api.Tag("legacy-crypto")
    void openWithEmptyPasswordOnNonPasswordedEncryptionWorks() {
        Path enc = fixturesDir.resolve("encrypted_cid_truetype.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(enc), "encrypted_cid_truetype.pdf not present");
        try (PdfDocument doc = PdfDocument.open(enc, "")) {
            assertThat(doc.pageCount()).isGreaterThan(0);
        }
    }

    @Test
    void autoExtractorExtractPageTypedReturnsAutoResult() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            AutoExtractor extractor = AutoExtractor.of(doc);
            fyi.oxide.pdf.auto.AutoResult r = extractor.extractPage(0);
            assertThat(r).isNotNull();
            assertThat(r.text()).isNotEmpty();
            assertThat(r.text()).containsIgnoringCase("hello");
            assertThat(r.confidence()).isBetween(0.0, 1.0);
            assertThat(r.regions()).isNotNull();
        }
    }

    @Test
    void autoExtractorExtractDocumentTypedReturnsAutoResult() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            AutoExtractor extractor = AutoExtractor.of(doc);
            fyi.oxide.pdf.auto.AutoResult r = extractor.extractDocument();
            assertThat(r).isNotNull();
            assertThat(r.text()).isNotEmpty();
            assertThat(r.pagesNeedingOcr()).isNotNull();
        }
    }

    @Test
    void autoExtractorExtractPageJsonContainsRichShape() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            AutoExtractor extractor = AutoExtractor.of(doc);
            String json = extractor.extractPageJson(0);
            assertThat(json).isNotEmpty();
            assertThat(json).startsWith("{").endsWith("}");
            assertThat(json).contains("\"page\"");
            assertThat(json).contains("\"text\"");
            assertThat(json).contains("\"regions\"");
            assertThat(json).contains("\"confidence\"");
            assertThat(json).contains("\"reason\"");
        }
    }

    @Test
    void autoExtractorExtractDocumentJsonAlsoWorks() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            AutoExtractor extractor = AutoExtractor.of(doc);
            String json = extractor.extractDocumentJson();
            assertThat(json).isNotEmpty().startsWith("{").endsWith("}");
        }
    }

    @Test
    void autoExtractorExtractAutoPageReturnsResult() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            AutoExtractor extractor = AutoExtractor.of(doc);
            fyi.oxide.pdf.auto.AutoResult r = extractor.extractAutoPage(0);
            assertThat(r).isNotNull();
            assertThat(r.text()).isNotEmpty();
            assertThat(r.text()).containsIgnoringCase("hello");
            assertThat(r.reason()).isEqualTo(fyi.oxide.pdf.auto.ExtractReason.OK);
            assertThat(r.regions()).isEmpty(); // simplified surface
        }
    }

    @Test
    void autoExtractorExtractTextConcatenatesPages() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            AutoExtractor extractor = AutoExtractor.of(doc);
            String all = extractor.extractText();
            assertThat(all).isNotEmpty();
            assertThat(all).containsIgnoringCase("hello");
            // Per-page split also works
            assertThat(extractor.extractTextForPage(0)).isNotEmpty();
        }
    }

    @Test
    void autoExtractorClassifyDocumentReturnsList() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            AutoExtractor extractor = AutoExtractor.of(doc);
            java.util.List<fyi.oxide.pdf.auto.PageClass> kinds = extractor.classifyDocumentKinds();
            assertThat(kinds).isNotNull();
            assertThat(kinds).hasSize(doc.pageCount());
        }
    }

    @Test
    void autoExtractorClassifyPageReturnsKind() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            AutoExtractor extractor = AutoExtractor.of(doc);
            fyi.oxide.pdf.auto.PageClass cls = extractor.classifyPageKind(0);
            // hello_structure.pdf has native text → TEXT_LAYER expected.
            assertThat(cls).isIn(fyi.oxide.pdf.auto.PageClass.TEXT_LAYER, fyi.oxide.pdf.auto.PageClass.MIXED);
        }
    }

    @Test
    void extractTextAutoOnNativeTextDocReturnsContent() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            String text = doc.extractTextAuto(0);
            // For a born-digital PDF, extractTextAuto should match
            // extractText since no OCR is needed.
            assertThat(text).isNotEmpty();
            assertThat(text).containsIgnoringCase("hello");
        }
    }

    @Test
    void extractTextAutoGracefulFallbackWhenOcrUnavailable() {
        // The .so under test is built WITHOUT the `ocr` Cargo feature.
        // On a scanned-image PDF, extractTextAuto must gracefully fall
        // back to the native text-layer (empty string here), NOT throw
        // PdfOcrUnavailableException. This is the v0.3.51
        // feedback_extraction_graceful_fallback contract.
        Path simple = fixturesDir.resolve("simple.pdf");
        try (PdfDocument doc = PdfDocument.open(simple)) {
            // No exception — just an empty string for a no-text PDF.
            String text = doc.extractTextAuto(0);
            assertThat(text).isNotNull();
        }
    }

    @Test
    void searchFindsLiteralText() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            java.util.List<fyi.oxide.pdf.search.SearchMatch> matches = doc.search("Hello");
            assertThat(matches).isNotNull().isNotEmpty();
            assertThat(matches.get(0).text()).containsIgnoringCase("hello");
            assertThat(matches.get(0).pageIndex()).isGreaterThanOrEqualTo(0);
            assertThat(matches.get(0).bbox()).isNotNull();
        }
    }

    @Test
    void searchCaseInsensitive() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            java.util.List<fyi.oxide.pdf.search.SearchMatch> ci = doc.search("hello", true, false, 0);
            assertThat(ci).isNotEmpty();
        }
    }

    @Test
    void searchNonexistentReturnsEmpty() {
        Path hello = fixturesDir.resolve("hello_structure.pdf");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(hello), "hello_structure.pdf not present");
        try (PdfDocument doc = PdfDocument.open(hello)) {
            assertThat(doc.search("xyzzyq42notthere")).isEmpty();
        }
    }

    @Test
    void formFieldsReturnsNonNullList() {
        Path simple = fixturesDir.resolve("simple.pdf");
        try (PdfDocument doc = PdfDocument.open(simple)) {
            java.util.List<fyi.oxide.pdf.form.FormField> fields = doc.formFields();
            // simple.pdf has no AcroForm — list should be empty but
            // non-null. Contract: no exception, no crash.
            assertThat(fields).isNotNull();
        }
    }

    @Test
    void producerAndCreatorAreOptional() {
        Path simple = fixturesDir.resolve("simple.pdf");
        try (PdfDocument doc = PdfDocument.open(simple)) {
            // Both must return an Optional (may be empty or populated);
            // the contract is "no exception, no crash".
            assertThat(doc.producer()).isNotNull();
            assertThat(doc.creator()).isNotNull();
        }
    }

    @Test
    void malformedFileThrowsPdfParseException() throws Exception {
        // Construct a tiny non-PDF file in /tmp; pdf_oxide should
        // reject it with Error::InvalidHeader → PdfParseException.
        Path tmp = Files.createTempFile("pdf-oxide-jni-test-", ".pdf");
        Files.write(tmp, new byte[] {'N', 'O', 'T', 'A', 'P', 'D', 'F', '\n'});
        try {
            assertThatThrownBy(() -> PdfDocument.open(tmp)).isInstanceOf(PdfParseException.class);
        } finally {
            Files.deleteIfExists(tmp);
        }
    }
}
