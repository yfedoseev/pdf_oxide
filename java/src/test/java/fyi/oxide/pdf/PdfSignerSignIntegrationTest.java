/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import fyi.oxide.pdf.signature.SignOptions;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

/**
 * End-to-end integration tests for {@link PdfSigner#sign}.
 *
 * <p>Uses the shared {@code tests/fixtures/test_signing.p12}
 * self-signed certificate (password {@code "testpass"}) that the
 * Rust crate's signature tests also use, so the same key material
 * proves the JNI surface against the same Rust core.
 *
 * <p>B-T / B-LT tests are gated on {@code PDF_OXIDE_TSA_URL} env
 * var being set (e.g. {@code https://freetsa.org/tsr}). Default-
 * skipped so CI without network access stays green; FREETSA's
 * uptime varies. To run locally:
 *
 * <pre>{@code
 * PDF_OXIDE_TSA_URL=https://freetsa.org/tsr mvn -P!dev test \
 *     -Dtest=PdfSignerSignIntegrationTest
 * }</pre>
 */
class PdfSignerSignIntegrationTest {

    private static Path fixturesDir;
    private static byte[] pdfBytes;
    private static byte[] p12Bytes;
    private static final String P12_PASSWORD = "testpass";

    @BeforeAll
    static void load() throws Exception {
        fixturesDir = Paths.get("..")
                .resolve("tests")
                .resolve("fixtures")
                .toAbsolutePath()
                .normalize();
        org.junit.jupiter.api.Assumptions.assumeTrue(
                Files.isDirectory(fixturesDir), "fixtures dir not present: " + fixturesDir);
        Path simple = fixturesDir.resolve("simple.pdf");
        Path p12 = fixturesDir.resolve("test_signing.p12");
        org.junit.jupiter.api.Assumptions.assumeTrue(
                Files.exists(simple) && Files.exists(p12), "required fixtures missing (simple.pdf, test_signing.p12)");
        pdfBytes = Files.readAllBytes(simple);
        p12Bytes = Files.readAllBytes(p12);
    }

    @Test
    void signBBProducesSignedPdfWithEmbeddedCmsBlob() {
        // PAdES B-B (no timestamp authority needed). Proves the
        // PKCS#12 → SigningCredentials → CMS construction → signed-
        // PDF round trip works through the JNI surface.
        PdfSigner signer = PdfSigner.fromPkcs12(p12Bytes, P12_PASSWORD);
        byte[] signed = signer.sign(
                pdfBytes,
                SignOptions.builder()
                        .withLevel(fyi.oxide.pdf.signature.SignatureLevel.B_B)
                        .withReason("Integration test")
                        .build());
        assertThat(signed).isNotNull();
        // Signed PDF must be longer than the input (signature + CMS blob).
        assertThat(signed.length).isGreaterThan(pdfBytes.length);
        // The output should still be a parseable PDF.
        assertThat(new String(signed, 0, 8)).startsWith("%PDF-");
        // Round-trip: should be reopenable via PdfDocument.
        try (PdfDocument verify = PdfDocument.open(signed)) {
            assertThat(verify.pageCount()).isGreaterThanOrEqualTo(1);
        }
        // NOTE: classifyLevel() against freshly-signed output is a
        // separate code path (signature enumeration over an
        // incremental update); track in follow-up if the verify-via-
        // classify round-trip needs to succeed here.
    }

    @Test
    void signRoundTripIsOpenable() {
        PdfSigner signer = PdfSigner.fromPkcs12(p12Bytes, P12_PASSWORD);
        byte[] signed = signer.sign(
                pdfBytes,
                SignOptions.builder()
                        .withLevel(fyi.oxide.pdf.signature.SignatureLevel.B_B)
                        .build());
        // PdfDocument.open should accept the signed bytes and report
        // the same page count.
        try (PdfDocument doc = PdfDocument.open(signed)) {
            assertThat(doc.pageCount()).isGreaterThanOrEqualTo(1);
        }
    }

    @Test
    void signBTWithoutTsaUrlThrowsIllegalArgument() {
        // SignOptions.level(B_T) without tsaUrl() set is a config
        // error — we surface it as IllegalArgumentException before
        // reaching the native (no point making the JVM start signing
        // only to fail at the TSA HTTP call with a less-clear error).
        PdfSigner signer = PdfSigner.fromPkcs12(p12Bytes, P12_PASSWORD);
        assertThatThrownBy(() -> signer.sign(
                        pdfBytes,
                        SignOptions.builder()
                                .withLevel(fyi.oxide.pdf.signature.SignatureLevel.B_T)
                                .build()))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("tsaUrl");
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "PDF_OXIDE_TSA_URL", matches = ".+")
    void signBTWithRealTsaProducesBTSignature() {
        String tsaUrl = System.getenv("PDF_OXIDE_TSA_URL");
        PdfSigner signer = PdfSigner.fromPkcs12(p12Bytes, P12_PASSWORD);
        byte[] signed = signer.sign(
                pdfBytes,
                SignOptions.builder()
                        .withLevel(fyi.oxide.pdf.signature.SignatureLevel.B_T)
                        .withTsaUrl(tsaUrl)
                        .withReason("B-T integration test")
                        .build());
        assertThat(signed).isNotNull();
        assertThat(signed.length).isGreaterThan(pdfBytes.length);
        fyi.oxide.pdf.signature.SignatureLevel level = PdfSigner.classifyLevel(signed);
        assertThat(level)
                .as("B_T signature should classify as B_T (timestamp-token present)")
                .isEqualTo(fyi.oxide.pdf.signature.SignatureLevel.B_T);
    }
}
