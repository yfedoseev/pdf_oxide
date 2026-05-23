/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link PdfSigner#classifyLevel(byte[])} — the read-only
 * PAdES classification path. The full sign/verify write path is a
 * follow-up (requires PKCS#12 key material + TSA HTTP plumbing).
 */
class PdfSignerTest {

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
    void classifyLevelOnUnsignedPdfThrowsIllegalState() throws Exception {
        // simple.pdf has no signatures; classification has no defined
        // answer, so the binding throws IllegalStateException rather
        // than silently returning B_B.
        byte[] bytes = Files.readAllBytes(fixturesDir.resolve("simple.pdf"));
        assertThatThrownBy(() -> PdfSigner.classifyLevel(bytes))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("no signatures");
    }
}
