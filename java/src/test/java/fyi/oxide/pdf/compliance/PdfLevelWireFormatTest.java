/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.compliance;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Locks in the PDF/A and PDF/UA wire-format integer mapping against
 * the cdylib's documented C ABI ({@code src/ffi.rs:1225} +
 * {@code src/ffi.rs:5538}). Every pdf_oxide binding (Java, Ruby, PHP,
 * C#, Go) must send the SAME integer for the SAME PDF/A level — any
 * future re-numbering surfaces here as a hard test failure rather
 * than as a silently-wrong validation verdict.
 *
 * <p>Pre-v0.3.55 Java had alphabetical-natural ordinal order
 * ({@code A_1A.ordinal() == 0, A_1B.ordinal() == 1, …}) — semantically
 * reversed against what the cdylib actually does ({@code 0 = A1b,
 * 1 = A1a, …}). Users calling
 * {@code PdfValidator.isPdfA(doc, A_1A)} got A1b validation back.
 */
class PdfLevelWireFormatTest {

    @Test
    void pdfALevelOrdinalsMatchCdylibABI() {
        // B before A within each level (1, 2, 3) — matches src/ffi.rs:1225.
        assertThat(PdfALevel.A_1B.ordinal()).isEqualTo(0);
        assertThat(PdfALevel.A_1A.ordinal()).isEqualTo(1);
        assertThat(PdfALevel.A_2B.ordinal()).isEqualTo(2);
        assertThat(PdfALevel.A_2A.ordinal()).isEqualTo(3);
        assertThat(PdfALevel.A_2U.ordinal()).isEqualTo(4);
        assertThat(PdfALevel.A_3B.ordinal()).isEqualTo(5);
        assertThat(PdfALevel.A_3A.ordinal()).isEqualTo(6);
        assertThat(PdfALevel.A_3U.ordinal()).isEqualTo(7);
    }

    @Test
    void pdfUaLevelCodesMatchCdylibABI() {
        // 1-indexed — cdylib treats `level == 2` as UA-2, else UA-1.
        // Java uses code() (not ordinal()) because the wire format is
        // 1-indexed; mirrors C#'s `Ua1 = 1, Ua2 = 2`.
        assertThat(PdfUaLevel.UA_1.code()).isEqualTo(1);
        assertThat(PdfUaLevel.UA_2.code()).isEqualTo(2);
    }
}
