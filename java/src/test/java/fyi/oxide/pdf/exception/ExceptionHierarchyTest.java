/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.exception;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

/**
 * Pure-Java tests for the exception taxonomy. Validates that every
 * subclass correctly carries its {@link PdfErrorKind} and that the
 * subclass hierarchy is catchable by base class. No native code
 * required — runs even without the .so.
 */
class ExceptionHierarchyTest {

    @Test
    void everySubclassPinsTheCorrectKind() {
        assertThat(new PdfParseException("p").kind()).isEqualTo(PdfErrorKind.PARSE);
        assertThat(new PdfEncryptedException("p").kind()).isEqualTo(PdfErrorKind.ENCRYPTED);
        assertThat(new PdfPermissionException("p").kind()).isEqualTo(PdfErrorKind.PERMISSION);
        assertThat(new PdfIoException("p").kind()).isEqualTo(PdfErrorKind.IO);
        assertThat(new PdfOcrUnavailableException("p").kind()).isEqualTo(PdfErrorKind.OCR_UNAVAILABLE);
        assertThat(new PdfSignatureException("p").kind()).isEqualTo(PdfErrorKind.SIGNATURE);
        assertThat(new PdfInvalidStateException("p").kind()).isEqualTo(PdfErrorKind.INVALID_STATE);
        assertThat(new PdfUnsupportedException("p").kind()).isEqualTo(PdfErrorKind.UNSUPPORTED);
    }

    @Test
    void allSubclassesAreCatchableAsPdfException() {
        for (PdfException e : new PdfException[] {
            new PdfParseException("a"),
            new PdfEncryptedException("a"),
            new PdfPermissionException("a"),
            new PdfIoException("a"),
            new PdfOcrUnavailableException("a"),
            new PdfSignatureException("a"),
            new PdfInvalidStateException("a"),
            new PdfUnsupportedException("a"),
        }) {
            assertThat(e).isInstanceOf(PdfException.class);
        }
    }

    @Test
    void allSubclassesAreUnchecked() {
        for (PdfException e : new PdfException[] {
            new PdfParseException("a"),
            new PdfEncryptedException("a"),
            new PdfPermissionException("a"),
            new PdfIoException("a"),
            new PdfOcrUnavailableException("a"),
            new PdfSignatureException("a"),
            new PdfInvalidStateException("a"),
            new PdfUnsupportedException("a"),
        }) {
            assertThat(e).isInstanceOf(RuntimeException.class);
        }
    }

    @Test
    void switchOnKindEnableDispatch() {
        PdfException e = new PdfEncryptedException("locked");
        String result;
        switch (e.kind()) {
            case ENCRYPTED:
                result = "ask for password";
                break;
            case PERMISSION:
                result = "show permission denied";
                break;
            case OCR_UNAVAILABLE:
                result = "install OCR models";
                break;
            default:
                result = "generic error";
        }
        assertThat(result).isEqualTo("ask for password");
    }

    @Test
    void causeChainPreserved() {
        Throwable cause = new RuntimeException("under");
        PdfException e = new PdfIoException("over", cause);
        assertThat(e.getCause()).isSameAs(cause);
        assertThat(e.kind()).isEqualTo(PdfErrorKind.IO);
    }

    @Test
    void nullKindRejected() {
        assertThatThrownBy(() -> new PdfException(null, "msg")).isInstanceOf(NullPointerException.class);
    }
}
