/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import fyi.oxide.pdf.compliance.PdfALevel;
import fyi.oxide.pdf.compliance.PdfUaLevel;
import fyi.oxide.pdf.compliance.PdfXLevel;
import fyi.oxide.pdf.compliance.ValidationResult;
import fyi.oxide.pdf.internal.NativeLoader;
import java.util.Collections;
import java.util.Objects;

/**
 * Static façade for PDF/A · PDF/X · PDF/UA compliance validation
 * (v0.3.50).
 *
 * <p>v0.3.53 ships the **simplified boolean variants**
 * {@link #isPdfA(PdfDocument, PdfALevel)} and
 * {@link #isPdfUa(PdfDocument, PdfUaLevel)}; the full
 * {@link ValidationResult} (with violations list) wires in a
 * follow-up.
 *
 * <p><b>Thread safety:</b> {@code validate*} takes a {@code &mut
 * PdfDocument} on the Rust side, so do not invoke concurrently
 * against the same document.
 */
public final class PdfValidator {

    static {
        NativeLoader.ensureLoaded();
    }

    private PdfValidator() {
        // Static-only.
    }

    /**
     * Quick PDF/A compliance check.
     *
     * @return true if the document conforms to {@code level}.
     * @throws fyi.oxide.pdf.exception.PdfUnsupportedException for
     *         PDF/A-4 levels (pdf_oxide ships PDF/A-1/2/3 only in v0.3.53).
     */
    public static boolean isPdfA(PdfDocument doc, PdfALevel level) {
        Objects.requireNonNull(doc, "doc");
        Objects.requireNonNull(level, "level");
        return nativeIsPdfA(doc.requireHandleForCallers(), level.ordinal());
    }

    /** Quick PDF/UA compliance check. */
    public static boolean isPdfUa(PdfDocument doc, PdfUaLevel level) {
        Objects.requireNonNull(doc, "doc");
        Objects.requireNonNull(level, "level");
        return nativeIsPdfUa(doc.requireHandleForCallers(), level.ordinal());
    }

    /**
     * Returns a simplified {@link ValidationResult} with just the
     * verdict. Full violations list ships in a follow-up.
     */
    public static ValidationResult validatePdfA(PdfDocument doc, PdfALevel level) {
        return new ValidationResult(isPdfA(doc, level), Collections.emptyList());
    }

    /** PDF/X validation — Phase 4 T16 follow-up (pdf_oxide PDF/X validator not yet exposed). */
    public static ValidationResult validatePdfX(PdfDocument doc, PdfXLevel level) {
        Objects.requireNonNull(doc, "doc");
        Objects.requireNonNull(level, "level");
        throw new UnsupportedOperationException(
                "PdfValidator.validatePdfX: pdf_oxide does not yet expose a PDF/X public validator (Phase 4 T16 follow-up)");
    }

    /** Returns a simplified ValidationResult mirroring {@link #isPdfUa}. */
    public static ValidationResult validatePdfUa(PdfDocument doc, PdfUaLevel level) {
        return new ValidationResult(isPdfUa(doc, level), Collections.emptyList());
    }

    private static native boolean nativeIsPdfA(long handle, int levelOrdinal);

    private static native boolean nativeIsPdfUa(long handle, int levelOrdinal);
}
