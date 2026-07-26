/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import fyi.oxide.pdf.compliance.ConversionResult;
import fyi.oxide.pdf.compliance.PdfALevel;
import fyi.oxide.pdf.exception.PdfUnsupportedException;
import fyi.oxide.pdf.internal.NativeLoader;
import java.util.Objects;

/**
 * Static bytes-in/bytes-out PDF/A converter, modeled on {@link PdfSigner}.
 *
 * <p>Wraps {@code pdf_oxide::compliance::convert_to_pdf_a}: embeds fonts,
 * adds XMP metadata and an output intent, removes prohibited features
 * (JavaScript, encryption), and flattens transparency where the target
 * level requires it. The result carries the converted bytes plus an
 * audit trail of what was changed ({@link ConversionResult#actions()})
 * and what could not be fixed automatically
 * ({@link ConversionResult#errors()}).
 *
 * <p>Only PDF/A-1/2/3 are supported; PDF/A-4 levels throw
 * {@link PdfUnsupportedException} without touching native code.
 */
public final class PdfAConverter {

    static {
        NativeLoader.ensureLoaded();
    }

    private PdfAConverter() {
        // Static-only.
    }

    /**
     * Convert a PDF to the requested PDF/A conformance level.
     *
     * @throws PdfUnsupportedException for PDF/A-4 levels (pdf_oxide ships
     *     PDF/A-1/2/3 only).
     */
    public static ConversionResult convert(byte[] pdf, PdfALevel level) {
        Objects.requireNonNull(pdf, "pdf");
        Objects.requireNonNull(level, "level");
        if (level.ordinal() > PdfALevel.A_3U.ordinal()) {
            throw new PdfUnsupportedException(
                    "PdfAConverter.convert: " + level + " is not supported by pdf_oxide (PDF/A-1/2/3 only)");
        }
        return nativeConvert(pdf, level.ordinal(), level);
    }

    private static native ConversionResult nativeConvert(byte[] pdf, int levelOrdinal, PdfALevel level);
}
