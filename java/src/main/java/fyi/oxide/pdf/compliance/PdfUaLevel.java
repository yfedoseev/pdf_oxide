/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.compliance;

/**
 * PDF/UA (Universal Accessibility) levels per ISO 14289.
 *
 * <p>Wire format matches the cdylib's {@code pdf_validate_pdf_ua}
 * (src/ffi.rs:5538), which treats {@code level == 2} as PDF/UA-2 and
 * anything else as PDF/UA-1. Explicit codes here keep Java in
 * lock-step with the C# binding ({@code Ua1=1, Ua2=2}) and the
 * underlying wire format the cdylib expects; callers use
 * {@link #code()} when crossing the JNI boundary, not
 * {@link #ordinal()}.
 */
public enum PdfUaLevel {
    /** PDF/UA-1 (ISO 14289-1, 2014). */
    UA_1(1),
    /** PDF/UA-2 (ISO 14289-2, 2024). */
    UA_2(2);

    private final int code;

    PdfUaLevel(int code) {
        this.code = code;
    }

    /** Wire-format integer expected by the cdylib. */
    public int code() {
        return code;
    }
}
