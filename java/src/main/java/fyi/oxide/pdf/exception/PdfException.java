/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.exception;

/**
 * Root of the pdf_oxide exception hierarchy.
 *
 * <p>Extends {@link RuntimeException} — pdf_oxide is unchecked.
 * Modern Java consensus (Effective Java Item 71): checked exceptions
 * are for recoverable conditions where the caller is expected to take
 * a corrective action right there. Most PDF failures are not — they
 * are "log + show user + skip", which {@code RuntimeException} serves
 * better. Spring-AI / LangChain4j adapters can integrate without
 * wrapping. See {@code docs/releases/plans/v0.3.53/00-common-foundation.md}
 * §5 for the full rationale.
 *
 * <p>Subclasses correspond 1:1 to the entries in {@link PdfErrorKind}.
 * Catch the subclass when the recovery path is type-specific; switch
 * on {@link #kind()} when generic dispatch is sufficient.
 */
public class PdfException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    private final PdfErrorKind kind;

    /**
     * Convenience constructor — defaults kind to
     * {@link PdfErrorKind#OTHER}. Used by the JNI shim's
     * {@code env.throw_new(...)} path, which can only invoke a
     * one-arg {@code (String)} constructor when throwing into
     * {@code PdfException} directly (not a subclass).
     */
    public PdfException(String message) {
        super(message);
        this.kind = PdfErrorKind.OTHER;
    }

    /**
     * Construct a {@code PdfException}.
     *
     * @param kind the canonical error category (never null).
     * @param message a human-readable description; may be null.
     */
    public PdfException(PdfErrorKind kind, String message) {
        super(message);
        this.kind = requireNonNull(kind);
    }

    /**
     * Construct a {@code PdfException} with a cause.
     *
     * @param kind the canonical error category (never null).
     * @param message a human-readable description; may be null.
     * @param cause the underlying cause; may be null.
     */
    public PdfException(PdfErrorKind kind, String message, Throwable cause) {
        super(message, cause);
        this.kind = requireNonNull(kind);
    }

    /**
     * @return the canonical error category for this exception.
     *         Useful for {@code switch}-on-enum dispatch when subclass
     *         instanceof checks would be too verbose.
     */
    public final PdfErrorKind kind() {
        return kind;
    }

    private static PdfErrorKind requireNonNull(PdfErrorKind k) {
        if (k == null) {
            throw new NullPointerException("PdfErrorKind must not be null");
        }
        return k;
    }
}
