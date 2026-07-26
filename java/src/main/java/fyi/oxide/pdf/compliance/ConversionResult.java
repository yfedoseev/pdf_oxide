/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.compliance;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Result of a {@link fyi.oxide.pdf.PdfAConverter#convert(byte[], PdfALevel)} run.
 *
 * <p>{@link #success()} reflects whether the converted document met the
 * requested {@link #level()}; {@link #convertedPdf()} holds the converted
 * bytes regardless (best-effort output is still returned on partial
 * success). {@link #actions()} is the audit trail of remediations applied;
 * {@link #errors()} lists violations the converter could not fix
 * automatically.
 */
public final class ConversionResult {
    private final boolean success;
    private final PdfALevel level;
    private final byte[] convertedPdf;
    private final List<ConversionAction> actions;
    private final List<ConversionError> errors;

    public ConversionResult(
            boolean success,
            PdfALevel level,
            byte[] convertedPdf,
            List<ConversionAction> actions,
            List<ConversionError> errors) {
        this.success = success;
        this.level = Objects.requireNonNull(level, "level");
        this.convertedPdf = Objects.requireNonNull(convertedPdf, "convertedPdf").clone();
        this.actions =
                Collections.unmodifiableList(new java.util.ArrayList<>(Objects.requireNonNull(actions, "actions")));
        this.errors = Collections.unmodifiableList(new java.util.ArrayList<>(Objects.requireNonNull(errors, "errors")));
    }

    public boolean success() {
        return success;
    }

    public PdfALevel level() {
        return level;
    }

    /** @return the converted PDF bytes (a defensive copy). */
    public byte[] convertedPdf() {
        return convertedPdf.clone();
    }

    public List<ConversionAction> actions() {
        return actions;
    }

    public List<ConversionError> errors() {
        return errors;
    }

    @Override
    public String toString() {
        return "ConversionResult[success=" + success
                + " level=" + level
                + " actions=" + actions.size()
                + " errors=" + errors.size()
                + "]";
    }
}
