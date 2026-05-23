/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.redaction;

/**
 * Result of {@link fyi.oxide.pdf.DocumentEditor#applyRedactionsDestructive()}.
 *
 * <p>Carries the count of regions actually redacted (may be &lt; the
 * staged count if some couldn't be applied), and a flag indicating
 * whether the destructive [BLOCK] oracle from v0.3.50
 * {@code feature-231-destructive-redaction.md} §6.3 was satisfied.
 */
public final class RedactResult {
    private final int regionsApplied;
    private final boolean oracleVerified;

    public RedactResult(int regionsApplied, boolean oracleVerified) {
        this.regionsApplied = regionsApplied;
        this.oracleVerified = oracleVerified;
    }

    public int regionsApplied() {
        return regionsApplied;
    }
    /**
     * @return true if the extract-and-assert-absent oracle passed
     *         (extracted text AND raw saved bytes contain none of the
     *         redacted content; idempotent under re-application).
     */
    public boolean oracleVerified() {
        return oracleVerified;
    }

    @Override
    public String toString() {
        return "RedactResult[regionsApplied=" + regionsApplied + " oracleVerified=" + oracleVerified + "]";
    }
}
