/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.compliance;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Result of a {@link fyi.oxide.pdf.PdfValidator} run.
 *
 * <p>{@link #valid()} is the verdict — true iff there are zero
 * violations at the requested level. {@link #violations()} surfaces
 * the violation list (empty if {@link #valid()}).
 */
public final class ValidationResult {
    private final boolean valid;
    private final List<ValidationViolation> violations;

    public ValidationResult(boolean valid, List<ValidationViolation> violations) {
        this.valid = valid;
        this.violations = Collections.unmodifiableList(
                new java.util.ArrayList<>(Objects.requireNonNull(violations, "violations")));
    }

    public boolean valid() {
        return valid;
    }

    public List<ValidationViolation> violations() {
        return violations;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ValidationResult)) return false;
        ValidationResult r = (ValidationResult) o;
        return valid == r.valid && violations.equals(r.violations);
    }

    @Override
    public int hashCode() {
        return Objects.hash(valid, violations);
    }

    @Override
    public String toString() {
        return "ValidationResult[valid=" + valid + " violations=" + violations.size() + "]";
    }
}
