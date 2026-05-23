/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.compliance;

import java.util.Objects;
import java.util.Optional;
import org.jspecify.annotations.Nullable;

/**
 * A single compliance violation reported by a {@link ValidationResult}.
 *
 * <p>The {@link #ruleId()} is a stable string identifier matching
 * pdf_oxide's compliance rule registry; consumers can dispatch on it.
 * Human-readable {@link #description()} explains it for end-user
 * surfacing.
 */
public final class ValidationViolation {
    private final String ruleId;
    private final String description;
    private final @Nullable Integer pageIndex;

    public ValidationViolation(String ruleId, String description, @Nullable Integer pageIndex) {
        this.ruleId = Objects.requireNonNull(ruleId, "ruleId");
        this.description = Objects.requireNonNull(description, "description");
        this.pageIndex = pageIndex;
    }

    public String ruleId() {
        return ruleId;
    }

    public String description() {
        return description;
    }
    /** @return the 0-based page index this violation applies to, if any. */
    public Optional<Integer> pageIndex() {
        return Optional.ofNullable(pageIndex);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ValidationViolation)) return false;
        ValidationViolation v = (ValidationViolation) o;
        return ruleId.equals(v.ruleId) && description.equals(v.description) && Objects.equals(pageIndex, v.pageIndex);
    }

    @Override
    public int hashCode() {
        return Objects.hash(ruleId, description, pageIndex);
    }

    @Override
    public String toString() {
        return "ValidationViolation[ruleId=" + ruleId
                + (pageIndex == null ? "" : " page=" + pageIndex)
                + " desc=" + description + "]";
    }
}
