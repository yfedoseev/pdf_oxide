/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.compliance;

import java.util.Objects;
import java.util.Optional;
import org.jspecify.annotations.Nullable;

/** A single remediation step taken by {@link fyi.oxide.pdf.PdfAConverter}. */
public final class ConversionAction {
    private final ActionType actionType;
    private final String description;
    private final @Nullable String fixedErrorCode;

    public ConversionAction(ActionType actionType, String description, @Nullable String fixedErrorCode) {
        this.actionType = Objects.requireNonNull(actionType, "actionType");
        this.description = Objects.requireNonNull(description, "description");
        this.fixedErrorCode = fixedErrorCode;
    }

    public ActionType actionType() {
        return actionType;
    }

    public String description() {
        return description;
    }

    /** @return the error code (e.g. {@code "XMP-002"}) this action fixed, if any. */
    public Optional<String> fixedErrorCode() {
        return Optional.ofNullable(fixedErrorCode);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ConversionAction)) return false;
        ConversionAction a = (ConversionAction) o;
        return actionType == a.actionType
                && description.equals(a.description)
                && Objects.equals(fixedErrorCode, a.fixedErrorCode);
    }

    @Override
    public int hashCode() {
        return Objects.hash(actionType, description, fixedErrorCode);
    }

    @Override
    public String toString() {
        return "ConversionAction[actionType=" + actionType + " description=" + description + "]";
    }
}
