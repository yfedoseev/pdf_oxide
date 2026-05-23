/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.policy;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * A crypto-governance policy (v0.3.50 #230). Pairs a {@link PolicyMode}
 * with optional per-algorithm overrides (allow/deny lists).
 *
 * <p>Use {@link fyi.oxide.pdf.PdfPolicy#compat()},
 * {@link fyi.oxide.pdf.PdfPolicy#strict()}, or
 * {@link fyi.oxide.pdf.PdfPolicy#fipsStrict()} for the named presets.
 * Tunable build via {@link #builder()}.
 */
public final class SecurityPolicy {

    private final PolicyMode mode;
    private final List<String> additionalAllow;
    private final List<String> additionalDeny;

    private SecurityPolicy(Builder b) {
        this.mode = Objects.requireNonNull(b.mode, "mode");
        this.additionalAllow = Collections.unmodifiableList(new java.util.ArrayList<>(b.additionalAllow));
        this.additionalDeny = Collections.unmodifiableList(new java.util.ArrayList<>(b.additionalDeny));
    }

    public PolicyMode mode() {
        return mode;
    }
    /** @return algorithm IDs explicitly allowed on top of the base mode. */
    public List<String> additionalAllow() {
        return additionalAllow;
    }
    /** @return algorithm IDs explicitly denied on top of the base mode. */
    public List<String> additionalDeny() {
        return additionalDeny;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private PolicyMode mode = PolicyMode.COMPAT;
        private final List<String> additionalAllow = new java.util.ArrayList<>();
        private final List<String> additionalDeny = new java.util.ArrayList<>();

        public Builder withMode(PolicyMode m) {
            this.mode = m;
            return this;
        }

        public Builder allow(String algId) {
            this.additionalAllow.add(algId);
            return this;
        }

        public Builder deny(String algId) {
            this.additionalDeny.add(algId);
            return this;
        }

        public SecurityPolicy build() {
            return new SecurityPolicy(this);
        }
    }
}
