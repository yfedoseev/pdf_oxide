/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.signature;

import java.util.Objects;
import java.util.Optional;
import org.jspecify.annotations.Nullable;

/**
 * Configuration for a PAdES signing operation. Builder-driven per
 * the kreuzberg-style {@code with}-prefix convention.
 */
public final class SignOptions {

    private final SignatureLevel level;
    private final @Nullable String reason;
    private final @Nullable String location;
    private final @Nullable String contactInfo;
    private final @Nullable String tsaUrl;

    private SignOptions(Builder b) {
        this.level = Objects.requireNonNull(b.level, "level");
        this.reason = b.reason;
        this.location = b.location;
        this.contactInfo = b.contactInfo;
        this.tsaUrl = b.tsaUrl;
    }

    public SignatureLevel level() {
        return level;
    }

    public Optional<String> reason() {
        return Optional.ofNullable(reason);
    }

    public Optional<String> location() {
        return Optional.ofNullable(location);
    }

    public Optional<String> contactInfo() {
        return Optional.ofNullable(contactInfo);
    }
    /** @return TSA endpoint URL; required for {@link SignatureLevel#B_T} and {@link SignatureLevel#B_LT}. */
    public Optional<String> tsaUrl() {
        return Optional.ofNullable(tsaUrl);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private SignatureLevel level = SignatureLevel.B_B;
        private @Nullable String reason;
        private @Nullable String location;
        private @Nullable String contactInfo;
        private @Nullable String tsaUrl;

        public Builder withLevel(SignatureLevel l) {
            this.level = l;
            return this;
        }

        public Builder withReason(@Nullable String r) {
            this.reason = r;
            return this;
        }

        public Builder withLocation(@Nullable String l) {
            this.location = l;
            return this;
        }

        public Builder withContactInfo(@Nullable String c) {
            this.contactInfo = c;
            return this;
        }

        public Builder withTsaUrl(@Nullable String u) {
            this.tsaUrl = u;
            return this;
        }

        public SignOptions build() {
            return new SignOptions(this);
        }
    }
}
