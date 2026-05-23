/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.auto;

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import org.jspecify.annotations.Nullable;

/**
 * Configuration for {@link fyi.oxide.pdf.AutoExtractor}. Built via
 * {@link #builder()}; all fields are nullable so the underlying Rust
 * core can pick a sensible per-field default. The kreuzberg-style
 * one-mega-config-with-nullable-nested-records pattern (see
 * {@code docs/releases/plans/v0.3.53/competitive-analysis.md} §1.2).
 *
 * <p>Presets ({@code fast()} / {@code balanced()} / {@code highFidelity()})
 * are exposed on {@link fyi.oxide.pdf.AutoExtractor} directly, not
 * here — the config is the lower-level escape hatch.
 */
public final class AutoExtractConfig {

    /** Empty config — every knob defaulted server-side. */
    public static final AutoExtractConfig DEFAULT = builder().build();

    private final @Nullable ExtractMode mode;
    private final @Nullable List<Integer> forceOcrPages;
    private final @Nullable Double minOcrConfidence;
    private final @Nullable List<String> ocrLanguages;
    private final @Nullable List<String> passwords;
    private final @Nullable Double topMarginFraction;
    private final @Nullable Double bottomMarginFraction;
    private final @Nullable Boolean allowSingleColumnTables;
    private final @Nullable Boolean ocrInlineImages;
    private final @Nullable String cancelToken;

    private AutoExtractConfig(Builder b) {
        this.mode = b.mode;
        this.forceOcrPages = b.forceOcrPages == null
                ? null
                : Collections.unmodifiableList(new java.util.ArrayList<>(b.forceOcrPages));
        this.minOcrConfidence = b.minOcrConfidence;
        this.ocrLanguages =
                b.ocrLanguages == null ? null : Collections.unmodifiableList(new java.util.ArrayList<>(b.ocrLanguages));
        this.passwords =
                b.passwords == null ? null : Collections.unmodifiableList(new java.util.ArrayList<>(b.passwords));
        this.topMarginFraction = b.topMarginFraction;
        this.bottomMarginFraction = b.bottomMarginFraction;
        this.allowSingleColumnTables = b.allowSingleColumnTables;
        this.ocrInlineImages = b.ocrInlineImages;
        this.cancelToken = b.cancelToken;
    }

    public Optional<ExtractMode> mode() {
        return Optional.ofNullable(mode);
    }

    public Optional<List<Integer>> forceOcrPages() {
        return Optional.ofNullable(forceOcrPages);
    }

    public Optional<Double> minOcrConfidence() {
        return Optional.ofNullable(minOcrConfidence);
    }

    public Optional<List<String>> ocrLanguages() {
        return Optional.ofNullable(ocrLanguages);
    }

    public Optional<List<String>> passwords() {
        return Optional.ofNullable(passwords);
    }

    public Optional<Double> topMarginFraction() {
        return Optional.ofNullable(topMarginFraction);
    }

    public Optional<Double> bottomMarginFraction() {
        return Optional.ofNullable(bottomMarginFraction);
    }

    public Optional<Boolean> allowSingleColumnTables() {
        return Optional.ofNullable(allowSingleColumnTables);
    }

    public Optional<Boolean> ocrInlineImages() {
        return Optional.ofNullable(ocrInlineImages);
    }

    public Optional<String> cancelToken() {
        return Optional.ofNullable(cancelToken);
    }

    public static Builder builder() {
        return new Builder();
    }

    public Builder toBuilder() {
        Builder b = new Builder();
        b.mode = this.mode;
        b.forceOcrPages = this.forceOcrPages;
        b.minOcrConfidence = this.minOcrConfidence;
        b.ocrLanguages = this.ocrLanguages;
        b.passwords = this.passwords;
        b.topMarginFraction = this.topMarginFraction;
        b.bottomMarginFraction = this.bottomMarginFraction;
        b.allowSingleColumnTables = this.allowSingleColumnTables;
        b.ocrInlineImages = this.ocrInlineImages;
        b.cancelToken = this.cancelToken;
        return b;
    }

    /**
     * Builder with {@code with}-prefixed setters per the
     * kreuzberg / Jackson POJO-builder convention
     * ({@code @JsonPOJOBuilder(withPrefix = "with")}).
     */
    public static final class Builder {
        private @Nullable ExtractMode mode;
        private @Nullable List<Integer> forceOcrPages;
        private @Nullable Double minOcrConfidence;
        private @Nullable List<String> ocrLanguages;
        private @Nullable List<String> passwords;
        private @Nullable Double topMarginFraction;
        private @Nullable Double bottomMarginFraction;
        private @Nullable Boolean allowSingleColumnTables;
        private @Nullable Boolean ocrInlineImages;
        private @Nullable String cancelToken;

        public Builder withMode(@Nullable ExtractMode m) {
            this.mode = m;
            return this;
        }

        public Builder withForceOcrPages(@Nullable List<Integer> p) {
            this.forceOcrPages = (p == null) ? null : new java.util.ArrayList<>(p);
            return this;
        }

        public Builder withMinOcrConfidence(@Nullable Double c) {
            this.minOcrConfidence = c;
            return this;
        }

        public Builder withOcrLanguages(@Nullable List<String> l) {
            this.ocrLanguages = (l == null) ? null : new java.util.ArrayList<>(l);
            return this;
        }

        public Builder withOcrLanguages(String... l) {
            this.ocrLanguages = java.util.Arrays.asList(l);
            return this;
        }

        public Builder withPasswords(@Nullable List<String> p) {
            this.passwords = (p == null) ? null : new java.util.ArrayList<>(p);
            return this;
        }

        public Builder withPasswords(String... p) {
            this.passwords = java.util.Arrays.asList(p);
            return this;
        }

        public Builder withTopMarginFraction(@Nullable Double f) {
            this.topMarginFraction = f;
            return this;
        }

        public Builder withTopMarginFraction(double f) {
            this.topMarginFraction = f;
            return this;
        }

        public Builder withBottomMarginFraction(@Nullable Double f) {
            this.bottomMarginFraction = f;
            return this;
        }

        public Builder withBottomMarginFraction(double f) {
            this.bottomMarginFraction = f;
            return this;
        }

        public Builder withAllowSingleColumnTables(@Nullable Boolean b) {
            this.allowSingleColumnTables = b;
            return this;
        }

        public Builder withAllowSingleColumnTables(boolean b) {
            this.allowSingleColumnTables = b;
            return this;
        }

        public Builder withOcrInlineImages(@Nullable Boolean b) {
            this.ocrInlineImages = b;
            return this;
        }

        public Builder withOcrInlineImages(boolean b) {
            this.ocrInlineImages = b;
            return this;
        }

        public Builder withCancelToken(@Nullable String t) {
            this.cancelToken = t;
            return this;
        }

        public AutoExtractConfig build() {
            return new AutoExtractConfig(this);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof AutoExtractConfig)) return false;
        AutoExtractConfig c = (AutoExtractConfig) o;
        return mode == c.mode
                && Objects.equals(forceOcrPages, c.forceOcrPages)
                && Objects.equals(minOcrConfidence, c.minOcrConfidence)
                && Objects.equals(ocrLanguages, c.ocrLanguages)
                && Objects.equals(passwords, c.passwords)
                && Objects.equals(topMarginFraction, c.topMarginFraction)
                && Objects.equals(bottomMarginFraction, c.bottomMarginFraction)
                && Objects.equals(allowSingleColumnTables, c.allowSingleColumnTables)
                && Objects.equals(ocrInlineImages, c.ocrInlineImages)
                && Objects.equals(cancelToken, c.cancelToken);
    }

    @Override
    public int hashCode() {
        return Objects.hash(
                mode,
                forceOcrPages,
                minOcrConfidence,
                ocrLanguages,
                passwords,
                topMarginFraction,
                bottomMarginFraction,
                allowSingleColumnTables,
                ocrInlineImages,
                cancelToken);
    }

    @Override
    public String toString() {
        return "AutoExtractConfig[mode=" + mode + " cancelToken=" + cancelToken + "]";
    }
}
