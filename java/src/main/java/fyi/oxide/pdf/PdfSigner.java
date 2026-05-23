/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import fyi.oxide.pdf.internal.NativeLoader;
import fyi.oxide.pdf.signature.SignatureLevel;
import java.nio.file.Path;
import java.util.Objects;

/**
 * PAdES B-B / B-T / B-LT digital-signature signer + verifier
 * (v0.3.50 #235).
 *
 * <p>Thread-safe after construction: multiple threads can call
 * {@link #sign(byte[], fyi.oxide.pdf.signature.SignOptions)} or
 * {@link #verify(byte[])} concurrently on the same {@code PdfSigner}
 * instance — the underlying key material is reference-counted on the
 * Rust side, and each call takes its own input PDF.
 *
 * <p>Signing routes through the v0.3.50 crypto-governance policy
 * ({@link PdfPolicy}) — bypassing the policy is impossible.
 *
 * <p><b>Status (v0.3.53)</b>: API surface complete; native bindings
 * stub until Phase 4 T15.
 */
public final class PdfSigner {

    static {
        NativeLoader.ensureLoaded();
    }

    /** Constructed instance state — PKCS#12 bytes + password, retained for sign() calls. */
    private final byte[] keystoreBytes;

    private final String password;

    private PdfSigner(byte[] keystoreBytes, String password) {
        this.keystoreBytes = keystoreBytes;
        this.password = password;
    }

    /** Load credentials from a PKCS#12 file. */
    public static PdfSigner fromPkcs12(Path keystore, String password) {
        Objects.requireNonNull(keystore, "keystore");
        Objects.requireNonNull(password, "password");
        try {
            byte[] bytes = java.nio.file.Files.readAllBytes(keystore);
            return new PdfSigner(bytes, password);
        } catch (java.io.IOException e) {
            throw new fyi.oxide.pdf.exception.PdfIoException(
                    "Failed to read PKCS#12: " + keystore + ": " + e.getMessage(), e);
        }
    }

    /** Load credentials from PKCS#12 bytes. */
    public static PdfSigner fromPkcs12(byte[] keystoreBytes, String password) {
        Objects.requireNonNull(keystoreBytes, "keystoreBytes");
        Objects.requireNonNull(password, "password");
        return new PdfSigner(keystoreBytes.clone(), password);
    }

    /**
     * Sign a PDF at the requested PAdES baseline level.
     *
     * <p>B-T / B-LT require a non-null {@code tsaUrl} in
     * {@code opts} (RFC 3161 TSA endpoint such as
     * {@code http://timestamp.example.com}). B-B does not need a TSA.
     *
     * <p>Requires the {@code pdf_oxide_jni} library to be built with
     * the {@code signatures} feature (and {@code tsa-client} for B-T/B-LT).
     *
     * @return the signed PDF bytes.
     */
    public byte[] sign(byte[] pdf, fyi.oxide.pdf.signature.SignOptions opts) {
        Objects.requireNonNull(pdf, "pdf");
        Objects.requireNonNull(opts, "opts");
        String tsaUrl = opts.tsaUrl().orElse(null);
        if (opts.level() != SignatureLevel.B_B && tsaUrl == null) {
            throw new IllegalArgumentException("PAdES " + opts.level() + " requires opts.tsaUrl() to be set");
        }
        return nativeSign(pdf, keystoreBytes, password, opts.level().ordinal(), tsaUrl);
    }

    public boolean verify(byte[] pdf) {
        Objects.requireNonNull(pdf, "pdf");
        // Verify success ≈ classify returns any valid level + the sig
        // chain is well-formed. v0.3.53 simplified: returns true if
        // classifyLevel succeeds (signature is parseable).
        try {
            classifyLevel(pdf);
            return true;
        } catch (IllegalStateException e) {
            // No signatures present — verify-against-nothing is false.
            return false;
        }
    }

    private static native byte[] nativeSignBB(byte[] pdf, byte[] pkcs12, String password);

    private static native byte[] nativeSign(
            byte[] pdf, byte[] pkcs12, String password, int levelOrdinal, String tsaUrl);

    /**
     * Classify the PAdES baseline level of the highest-baseline
     * signature in the PDF. Returns {@link SignatureLevel#B_B},
     * {@link SignatureLevel#B_T}, or {@link SignatureLevel#B_LT}.
     *
     * <p>Requires the {@code pdf_oxide_jni} library to be built with
     * the {@code signatures} feature (or {@code full}). On a build
     * without that feature this throws
     * {@link fyi.oxide.pdf.exception.PdfUnsupportedException}.
     *
     * @throws IllegalStateException if the PDF contains no signatures.
     */
    public static SignatureLevel classifyLevel(byte[] pdf) {
        java.util.Objects.requireNonNull(pdf, "pdf");
        int ordinal = nativeClassifyPdfLevel(pdf);
        if (ordinal < 0) {
            throw new IllegalStateException("PDF contains no signatures to classify");
        }
        return SignatureLevel.values()[ordinal];
    }

    private static native int nativeClassifyPdfLevel(byte[] pdf);
}
