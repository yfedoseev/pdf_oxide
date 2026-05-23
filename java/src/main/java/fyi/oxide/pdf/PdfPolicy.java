/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import fyi.oxide.pdf.internal.NativeLoader;
import fyi.oxide.pdf.policy.PolicyMode;

/**
 * Process-global crypto-governance policy (v0.3.50 #230).
 *
 * <p>Selects which cryptographic algorithms are accepted for reads
 * and writes. Composes with the build-time feature flags
 * ({@code legacy-crypto}, {@code fips}) — if a build is missing
 * {@code legacy-crypto} then {@link PolicyMode#COMPAT} can't enable
 * RC4/MD5-KDF (the algorithm isn't compiled in regardless of policy).
 *
 * <p><b>Set-once semantics.</b> pdf_oxide installs the policy at
 * most once per process: call {@link #set} <b>before</b> any other
 * pdf_oxide operation (including {@link #current}). A second
 * {@link #set} call — or one after any document has been opened
 * — throws {@link fyi.oxide.pdf.exception.PdfException} with a
 * message containing {@code "already set"}. This is deliberate: a
 * runtime policy downgrade would be a security attack vector.
 *
 * <p>If no explicit {@link #set} call is made, {@link #current} (or
 * any first crypto access) lazily installs {@link PolicyMode#COMPAT}.
 */
public final class PdfPolicy {

    static {
        NativeLoader.ensureLoaded();
    }

    private PdfPolicy() {
        // Static-only.
    }

    /** @return the process-current policy mode. */
    public static PolicyMode current() {
        return ORDINAL_TO_MODE[nativeCurrentOrdinal()];
    }

    /** Set the process-global policy mode. */
    public static void set(PolicyMode mode) {
        java.util.Objects.requireNonNull(mode, "mode");
        nativeSetByOrdinal(mode.ordinal());
    }

    /**
     * Lookup table indexed by the {@link PolicyMode} ordinal — must
     * stay in sync with the constants in
     * {@code pdf_oxide_jni/src/policy.rs}. Validated by a unit test
     * that checks the enum constant order.
     */
    private static final PolicyMode[] ORDINAL_TO_MODE = PolicyMode.values();

    private static native int nativeCurrentOrdinal();

    private static native void nativeSetByOrdinal(int ordinal);

    /** Preset: accept all algorithms (RC4, MD5-KDF). Default mode. */
    public static PolicyMode compat() {
        return PolicyMode.COMPAT;
    }
    /** Preset: reject legacy algorithms. */
    public static PolicyMode strict() {
        return PolicyMode.STRICT;
    }
    /** Preset: FIPS 140-3 only. Requires the {@code fips} build feature. */
    public static PolicyMode fipsStrict() {
        return PolicyMode.FIPS_STRICT;
    }
}
