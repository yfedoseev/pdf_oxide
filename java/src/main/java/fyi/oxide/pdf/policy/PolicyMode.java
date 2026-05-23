/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.policy;

/**
 * Crypto-governance policy modes per v0.3.50 #230. Selects which
 * algorithms the engine will use for reads vs writes.
 *
 * <ul>
 *   <li>{@link #COMPAT} — accept all legacy algorithms (RC4, MD5-KDF, …)
 *       for reads; default. Matches the pre-v0.3.50 behaviour for
 *       backward compatibility.</li>
 *   <li>{@link #STRICT} — reject legacy algorithms for both reads and
 *       writes. Use for new content / hardened environments.</li>
 *   <li>{@link #FIPS_STRICT} — FIPS 140-3 mode: only FIPS-approved
 *       algorithms. Requires building pdf_oxide with the {@code fips}
 *       feature (and NOT {@code legacy-crypto}).</li>
 * </ul>
 */
public enum PolicyMode {
    COMPAT,
    STRICT,
    FIPS_STRICT
}
