/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.signature;

/**
 * PAdES (PDF Advanced Electronic Signatures) baseline levels per
 * ETSI EN 319 142-1. v0.3.53 ships through B-LT (long-term
 * validation) — B-LTA (with archival timestamp) is a follow-up
 * artifact for v0.3.54.
 */
public enum SignatureLevel {
    /** Basic — signed-attributes only (no timestamp, no revocation material). */
    B_B,
    /** Basic-T — adds a signature-time-stamp (TSA) unsigned attribute. */
    B_T,
    /** Basic-LT — adds DSS / VRI revocation material for long-term verifiability. */
    B_LT
}
