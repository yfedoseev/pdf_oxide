/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import fyi.oxide.pdf.exception.PdfException;
import fyi.oxide.pdf.policy.PolicyMode;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;

/**
 * Tests the global crypto-governance policy (v0.3.50 #230). pdf_oxide
 * is **set-once**: a single {@link PdfPolicy#set(PolicyMode)} call at
 * process startup, before any other crypto operation, is permitted.
 * Subsequent {@code set} calls throw. The default lazy initialisation
 * (any first {@link PdfPolicy#current()} or other crypto access) seeds
 * the policy to {@link PolicyMode#COMPAT}.
 *
 * <p>Surefire is configured with {@code reuseForks=false}, so each
 * test class gets a fresh JVM. We use {@code @Order} within this
 * class to make sure the {@code set()} attempt runs BEFORE any
 * {@code current()} read that would lazily lock the policy.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class PdfPolicyTest {

    /**
     * Run FIRST in this JVM fork: this is the only safe place to
     * call {@code set()} before another test's {@code current()}
     * lazily initialises the policy to COMPAT.
     */
    @Test
    @Order(1)
    void setSwitchesToStrictAtProcessStart() {
        PdfPolicy.set(PolicyMode.STRICT);
        assertThat(PdfPolicy.current()).isEqualTo(PolicyMode.STRICT);
    }

    @Test
    @Order(2)
    void secondSetThrowsAlreadySet() {
        // The previous test set the policy to STRICT. Any further
        // set() call should fail with the set-once error.
        assertThatThrownBy(() -> PdfPolicy.set(PolicyMode.COMPAT))
                .isInstanceOf(PdfException.class)
                .hasMessageContaining("already set");
    }

    @Test
    @Order(3)
    void presetAccessorsReturnTheRightMode() {
        // Read-only — independent of process state.
        assertThat(PdfPolicy.compat()).isEqualTo(PolicyMode.COMPAT);
        assertThat(PdfPolicy.strict()).isEqualTo(PolicyMode.STRICT);
        assertThat(PdfPolicy.fipsStrict()).isEqualTo(PolicyMode.FIPS_STRICT);
    }
}
