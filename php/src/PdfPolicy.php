<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide;

use PdfOxide\Exceptions\PdfException;
use PdfOxide\FFI\NativeLibrary;
use PdfOxide\FFI\StringMarshaller;

/**
 * Process-global crypto-governance policy (v0.3.50 #230).
 *
 * Mirrors `fyi.oxide.pdf.PdfPolicy` from the Java binding.
 *
 * Selects which cryptographic algorithms pdf_oxide accepts for reads
 * and writes. Composes with the build-time feature flags
 * (`legacy-crypto`, `fips`).
 *
 * SET-ONCE semantics. pdf_oxide installs the policy at most once per
 * process: call {@see set()} BEFORE any other pdf_oxide operation. A
 * second {@see set()} call — or one after a document has been opened
 * — throws {@see PdfException} with `"already set"` in the message.
 * This is deliberate: a runtime policy downgrade would be a security
 * attack vector.
 */
final class PdfPolicy
{
    // ─────────────── Policy modes (string specs) ───────────
    // The C FFI takes a string spec; these are the canonical tokens.

    public const COMPAT = 'compat';

    public const STRICT = 'strict';

    public const FIPS_STRICT = 'fips_strict';

    /** Static-only. */
    private function __construct() {}

    /**
     * @return string the process-current policy mode (one of
     *                {@see COMPAT}, {@see STRICT}, {@see FIPS_STRICT}).
     */
    public static function current(): string
    {
        $ffi = NativeLibrary::getInstance();
        $cStr = $ffi->pdf_oxide_crypto_policy();
        if ($cStr === null) {
            return self::COMPAT; // lazy default
        }
        $out = \FFI::string($cStr);
        // pdf_oxide_crypto_policy() returns a leaked C string; per
        // pdf_oxide.h, free_string() releases it.
        $ffi->free_string($cStr);
        return $out;
    }

    /**
     * Install the process-global crypto policy. Throws if a policy is
     * already installed (set-once semantics).
     *
     * @throws PdfException when a policy is already installed
     */
    public static function set(string $mode): void
    {
        $ffi = NativeLibrary::getInstance();
        $cSpec = StringMarshaller::toCString($mode);
        try {
            $rc = $ffi->pdf_oxide_crypto_set_policy($cSpec);
        } finally {
            unset($cSpec);
        }
        if ((int) $rc !== 0) {
            throw new PdfException(
                "pdf_oxide crypto policy already set (set-once); requested={$mode} rc={$rc}",
                'POLICY_ALREADY_SET'
            );
        }
    }

    /** Whether FIPS-validated crypto is available on this build. */
    public static function fipsAvailable(): bool
    {
        $ffi = NativeLibrary::getInstance();
        return ((int) $ffi->pdf_oxide_crypto_fips_available()) === 1;
    }

    /** @return string the active crypto provider name (e.g. `aws-lc-rs`). */
    public static function activeProvider(): string
    {
        $ffi = NativeLibrary::getInstance();
        $cStr = $ffi->pdf_oxide_crypto_active_provider();
        if ($cStr === null) {
            return '';
        }
        $out = \FFI::string($cStr);
        $ffi->free_string($cStr);
        return $out;
    }

    // ─────────────── preset accessors ──────────────────────

    public static function compat(): string
    {
        return self::COMPAT;
    }

    public static function strict(): string
    {
        return self::STRICT;
    }

    public static function fipsStrict(): string
    {
        return self::FIPS_STRICT;
    }
}
