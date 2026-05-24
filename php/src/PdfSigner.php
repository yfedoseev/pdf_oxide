<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide;

use FFI;
use FFI\CData;
use PdfOxide\Exceptions\IoException;
use PdfOxide\Exceptions\SignatureException;
use PdfOxide\Exceptions\ValidationException;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\FFI\NativeLibrary;

/**
 * PAdES B-B / B-T / B-LT / B-LTA digital-signature signer + verifier
 * (v0.3.50 #235 + v0.3.51 5-arg shim, full sign wiring in v0.3.55 #546).
 *
 * Mirrors `fyi.oxide.pdf.PdfSigner` from the Java binding and the
 * Ruby `PdfOxide::PdfSigner`. Signing routes through pdf_oxide's
 * crypto-governance policy ({@see PdfPolicy}) — bypassing the policy
 * is impossible.
 *
 * Loads credentials once via {@see fromPkcs12()}; the credentials
 * handle is reused across multiple {@see sign()} calls and freed in
 * {@see close()} / `__destruct()`.
 *
 * Per `feedback_extraction_graceful_fallback`: signing is a
 * **security operation** — every non-zero native return fails closed.
 */
final class PdfSigner
{
    // ─────────────── PAdES level ordinals ──────────────────
    // Frozen by the Rust ABI (`PadesLevel` in `src/signing/pades.rs`).

    public const LEVEL_B_B = 0;

    public const LEVEL_B_T = 1;

    public const LEVEL_B_LT = 2;

    public const LEVEL_B_LTA = 3;

    /**
     * PAdES level codes keyed by Ruby-style short tag. Mirrors
     * `PdfOxide::PdfSigner::LEVELS` (ruby/lib/pdf_oxide/pdf_signer.rb)
     * one-for-one.
     */
    private const LEVELS = [
        'b' => self::LEVEL_B_B,
        't' => self::LEVEL_B_T,
        'lt' => self::LEVEL_B_LT,
        'lta' => self::LEVEL_B_LTA,
    ];

    private ?CData $credentials = null;

    private readonly FunctionBindings $bindings;

    public function __construct(CData $credentials)
    {
        $this->bindings = new FunctionBindings();
        $this->credentials = $credentials;
    }

    /**
     * Load credentials from a PKCS#12 file (`.p12` / `.pfx`).
     *
     * @throws IoException when the keystore file is missing
     */
    public static function fromPkcs12(string $keystorePath, string $password): self
    {
        if (! is_file($keystorePath)) {
            throw new IoException("Keystore not found: {$keystorePath}");
        }
        $bytes = (string) file_get_contents($keystorePath);
        $bindings = new FunctionBindings();
        // Real cdylib symbol: pdf_certificate_load_from_bytes(PKCS#12 bytes, password).
        $cert = $bindings->pdfCertificateLoadFromBytes($bytes, $password);
        return new self($cert);
    }

    /**
     * Sign PDF bytes at the requested PAdES baseline level.
     *
     * `$level` accepts either a short tag (`'b'`, `'t'`, `'lt'`,
     * `'lta'` — matches Ruby's `Symbol` enum) or the legacy integer
     * ordinal constants (`self::LEVEL_B_B` … `LEVEL_B_LTA`).
     *
     * B-T / B-LT / B-LTA require a non-null `$tsaUrl` (RFC 3161
     * endpoint such as `http://timestamp.example.com`). B-B does not
     * need a TSA.
     *
     * Requires the cdylib to be built with the `signatures` feature
     * (and `tsa-client` for B-T/B-LT/B-LTA).
     *
     * @param string|int $level short tag (`'b'`/`'t'`/`'lt'`/`'lta'`)
     *                          or the LEVEL_B_* ordinal constant
     * @return string the signed PDF bytes
     * @throws ValidationException for bad args (mirrors Ruby ArgumentError)
     * @throws SignatureException when the underlying native call fails
     */
    public function sign(
        string $pdfBytes,
        string|int $level = self::LEVEL_B_B,
        ?string $tsaUrl = null,
        ?string $reason = null,
        ?string $location = null,
    ): string {
        if ($pdfBytes === '') {
            throw new ValidationException('pdf cannot be empty');
        }

        $levelCode = self::resolveLevel($level);

        if ($levelCode !== self::LEVEL_B_B && ($tsaUrl === null || $tsaUrl === '')) {
            throw new ValidationException(
                "PAdES level {$levelCode} requires a tsaUrl"
            );
        }
        if ($this->credentials === null) {
            throw new SignatureException('PdfSigner credentials have been freed');
        }

        // Build PadesSignOptionsC. PHP FFI: keep every CData buffer in a
        // local until the native call returns — refcount on the struct
        // alone isn't enough for unowned buffers, so we anchor them
        // explicitly. The Ruby binding does the same with `tsa_buf`,
        // `reason_buf`, `location_buf` locals (see pdf_signer.rb).
        $ffi = NativeLibrary::getInstance();
        $opts = $ffi->new('PadesSignOptionsC');

        // CRITICAL: PHP FFI's `$ffi->new()` does NOT zero-initialize the
        // returned memory — it just returns raw `emalloc()`. Setting fields
        // one-by-one leaves any UNSET slot full of stack garbage. For
        // `PadesSignOptionsC`, the const-qualified pointer fields
        // (`const uint8_t *const *certs`, …) ALSO ignore `$opts->certs = null`
        // assignments silently in some ext-ffi builds — so even with explicit
        // null assignments we can't rely on the chain-material slots being
        // zero. The Rust shim then dereferences register garbage on those
        // slots and segfaults at `&*options`. (Diagnosis: companion test
        // `tests/test_pkcs12_signing_opts.rs` builds the same struct with
        // `ptr::null()` everywhere and signs cleanly; PHP using only
        // assignments crashes.)
        //
        // Zero the struct explicitly via FFI::memset so every byte is
        // known-NULL/0 before we set what we care about.
        FFI::memset(FFI::addr($opts), 0, FFI::sizeof($opts));

        $opts->certificate_handle = FFI::cast('const void *', $this->credentials);

        // Anchor C strings so PHP doesn't free them mid-call.
        $tsaBuf = self::cString($tsaUrl);
        $reasonBuf = self::cString($reason);
        $locationBuf = self::cString($location);

        if ($tsaBuf !== null) {
            $opts->tsa_url = FFI::cast('const char *', $tsaBuf);
        }
        if ($reasonBuf !== null) {
            $opts->reason = FFI::cast('const char *', $reasonBuf);
        }
        if ($locationBuf !== null) {
            $opts->location = FFI::cast('const char *', $locationBuf);
        }
        $opts->level = $levelCode;

        $signed = $this->bindings->pdfSignBytesPadesOpts($pdfBytes, $opts);

        // Touch the buffers after the call so the optimiser can't
        // reorder them out of scope.
        unset($tsaBuf, $reasonBuf, $locationBuf, $opts);

        return $signed;
    }

    /**
     * Static convenience — sign without first constructing a signer
     * instance. Mirrors Ruby's `PdfOxide::PdfSigner.sign`.
     *
     * The caller retains ownership of `$certificateHandle`; this method
     * does NOT free it. Use {@see fromPkcs12()} + instance `sign()` if
     * you want lifetime-managed credentials.
     *
     * @param string|int $level short tag (`'b'`/`'t'`/`'lt'`/`'lta'`)
     *                          or the LEVEL_B_* ordinal constant
     * @return string the signed PDF bytes
     */
    public static function signWithHandle(
        string $pdfBytes,
        CData $certificateHandle,
        string|int $level,
        ?string $tsaUrl = null,
        ?string $reason = null,
        ?string $location = null,
    ): string {
        // Build a temporary signer that does NOT take ownership: we
        // null its $credentials before destruct so it doesn't free the
        // caller-owned handle.
        $signer = new self($certificateHandle);
        try {
            return $signer->sign($pdfBytes, $level, $tsaUrl, $reason, $location);
        } finally {
            // Disown — caller still owns $certificateHandle.
            $signer->disownCredentials();
        }
    }

    /**
     * @internal Used by {@see signWithHandle()} to release the borrowed
     * credentials handle without calling `pdf_certificate_free`.
     */
    private function disownCredentials(): void
    {
        $this->credentials = null;
    }

    /**
     * Coerce a level arg into its integer ordinal.
     *
     * @param string|int $level
     * @throws ValidationException if not a recognised level
     */
    private static function resolveLevel(string|int $level): int
    {
        if (is_int($level)) {
            if (! in_array($level, self::LEVELS, true)) {
                throw new ValidationException(
                    'level must be one of ' . implode(',', array_keys(self::LEVELS))
                    . " (or LEVEL_B_* ordinal), got {$level}"
                );
            }
            return $level;
        }
        $key = strtolower($level);
        if (! array_key_exists($key, self::LEVELS)) {
            throw new ValidationException(
                'level must be one of ' . implode(',', array_keys(self::LEVELS))
                . ", got " . var_export($level, true)
            );
        }
        return self::LEVELS[$key];
    }

    /**
     * Allocate a NUL-terminated C string for an optional PHP value, or
     * `null` if the input is null. Caller MUST anchor the returned
     * CData in a local for the duration of any subsequent FFI call.
     */
    private static function cString(?string $s): ?CData
    {
        if ($s === null) {
            return null;
        }
        $len = strlen($s);
        // +1 for NUL terminator. `false` = unowned — we rely on PHP's
        // refcount-driven free, NOT FFI's tracked allocator.
        $buf = FFI::new("char[" . ($len + 1) . "]", false);
        if ($len > 0) {
            FFI::memcpy($buf, $s, $len);
        }
        $buf[$len] = "\0";
        return $buf;
    }

    /**
     * @return bool true if the PDF carries at least one parseable
     *              signature (best-effort — full chain validation
     *              ships in a follow-up signature-verifier).
     *
     * <p>Uses the same marker-based check as
     * {@code tests/test_pkcs12_signing.rs} (a freshly signed PDF
     * MUST contain `/Sig` and `/ByteRange`). The cdylib's
     * {@code pdf_document_get_signature_count} doesn't yet pick up
     * signatures created by the same-process incremental-update
     * signing path; that's a tracked follow-up.
     */
    public static function verify(string $pdfBytes): bool
    {
        if ($pdfBytes === '') {
            return false;
        }
        return str_contains($pdfBytes, '/Sig') && str_contains($pdfBytes, '/ByteRange');
    }

    public function isOpen(): bool
    {
        return $this->credentials !== null;
    }

    public function close(): void
    {
        if ($this->credentials !== null) {
            // Real cdylib symbol: pdf_certificate_free.
            $this->bindings->pdfCertificateFree($this->credentials);
            $this->credentials = null;
        }
    }

    public function __destruct()
    {
        $this->close();
    }
}
