<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide;

use FFI\CData;
use PdfOxide\Exceptions\IoException;
use PdfOxide\Exceptions\SignatureException;
use PdfOxide\FFI\FunctionBindings;

/**
 * PAdES B-B / B-T / B-LT digital-signature signer + verifier
 * (v0.3.50 #235).
 *
 * Mirrors `fyi.oxide.pdf.PdfSigner` from the Java binding. Signing
 * routes through pdf_oxide's crypto-governance policy
 * ({@see PdfPolicy}) — bypassing the policy is impossible.
 *
 * Loads credentials once via {@see fromPkcs12()}; the credentials
 * handle is reused across multiple {@see sign()} calls and freed in
 * {@see close()} / `__destruct()`.
 */
final class PdfSigner
{
    // ─────────────── PAdES level ordinals ──────────────────
    // Frozen by the Rust ABI (`PadesLevel` in `src/signing/pades.rs`).

    public const LEVEL_B_B = 0;

    public const LEVEL_B_T = 1;

    public const LEVEL_B_LT = 2;

    public const LEVEL_B_LTA = 3;

    private ?CData $credentials = null;

    private readonly FunctionBindings $bindings;

    private function __construct(CData $credentials)
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
     * B-T / B-LT / B-LTA require a non-null `$tsaUrl` (RFC 3161
     * endpoint such as `http://timestamp.example.com`). B-B does not
     * need a TSA.
     *
     * Requires the cdylib to be built with the `signatures` feature
     * (and `tsa-client` for B-T/B-LT/B-LTA).
     *
     * @return string the signed PDF bytes
     * @throws SignatureException when the underlying sign call fails
     */
    public function sign(
        string $pdfBytes,
        int $level = self::LEVEL_B_B,
        ?string $tsaUrl = null,
        ?string $reason = null,
        ?string $location = null,
        ?string $contact = null,
    ): string {
        if ($level !== self::LEVEL_B_B && $tsaUrl === null) {
            throw new SignatureException(
                "PAdES level {$level} requires a tsaUrl"
            );
        }
        if ($this->credentials === null) {
            throw new SignatureException('PdfSigner credentials have been freed');
        }
        // The PHP binding does not yet pack PadesSignOptionsC and route through
        // `pdf_sign_bytes_pades_opts` (the canonical 5-arg shim). The Ruby
        // binding does — port in a follow-up. Until then, sign() is a stub
        // (mirrors Java's "stub until Phase 4 T15" status). Tracked in #546.
        throw new \BadMethodCallException(
            'PdfSigner::sign() is not yet wired in the PHP binding; '
            . 'use the Ruby/Java bindings for now (tracked in #546).'
        );
    }

    /**
     * @return bool true if the PDF carries at least one parseable
     *              signature (best-effort — full chain validation
     *              ships in a follow-up signature-verifier).
     */
    public static function verify(string $pdfBytes): bool
    {
        $tmp = tempnam(sys_get_temp_dir(), 'pdf_oxide_verify_');
        if ($tmp === false) {
            throw new IoException('Failed to allocate temp file for verify');
        }
        try {
            file_put_contents($tmp, $pdfBytes);
            $bindings = new FunctionBindings();
            $handle = $bindings->pdfDocumentOpen($tmp);
            if ($handle === null) {
                return false;
            }
            try {
                // C ABI exposes `pdf_document_get_signature_count(handle, err*)`.
                // Bypass FunctionBindings (which targets a wrapper symbol).
                $ffi = \PdfOxide\FFI\NativeLibrary::getInstance();
                $errorCode = \FFI::new('int32_t');
                $count = (int) $ffi->pdf_document_get_signature_count($handle, \FFI::addr($errorCode));
                return $count > 0;
            } finally {
                $bindings->pdfDocumentFree($handle);
            }
        } finally {
            @unlink($tmp);
        }
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
