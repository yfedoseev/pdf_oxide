<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\Exceptions\ValidationException;
use PdfOxide\PdfSigner;

/**
 * Real-cdylib sign() tests for {@see PdfSigner}, exercising the
 * v0.3.55 #546 PadesSignOptionsC struct-packing implementation.
 *
 * Mirrors `ruby/spec/pdf_signer_spec.rb` and the Java
 * `PdfSignerTest` sign cases. Auto-skips when the cdylib isn't
 * reachable (IntegrationTestCase contract).
 *
 * @requires extension ffi
 */
final class PdfSignerSignTest extends IntegrationTestCase
{
    /**
     * Path to the PKCS#12 keystore shared with the Rust integration
     * tests. Password is `testpass` — see
     * `tests/test_pkcs12_signing.rs` and the README in
     * `tests/fixtures/`.
     */
    private const KEYSTORE_PASSWORD = 'testpass';

    /**
     * Locate the shared PKCS#12 fixture. Lives at the repo-wide
     * `tests/fixtures/test_signing.p12` (committed in 10b417e6).
     * Skips the test rather than failing when missing — the fixture
     * is gated behind the `signatures` cargo feature in CI.
     */
    private function keystorePath(): string
    {
        $candidates = [
            __DIR__ . '/../fixtures/test_signing.p12',
            __DIR__ . '/../../../tests/fixtures/test_signing.p12',
        ];
        foreach ($candidates as $c) {
            if (is_file($c)) {
                return (string) realpath($c);
            }
        }
        $this->markTestSkipped('test_signing.p12 fixture not found');
    }

    public function testSignAtLevelBProducesPdf(): void
    {
        $pdfPath = $this->fixture('tiny.pdf');
        $pdfBytes = (string) file_get_contents($pdfPath);

        $signer = PdfSigner::fromPkcs12($this->keystorePath(), self::KEYSTORE_PASSWORD);
        try {
            $signed = $signer->sign($pdfBytes, 'b');
        } finally {
            $signer->close();
        }

        $this->assertNotSame('', $signed, 'signed PDF must be non-empty');
        $this->assertStringStartsWith(
            '%PDF-',
            $signed,
            'signed output must carry the PDF header'
        );
        $this->assertGreaterThan(
            strlen($pdfBytes),
            strlen($signed),
            'signed PDF must be larger than the input (incremental update)'
        );
    }

    public function testSignWithHandleStaticConvenience(): void
    {
        $pdfPath = $this->fixture('tiny.pdf');
        $pdfBytes = (string) file_get_contents($pdfPath);

        // Borrow the cert handle from a manual loader so we can prove
        // that signWithHandle() does NOT free it (we close it after).
        $signer = PdfSigner::fromPkcs12($this->keystorePath(), self::KEYSTORE_PASSWORD);
        try {
            // Reach in via reflection to get the underlying handle for
            // the static-helper path.
            $ref = new \ReflectionClass(PdfSigner::class);
            $prop = $ref->getProperty('credentials');
            $prop->setAccessible(true);
            $handle = $prop->getValue($signer);
            $this->assertNotNull($handle);

            $signed = PdfSigner::signWithHandle($pdfBytes, $handle, 'b');
            $this->assertStringStartsWith('%PDF-', $signed);

            // The handle must still be usable after signWithHandle — if
            // signWithHandle had freed it, this second call would
            // dereference freed memory. (Best-effort proof; segfault on
            // failure.)
            $signed2 = PdfSigner::signWithHandle($pdfBytes, $handle, 'b');
            $this->assertStringStartsWith('%PDF-', $signed2);
        } finally {
            $signer->close();
        }
    }

    public function testSignRejectsEmptyPdf(): void
    {
        $signer = PdfSigner::fromPkcs12($this->keystorePath(), self::KEYSTORE_PASSWORD);
        try {
            $this->expectException(ValidationException::class);
            $this->expectExceptionMessageMatches('/pdf/i');
            $signer->sign('', 'b');
        } finally {
            $signer->close();
        }
    }

    public function testSignRejectsUnknownLevel(): void
    {
        $signer = PdfSigner::fromPkcs12($this->keystorePath(), self::KEYSTORE_PASSWORD);
        try {
            $this->expectException(ValidationException::class);
            $this->expectExceptionMessageMatches('/level/');
            $signer->sign('%PDF-1.7', 'forged');
        } finally {
            $signer->close();
        }
    }

    public function testSignRequiresTsaUrlForLevelT(): void
    {
        $signer = PdfSigner::fromPkcs12($this->keystorePath(), self::KEYSTORE_PASSWORD);
        try {
            $this->expectException(ValidationException::class);
            $this->expectExceptionMessageMatches('/tsaUrl|tsa_url/i');
            $signer->sign('%PDF-1.7', 't');
        } finally {
            $signer->close();
        }
    }

    public function testSignedPdfPassesVerify(): void
    {
        $pdfPath = $this->fixture('tiny.pdf');
        $pdfBytes = (string) file_get_contents($pdfPath);

        $signer = PdfSigner::fromPkcs12($this->keystorePath(), self::KEYSTORE_PASSWORD);
        try {
            $signed = $signer->sign($pdfBytes, 'b');
        } finally {
            $signer->close();
        }

        $this->assertTrue(
            PdfSigner::verify($signed),
            'signed PDF must report at least one signature via verify()'
        );
    }

    public function testLevelCanBeIntegerOrdinal(): void
    {
        $pdfPath = $this->fixture('tiny.pdf');
        $pdfBytes = (string) file_get_contents($pdfPath);

        $signer = PdfSigner::fromPkcs12($this->keystorePath(), self::KEYSTORE_PASSWORD);
        try {
            $signed = $signer->sign($pdfBytes, PdfSigner::LEVEL_B_B);
            $this->assertStringStartsWith('%PDF-', $signed);
        } finally {
            $signer->close();
        }
    }
}
