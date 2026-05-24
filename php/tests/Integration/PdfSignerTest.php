<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\Exceptions\IoException;
use PdfOxide\PdfSigner;

/**
 * Smoke tests for {@see PdfSigner}, mirroring
 * `java/src/test/java/fyi/oxide/pdf/PdfSignerTest.java`.
 *
 * Signing requires the cdylib to be built with the `signatures`
 * feature (and `tsa-client` for B-T/B-LT). The verify+constants tests
 * here don't need it; full sign tests run when a PKCS#12 fixture is
 * present.
 */
final class PdfSignerTest extends IntegrationTestCase
{
    public function testPadesLevelConstants(): void
    {
        $this->assertSame(0, PdfSigner::LEVEL_B_B);
        $this->assertSame(1, PdfSigner::LEVEL_B_T);
        $this->assertSame(2, PdfSigner::LEVEL_B_LT);
        $this->assertSame(3, PdfSigner::LEVEL_B_LTA);
    }

    public function testFromPkcs12NonexistentThrowsIo(): void
    {
        $this->expectException(IoException::class);
        PdfSigner::fromPkcs12('/tmp/__pdf_oxide_p12_does_not_exist__.p12', 'pw');
    }

    public function testVerifyOnUnsignedPdfReturnsFalse(): void
    {
        $bytes = (string) file_get_contents($this->fixture('simple.pdf'));
        $this->assertFalse(PdfSigner::verify($bytes));
    }

    public function testFromPkcs12LoadsRealKeystore(): void
    {
        $p12 = $this->fixture('signatures/test_keystore.p12');
        $signer = PdfSigner::fromPkcs12($p12, 'test');
        try {
            $this->assertTrue($signer->isOpen());
        } finally {
            $signer->close();
        }
    }
}
