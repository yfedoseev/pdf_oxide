<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\Enums\PadesLevel;
use PdfOxide\PdfDocument;
use PHPUnit\Framework\TestCase;

/**
 * Integration tests for v0.3.50 #235 PAdES sign / verify high-level
 * API and the no-signatures-doc graceful path (Phase 7 scaffold-bug
 * fix).
 *
 * Full sign-with-cert path needs a test certificate fixture (deferred
 * to a future phase); this test asserts the surface loads cleanly +
 * the enum is wired + `hasDocumentTimestamp()` returns a boolean +
 * unsigned-PDF queries no longer throw.
 *
 * @requires extension ffi
 */
final class SignatureManagerPadesTest extends TestCase
{
    protected function setUp(): void
    {
        if (! extension_loaded('ffi')) {
            $this->markTestSkipped('ext-ffi not loaded');
        }
        if (PDF_OXIDE_NATIVE_LIB === null) {
            $this->markTestSkipped('libpdf_oxide cdylib not found.');
        }
        if (PDF_OXIDE_SAMPLE_PDF === null) {
            $this->markTestSkipped('No sample PDF available.');
        }
    }

    public function testPadesLevelEnumOrdinalsFrozen(): void
    {
        // FROZEN by the Rust ABI; renumbering would silently corrupt
        // signatures across binding boundaries.
        $this->assertSame(0, PadesLevel::BB->value);
        $this->assertSame(1, PadesLevel::BT->value);
        $this->assertSame(2, PadesLevel::BLT->value);
        $this->assertSame(3, PadesLevel::BLTA->value);

        $this->assertFalse(PadesLevel::BB->requiresTsa());
        $this->assertTrue(PadesLevel::BT->requiresTsa());
        $this->assertTrue(PadesLevel::BLT->requiresTsa());
        $this->assertTrue(PadesLevel::BLTA->requiresTsa());
    }

    public function testNoSignaturesDocReturnsZeroCount(): void
    {
        // Pre-Phase 7 scaffold bug: `getSignatureCount()` raised on
        // unsigned PDFs because the underlying ABI surfaces the
        // absent-AcroForm path as an error. The Phase 7 fix degrades
        // to 0 to match Python / Java.
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $sigs = $doc->signatures();

        $count = $sigs->getSignatureCount();
        $this->assertIsInt($count);
        $this->assertSame(0, $count, 'Unsigned PDF must report 0 signatures, not throw.');

        $this->assertFalse($sigs->hasSignatures());
        $this->assertSame([], $sigs->getSignatures());
    }

    public function testSignaturesAccessorAndTimestampQuery(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $sigs = $doc->signatures();
        $this->assertNotNull($sigs);
        $this->assertIsBool($sigs->hasDocumentTimestamp());
    }

    public function testSignPadesRequiresTsaForBT(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $sigs = $doc->signatures();

        // The InvalidArgumentException must fire BEFORE the FFI call
        // (we validate $level->requiresTsa() first).
        $this->expectException(\InvalidArgumentException::class);

        $dummy = \FFI::new('char');
        $sigs->signPades(
            pdfData: 'irrelevant',
            certificateHandle: $dummy,
            level: PadesLevel::BT,
            tsaUrl: null,
        );
    }
}
