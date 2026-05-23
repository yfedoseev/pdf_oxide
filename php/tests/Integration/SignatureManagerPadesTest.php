<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\Enums\PadesLevel;
use PdfOxide\PdfDocument;
use PHPUnit\Framework\TestCase;

/**
 * Smoke test for v0.3.50 #235 PAdES sign / verify high-level API.
 *
 * Full sign-with-cert path needs a test certificate fixture (not
 * shipped in v0.3.55 Phase 6); this test asserts the surface loads
 * cleanly + the enum is wired + `hasDocumentTimestamp()` returns a
 * boolean.
 */
final class SignatureManagerPadesTest extends TestCase
{
    protected function setUp(): void
    {
        if (PDF_OXIDE_NATIVE_LIB === null) {
            $this->markTestSkipped('libpdf_oxide not built.');
        }
        if (PDF_OXIDE_SAMPLE_PDF === null) {
            $this->markTestSkipped('No sample PDF available.');
        }
    }

    public function testPadesLevelEnumOrdinalsFrozen(): void
    {
        // The integer ordinals are FROZEN by the Rust ABI; renumbering
        // would silently corrupt signatures across binding boundaries.
        $this->assertSame(0, PadesLevel::BB->value);
        $this->assertSame(1, PadesLevel::BT->value);
        $this->assertSame(2, PadesLevel::BLT->value);
        $this->assertSame(3, PadesLevel::BLTA->value);

        $this->assertFalse(PadesLevel::BB->requiresTsa());
        $this->assertTrue(PadesLevel::BT->requiresTsa());
        $this->assertTrue(PadesLevel::BLT->requiresTsa());
        $this->assertTrue(PadesLevel::BLTA->requiresTsa());
    }

    public function testSignaturesAccessorAndTimestampQuery(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $sigs = $doc->signatures();
        $this->assertNotNull($sigs);
        // hasDocumentTimestamp wraps `pdf_document_has_timestamp` —
        // a v0.3.50 ABI that returns 0/1 without error for unsigned
        // PDFs. (Contrast `getSignatureCount`, which the pre-existing
        // scaffold wires to a Rust path that classifies "no sigs" as
        // a signature error; that's tracked as a separate scaffold bug.)
        $this->assertIsBool($sigs->hasDocumentTimestamp());
    }

    public function testSignPadesRequiresTsaForBT(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $sigs = $doc->signatures();

        // No certificate, but the InvalidArgumentException must fire
        // BEFORE the FFI call (we validate $level->requiresTsa() first).
        $this->expectException(\InvalidArgumentException::class);

        // We pass a dummy CData; never reached because the TSA check
        // fires first. Use a real FFI 'char' allocation to satisfy the
        // type-hint without actually constructing a certificate.
        $dummy = \FFI::new('char');
        $sigs->signPades(
            pdfData: 'irrelevant',
            certificateHandle: $dummy,
            level: PadesLevel::BT,
            tsaUrl: null,
        );
    }
}
