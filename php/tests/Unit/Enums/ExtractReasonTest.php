<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Unit\Enums;

use PdfOxide\Enums\ExtractReason;
use PHPUnit\Framework\TestCase;

/**
 * Unit tests for {@see ExtractReason}.
 *
 * The string backing values are FROZEN by the v0.3.51 cross-binding
 * contract — they ride the JSON envelope from Rust serde and any rename
 * would corrupt every binding silently. These tests guard the wire
 * format.
 */
final class ExtractReasonTest extends TestCase
{
    public function testWireTokensFrozen(): void
    {
        $this->assertSame('ok', ExtractReason::Ok->value);
        $this->assertSame('native_text_high_confidence', ExtractReason::NativeTextHighConfidence->value);
        $this->assertSame('no_text_layer_present', ExtractReason::NoTextLayerPresent->value);
        $this->assertSame('text_layer_below_threshold', ExtractReason::TextLayerBelowThreshold->value);
        $this->assertSame('glyph_mapping_missing', ExtractReason::GlyphMappingMissing->value);
        $this->assertSame('encrypted_no_extract_permission', ExtractReason::EncryptedNoExtractPermission->value);
        $this->assertSame('image_table_reconstructed', ExtractReason::ImageTableReconstructed->value);
        $this->assertSame('image_table_no_structure', ExtractReason::ImageTableNoStructure->value);
        $this->assertSame('chart_not_transcribed', ExtractReason::ChartNotTranscribed->value);
        $this->assertSame('ocr_requested_but_unavailable', ExtractReason::OcrRequestedButUnavailable->value);
        $this->assertSame('ocr_low_confidence_fallback', ExtractReason::OcrLowConfidenceFallback->value);
        $this->assertSame('empty', ExtractReason::Empty->value);
    }

    public function testFromWireMapsCanonicalTokens(): void
    {
        $this->assertSame(ExtractReason::Ok, ExtractReason::fromWire('ok'));
        $this->assertSame(
            ExtractReason::OcrRequestedButUnavailable,
            ExtractReason::fromWire('ocr_requested_but_unavailable')
        );
    }

    public function testFromWireFallsBackToOkOnUnknown(): void
    {
        // Rust enum is `#[non_exhaustive]` — unknown variants in a
        // future minor must not crash; they degrade to Ok.
        $this->assertSame(ExtractReason::Ok, ExtractReason::fromWire('a_future_reason_we_dont_know_about'));
    }

    public function testFromWireHandlesNull(): void
    {
        $this->assertSame(ExtractReason::Ok, ExtractReason::fromWire(null));
    }
}
