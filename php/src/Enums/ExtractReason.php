<?php

declare(strict_types=1);

namespace PdfOxide\Enums;

/**
 * Typed reason explaining why an auto-extraction or classification
 * is in a particular state.
 *
 * The v0.3.51 "tell me why" feature (#519): a non-degraded result is
 * {@see ExtractReason::Ok}; any degraded outcome MUST name the cause.
 *
 * The string backing values are the canonical snake_case wire tokens
 * from the Rust `ReasonCode` enum (`src/extractors/auto.rs`) — they
 * survive the JSON envelope at the FFI boundary and must never be
 * renamed/renumbered (frozen for cross-binding parity with Python /
 * Java / Node / Go / C# / Ruby).
 *
 * Mirrors the canonical taxonomy used by every other pdf_oxide
 * binding; the Java reference is `fyi.oxide.pdf.auto.ExtractReason`.
 */
enum ExtractReason: string
{
    /** Extracted cleanly — no degradation. */
    case Ok = 'ok';

    /** Native text present and high-confidence. */
    case NativeTextHighConfidence = 'native_text_high_confidence';

    /** No text layer on the page at all; OCR ran (if available) or wasn't requested. */
    case NoTextLayerPresent = 'no_text_layer_present';

    /** Text layer present but below the usable-quality threshold. */
    case TextLayerBelowThreshold = 'text_layer_below_threshold';

    /** Glyphs without usable ToUnicode / `(cid:NN)` / garbled mapping. */
    case GlyphMappingMissing = 'glyph_mapping_missing';

    /** Encrypted and not authorised to extract. */
    case EncryptedNoExtractPermission = 'encrypted_no_extract_permission';

    /** An image-table was reconstructed into structured TableData. */
    case ImageTableReconstructed = 'image_table_reconstructed';

    /** A table region was detected but structure could not be recovered. */
    case ImageTableNoStructure = 'image_table_no_structure';

    /** A chart/figure was detected; its internal data is NOT transcribed. */
    case ChartNotTranscribed = 'chart_not_transcribed';

    /**
     * OCR was needed but unavailable (feature off / models absent /
     * `mode = TextOnly`) → fell back to native text + warned.
     */
    case OcrRequestedButUnavailable = 'ocr_requested_but_unavailable';

    /** OCR ran but confidence was low → native used/merged. */
    case OcrLowConfidenceFallback = 'ocr_low_confidence_fallback';

    /** Region/page yielded no recoverable content. */
    case Empty = 'empty';

    /**
     * Parse a Rust-side snake_case wire token into an ExtractReason.
     *
     * Tolerant: unknown values map to {@see ExtractReason::Ok} since
     * the Rust enum is `#[non_exhaustive]` and may grow new variants
     * in a future minor release.
     */
    public static function fromWire(?string $wire): self
    {
        if ($wire === null) {
            return self::Ok;
        }
        return self::tryFrom($wire) ?? self::Ok;
    }
}
