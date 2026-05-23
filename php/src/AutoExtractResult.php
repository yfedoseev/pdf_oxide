<?php

declare(strict_types=1);

namespace PdfOxide;

use PdfOxide\Enums\ExtractReason;
use PdfOxide\Enums\PageKind;

/**
 * Result of an {@see AutoExtractor} call.
 *
 * Carries the extracted text + the typed reason explaining the
 * quality / fallback state + an optional decoded JSON envelope from
 * the FFI boundary.
 *
 * Frozen readonly value object (PHP 8.2+). Properties carry the same
 * names as Python's `AutoExtractResult` dataclass for cross-binding
 * documentation parity.
 */
final readonly class AutoExtractResult
{
    public function __construct(
        public string $text,
        public ExtractReason $reason,
        public PageKind $kind,
        public float $confidence,
        /** @var array<string,mixed>|null */
        public ?array $classification,
    ) {
    }

    /** Whether the extraction succeeded with no degradation. */
    public function isOk(): bool
    {
        return $this->reason === ExtractReason::Ok
            || $this->reason === ExtractReason::NativeTextHighConfidence;
    }

    /** Whether the OCR-unavailable graceful-fallback path engaged. */
    public function isOcrFallback(): bool
    {
        return $this->reason === ExtractReason::OcrRequestedButUnavailable
            || $this->reason === ExtractReason::OcrLowConfidenceFallback;
    }
}
