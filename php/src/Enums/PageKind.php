<?php

declare(strict_types=1);

namespace PdfOxide\Enums;

/**
 * Auto-classifier's per-page kind, mirroring Rust's
 * `pdf_oxide::extractors::auto::PageKind`.
 *
 * Wire tokens are the snake_case strings emitted by serde at the FFI
 * JSON boundary. Frozen for cross-binding parity.
 */
enum PageKind: string
{
    /** Native text layer is present and high-quality. */
    case TextLayer = 'text_layer';

    /** No text layer; the page is essentially a scan / raster image. */
    case Scanned = 'scanned';

    /** Page has both native text and images-that-contain-text. */
    case ImageText = 'image_text';

    /** Mixed kind (table + figures + body etc.). */
    case Mixed = 'mixed';

    /** Page produced no recoverable content. */
    case Empty = 'empty';

    public static function fromWire(?string $wire): self
    {
        if ($wire === null) {
            return self::Mixed;
        }
        return self::tryFrom($wire) ?? self::Mixed;
    }
}
