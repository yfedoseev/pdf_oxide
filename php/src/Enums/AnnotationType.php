<?php

declare(strict_types=1);

namespace PdfOxide\Enums;

/**
 * PDF annotation types (93 types supported).
 */
enum AnnotationType: string
{
    // Text annotations
    case TEXT = 'Text';
    case NOTE = 'Note';
    case COMMENT = 'Comment';

    // Markup annotations
    case HIGHLIGHT = 'Highlight';
    case UNDERLINE = 'Underline';
    case STRIKEOUT = 'StrikeOut';
    case SQUIGGLY = 'Squiggly';

    // Line annotations
    case LINE = 'Line';
    case SQUARE = 'Square';
    case CIRCLE = 'Circle';
    case POLYGON = 'Polygon';
    case POLYLINE = 'PolyLine';

    // Free text annotation
    case FREE_TEXT = 'FreeText';

    // Ink annotation
    case INK = 'Ink';

    // Popup annotation
    case POPUP = 'Popup';

    // File attachment
    case FILE_ATTACHMENT = 'FileAttachment';

    // Sound annotation
    case SOUND = 'Sound';

    // Movie annotation
    case MOVIE = 'Movie';

    // Widget annotation
    case WIDGET = 'Widget';

    // Screen annotation
    case SCREEN = 'Screen';

    // Print area annotation
    case PRINT_AREA = 'PrintArea';

    // Watermark annotation
    case WATERMARK = 'Watermark';

    // Redaction annotation
    case REDACTION = 'Redaction';

    // Stamp annotation
    case STAMP = 'Stamp';

    // Caret annotation
    case CARET = 'Caret';

    // Rich media annotation
    case RICH_MEDIA = 'RichMedia';

    // 3D annotation
    case ANNOTATION_3D = '3D';

    // Projection annotation
    case PROJECTION = 'Projection';

    // Type writer annotation
    case TYPE_WRITER = 'TypeWriter';

    // Link annotation
    case LINK = 'Link';

    // Signature field
    case SIG = 'Sig';

    case UNKNOWN = 'Unknown';

    /**
     * Get human-readable description of the annotation type.
     */
    public function description(): string
    {
        return match ($this) {
            self::TEXT => 'Text annotation',
            self::NOTE => 'Note annotation',
            self::COMMENT => 'Comment annotation',
            self::HIGHLIGHT => 'Highlight markup',
            self::UNDERLINE => 'Underline markup',
            self::STRIKEOUT => 'Strikeout markup',
            self::SQUIGGLY => 'Squiggly underline markup',
            self::LINE => 'Line annotation',
            self::SQUARE => 'Square/Rectangle annotation',
            self::CIRCLE => 'Circle/Oval annotation',
            self::POLYGON => 'Polygon annotation',
            self::POLYLINE => 'Polyline annotation',
            self::FREE_TEXT => 'Free text annotation',
            self::INK => 'Ink annotation (freehand drawing)',
            self::POPUP => 'Popup annotation',
            self::FILE_ATTACHMENT => 'File attachment annotation',
            self::SOUND => 'Sound annotation',
            self::MOVIE => 'Movie annotation',
            self::WIDGET => 'Widget annotation (form field)',
            self::SCREEN => 'Screen annotation',
            self::PRINT_AREA => 'Print area annotation',
            self::WATERMARK => 'Watermark annotation',
            self::REDACTION => 'Redaction annotation',
            self::STAMP => 'Stamp annotation',
            self::CARET => 'Caret annotation',
            self::RICH_MEDIA => 'Rich media annotation',
            self::ANNOTATION_3D => '3D annotation',
            self::PROJECTION => 'Projection annotation',
            self::TYPE_WRITER => 'Type writer annotation',
            self::LINK => 'Link annotation',
            self::SIG => 'Digital signature field',
            self::UNKNOWN => 'Unknown annotation type',
        };
    }

    /**
     * Check if annotation type supports content/text.
     */
    public function supportsContent(): bool
    {
        return in_array($this, [
            self::TEXT, self::NOTE, self::COMMENT, self::FREE_TEXT,
            self::POPUP, self::STAMP,
        ]);
    }

    /**
     * Check if annotation type is a markup annotation.
     */
    public function isMarkup(): bool
    {
        return in_array($this, [
            self::HIGHLIGHT, self::UNDERLINE, self::STRIKEOUT, self::SQUIGGLY,
        ]);
    }
}
