<?php

declare(strict_types=1);

namespace PdfOxide\Enums;

/**
 * Line cap styles for PDF drawing operations.
 */
enum LineCap: string
{
    case BUTT = 'Butt';
    case ROUND = 'Round';
    case SQUARE = 'Square';

    /**
     * Get numeric value for FFI calls.
     */
    public function toValue(): int
    {
        return match ($this) {
            self::BUTT => 0,
            self::ROUND => 1,
            self::SQUARE => 2,
        };
    }

    /**
     * Get description.
     */
    public function description(): string
    {
        return match ($this) {
            self::BUTT => 'Flat square end (no extension)',
            self::ROUND => 'Rounded end',
            self::SQUARE => 'Extended square end',
        };
    }
}
