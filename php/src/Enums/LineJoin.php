<?php

declare(strict_types=1);

namespace PdfOxide\Enums;

/**
 * Line join styles for PDF drawing operations.
 */
enum LineJoin: string
{
    case MITER = 'Miter';
    case ROUND = 'Round';
    case BEVEL = 'Bevel';

    /**
     * Get numeric value for FFI calls.
     */
    public function toValue(): int
    {
        return match ($this) {
            self::MITER => 0,
            self::ROUND => 1,
            self::BEVEL => 2,
        };
    }

    /**
     * Get description.
     */
    public function description(): string
    {
        return match ($this) {
            self::MITER => 'Sharp pointed join',
            self::ROUND => 'Rounded join',
            self::BEVEL => 'Beveled join',
        };
    }
}
