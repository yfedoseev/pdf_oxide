<?php

declare(strict_types=1);

namespace PdfOxide\Enums;

/**
 * Standard page sizes for PDF creation.
 */
enum PageSize: string
{
    case A0 = 'A0';
    case A1 = 'A1';
    case A2 = 'A2';
    case A3 = 'A3';
    case A4 = 'A4';
    case A5 = 'A5';
    case A6 = 'A6';
    case LETTER = 'LETTER';
    case LEGAL = 'LEGAL';
    case TABLOID = 'TABLOID';
    case LEDGER = 'LEDGER';

    /**
     * Get the width and height in points for this page size.
     *
     * @return array{width: float, height: float} Width and height in points
     */
    public function getDimensions(): array
    {
        return match ($this) {
            // ISO A series (each halves the previous)
            self::A0 => ['width' => 2384, 'height' => 3370],
            self::A1 => ['width' => 1684, 'height' => 2384],
            self::A2 => ['width' => 1191, 'height' => 1684],
            self::A3 => ['width' => 842, 'height' => 1191],
            self::A4 => ['width' => 595, 'height' => 842],
            self::A5 => ['width' => 420, 'height' => 595],
            self::A6 => ['width' => 298, 'height' => 420],
            // North American
            self::LETTER => ['width' => 612, 'height' => 792],
            self::LEGAL => ['width' => 612, 'height' => 1008],
            self::TABLOID => ['width' => 792, 'height' => 1224],
            self::LEDGER => ['width' => 1224, 'height' => 792],
        };
    }

    /**
     * Get the width in millimeters.
     */
    public function getWidthMm(): float
    {
        return round($this->getDimensions()['width'] * 0.352778, 1);
    }

    /**
     * Get the height in millimeters.
     */
    public function getHeightMm(): float
    {
        return round($this->getDimensions()['height'] * 0.352778, 1);
    }
}
