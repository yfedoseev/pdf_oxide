<?php

declare(strict_types=1);

namespace PdfOxide\Types;

/**
 * Represents an RGBA color.
 */
readonly class Color
{
    public function __construct(
        public int $red,
        public int $green,
        public int $blue,
        public int $alpha = 255
    ) {
        if ($red < 0 || $red > 255 || $green < 0 || $green > 255
            || $blue < 0 || $blue > 255 || $alpha < 0 || $alpha > 255) {
            throw new \ValueError('Color values must be between 0 and 255');
        }
    }

    /**
     * Create a color from hex string.
     *
     * @param string $hex Hex color string (e.g., '#FF0000' or 'FF0000')
     */
    public static function fromHex(string $hex): self
    {
        $hex = ltrim($hex, '#');

        if (strlen($hex) !== 6 && strlen($hex) !== 8) {
            throw new \ValueError('Invalid hex color format');
        }

        $red = hexdec(substr($hex, 0, 2));
        $green = hexdec(substr($hex, 2, 2));
        $blue = hexdec(substr($hex, 4, 2));
        $alpha = strlen($hex) === 8 ? hexdec(substr($hex, 6, 2)) : 255;

        return new self($red, $green, $blue, $alpha);
    }

    /**
     * Create a color from RGB values (0-1 float range).
     */
    public static function fromRgbFloat(float $red, float $green, float $blue, float $alpha = 1.0): self
    {
        return new self(
            (int)round($red * 255),
            (int)round($green * 255),
            (int)round($blue * 255),
            (int)round($alpha * 255)
        );
    }

    /**
     * Convert to hex string.
     */
    public function toHex(bool $includeAlpha = false): string
    {
        $hex = sprintf('%02X%02X%02X', $this->red, $this->green, $this->blue);
        if ($includeAlpha) {
            $hex .= sprintf('%02X', $this->alpha);
        }
        return '#' . $hex;
    }

    /**
     * Convert to 32-bit integer (ARGB format).
     */
    public function toArgb(): int
    {
        return ($this->alpha << 24) | ($this->red << 16) | ($this->green << 8) | $this->blue;
    }

    /**
     * Convert to 32-bit integer (RGBA format).
     */
    public function toRgba(): int
    {
        return ($this->red << 24) | ($this->green << 16) | ($this->blue << 8) | $this->alpha;
    }

    /**
     * Convert to array format.
     */
    public function toArray(): array
    {
        return [
            'red' => $this->red,
            'green' => $this->green,
            'blue' => $this->blue,
            'alpha' => $this->alpha,
        ];
    }

    /**
     * Create common colors.
     */
    public static function black(): self { return new self(0, 0, 0); }
    public static function white(): self { return new self(255, 255, 255); }
    public static function red(): self { return new self(255, 0, 0); }
    public static function green(): self { return new self(0, 128, 0); }
    public static function blue(): self { return new self(0, 0, 255); }
    public static function yellow(): self { return new self(255, 255, 0); }
    public static function cyan(): self { return new self(0, 255, 255); }
    public static function magenta(): self { return new self(255, 0, 255); }
    public static function gray(): self { return new self(128, 128, 128); }
}
