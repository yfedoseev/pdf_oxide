<?php

declare(strict_types=1);

namespace PdfOxide\Utilities;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;

/**
 * Helper for advanced PDF rendering operations.
 *
 * Provides convenient methods for creating and configuring rendering options.
 */
class RenderingHelper
{
    private FunctionBindings $bindings;

    public function __construct()
    {
        $this->bindings = new FunctionBindings();
    }

    /**
     * Image format constants.
     */
    public const FORMAT_JPEG = 0;
    public const FORMAT_PNG = 1;
    public const FORMAT_TIFF = 2;
    public const FORMAT_BMP = 3;

    /**
     * Interpolation method constants.
     */
    public const INTERPOLATION_NEAREST = 0;
    public const INTERPOLATION_LINEAR = 1;
    public const INTERPOLATION_CUBIC = 2;
    public const INTERPOLATION_HIGH_QUALITY = 3;

    /**
     * Create rendering options with common presets.
     */
    public function createOptions(float $dpi = 150.0, int $format = self::FORMAT_PNG, int $quality = 85): ?CData
    {
        return $this->bindings->pdfCreateRenderingOptions($dpi, $format, $quality);
    }

    /**
     * Create high-quality rendering options.
     */
    public function createHighQualityOptions(): ?CData
    {
        return $this->createOptions(300.0, self::FORMAT_PNG, 95);
    }

    /**
     * Create screen-resolution rendering options.
     */
    public function createScreenOptions(): ?CData
    {
        return $this->createOptions(72.0, self::FORMAT_PNG, 75);
    }

    /**
     * Create print-quality rendering options.
     */
    public function createPrintOptions(): ?CData
    {
        return $this->createOptions(300.0, self::FORMAT_TIFF, 100);
    }

    /**
     * Create web-optimized rendering options.
     */
    public function createWebOptions(): ?CData
    {
        return $this->createOptions(150.0, self::FORMAT_JPEG, 80);
    }

    /**
     * Set option on rendering options object.
     */
    public function setOption(?CData $opts, string $key, string $value): void
    {
        if ($opts !== null) {
            $this->bindings->pdfRenderingOptionsSet($opts, $key, $value);
        }
    }

    /**
     * Get supported image formats as array.
     */
    public function getSupportedFormats(): array
    {
        return $this->bindings->pdfGetSupportedImageFormats();
    }

    /**
     * Get format name from constant.
     */
    public static function getFormatName(int $format): string
    {
        return match ($format) {
            self::FORMAT_JPEG => 'JPEG',
            self::FORMAT_PNG => 'PNG',
            self::FORMAT_TIFF => 'TIFF',
            self::FORMAT_BMP => 'BMP',
            default => 'Unknown',
        };
    }

    /**
     * Get interpolation method name.
     */
    public static function getInterpolationName(int $method): string
    {
        return match ($method) {
            self::INTERPOLATION_NEAREST => 'Nearest Neighbor',
            self::INTERPOLATION_LINEAR => 'Linear',
            self::INTERPOLATION_CUBIC => 'Cubic',
            self::INTERPOLATION_HIGH_QUALITY => 'High Quality',
            default => 'Unknown',
        };
    }
}
