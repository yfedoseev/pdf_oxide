<?php

declare(strict_types=1);

namespace PdfOxide\Enums;

/**
 * Image format for PDF page rendering.
 *
 * Specifies the output format when rendering PDF pages to images.
 *
 * @since 0.4.0
 */
enum ImageFormat: string
{
    /**
     * PNG format - lossless compression, supports transparency.
     * Best for: Graphics, screenshots, documents requiring transparency.
     */
    case PNG = 'png';

    /**
     * JPEG format - lossy compression, smaller file sizes.
     * Best for: Photographs, general document rendering.
     */
    case JPEG = 'jpeg';

    /**
     * WebP format - modern format with better compression than JPEG.
     * Best for: Web delivery, reduced bandwidth requirements.
     */
    case WEBP = 'webp';

    /**
     * Get MIME type for this format.
     *
     * @return string MIME type string
     *
     * Example:
     *     echo ImageFormat::PNG->mimeType();  // 'image/png'
     */
    public function mimeType(): string
    {
        return match ($this) {
            self::PNG => 'image/png',
            self::JPEG => 'image/jpeg',
            self::WEBP => 'image/webp',
        };
    }

    /**
     * Get file extension for this format.
     *
     * @return string File extension without leading dot
     *
     * Example:
     *     echo ImageFormat::PNG->extension();  // 'png'
     */
    public function extension(): string
    {
        return match ($this) {
            self::PNG => 'png',
            self::JPEG => 'jpg',
            self::WEBP => 'webp',
        };
    }

    /**
     * Get human-readable description.
     *
     * @return string Description of the format
     *
     * Example:
     *     echo ImageFormat::JPEG->description();
     *     // 'JPEG - lossy compression format'
     */
    public function description(): string
    {
        return match ($this) {
            self::PNG => 'PNG - Portable Network Graphics (lossless)',
            self::JPEG => 'JPEG - Joint Photographic Experts Group (lossy)',
            self::WEBP => 'WebP - Modern web image format',
        };
    }

    /**
     * Check if format supports lossless compression.
     *
     * @return bool True if format is lossless
     */
    public function isLossless(): bool
    {
        return match ($this) {
            self::PNG => true,
            self::JPEG => false,
            self::WEBP => true,  // WebP supports both, but can be lossless
        };
    }

    /**
     * Check if format supports transparency.
     *
     * @return bool True if format supports alpha channel
     */
    public function supportsTransparency(): bool
    {
        return match ($this) {
            self::PNG => true,
            self::JPEG => false,
            self::WEBP => true,
        };
    }

    /**
     * Get default quality for lossy formats.
     *
     * @return int Default quality (0-100), or 0 if not applicable
     */
    public function defaultQuality(): int
    {
        return match ($this) {
            self::PNG => 0,     // PNG doesn't use quality
            self::JPEG => 85,   // Standard JPEG quality
            self::WEBP => 80,   // Standard WebP quality
        };
    }

    /**
     * Create from format string.
     *
     * @param string $format Format string (png, jpeg, webp)
     * @return self Format enum value
     * @throws \ValueError If format is unsupported
     *
     * Example:
     *     $format = ImageFormat::from('png');
     */
    public static function tryFromString(string $format): ?self
    {
        $normalized = strtolower(trim($format));
        return match ($normalized) {
            'png' => self::PNG,
            'jpeg', 'jpg' => self::JPEG,
            'webp' => self::WEBP,
            default => null,
        };
    }

    /**
     * Check if format string is valid.
     *
     * @param string $format Format string to validate
     * @return bool True if format is recognized
     */
    public static function isValid(string $format): bool
    {
        return self::tryFromString($format) !== null;
    }
}
