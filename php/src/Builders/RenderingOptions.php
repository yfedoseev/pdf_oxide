<?php

declare(strict_types=1);

namespace PdfOxide\Builders;

/**
 * Options for PDF page rendering operations.
 *
 * Provides fluent interface for configuring rendering behavior.
 */
class RenderingOptions
{
    private int $dpi = 150;
    private string $imageFormat = 'png';
    private int $jpegQuality = 85;
    private int $webpQuality = 80;
    private string $colorSpace = 'srgb';
    private bool $antialias = true;
    private string $backgroundColor = '#FFFFFF';
    private int $maxWidth = 0; // 0 = no limit
    private int $maxHeight = 0; // 0 = no limit
    private bool $applyColorProfile = true;

    /**
     * Set DPI for rendering.
     */
    public function dpi(int $dpi): self
    {
        $this->dpi = max(50, min(600, $dpi));
        return $this;
    }

    /**
     * Set output image format.
     */
    public function imageFormat(string $format): self
    {
        $format = strtolower($format);
        if (!in_array($format, ['png', 'jpeg', 'webp'])) {
            throw new \ValueError("Unsupported image format: {$format}");
        }
        $this->imageFormat = $format;
        return $this;
    }

    /**
     * Set JPEG quality (0-100).
     */
    public function jpegQuality(int $quality): self
    {
        $this->jpegQuality = max(0, min(100, $quality));
        return $this;
    }

    /**
     * Set WebP quality (0-100).
     */
    public function webpQuality(int $quality): self
    {
        $this->webpQuality = max(0, min(100, $quality));
        return $this;
    }

    /**
     * Set color space.
     */
    public function colorSpace(string $space): self
    {
        $space = strtolower($space);
        if (!in_array($space, ['srgb', 'device_rgb', 'linear_rgb'])) {
            throw new \ValueError("Unsupported color space: {$space}");
        }
        $this->colorSpace = $space;
        return $this;
    }

    /**
     * Set whether to use antialiasing.
     */
    public function antialias(bool $antialias): self
    {
        $this->antialias = $antialias;
        return $this;
    }

    /**
     * Set background color.
     */
    public function backgroundColor(string $color): self
    {
        $this->backgroundColor = $color;
        return $this;
    }

    /**
     * Set maximum width for rendering.
     */
    public function maxWidth(int $width): self
    {
        $this->maxWidth = max(0, $width);
        return $this;
    }

    /**
     * Set maximum height for rendering.
     */
    public function maxHeight(int $height): self
    {
        $this->maxHeight = max(0, $height);
        return $this;
    }

    /**
     * Set whether to apply color profile.
     */
    public function applyColorProfile(bool $apply): self
    {
        $this->applyColorProfile = $apply;
        return $this;
    }

    // Getters
    public function getDpi(): int { return $this->dpi; }
    public function getImageFormat(): string { return $this->imageFormat; }
    public function getJpegQuality(): int { return $this->jpegQuality; }
    public function getWebpQuality(): int { return $this->webpQuality; }
    public function getColorSpace(): string { return $this->colorSpace; }
    public function isAntialiasing(): bool { return $this->antialias; }
    public function getBackgroundColor(): string { return $this->backgroundColor; }
    public function getMaxWidth(): int { return $this->maxWidth; }
    public function getMaxHeight(): int { return $this->maxHeight; }
    public function isApplyingColorProfile(): bool { return $this->applyColorProfile; }

    /**
     * Convert to array for FFI calls.
     */
    public function toArray(): array
    {
        return [
            'dpi' => $this->dpi,
            'image_format' => $this->imageFormat,
            'jpeg_quality' => $this->jpegQuality,
            'webp_quality' => $this->webpQuality,
            'color_space' => $this->colorSpace,
            'antialias' => $this->antialias,
            'background_color' => $this->backgroundColor,
            'max_width' => $this->maxWidth,
            'max_height' => $this->maxHeight,
            'apply_color_profile' => $this->applyColorProfile,
        ];
    }

    /**
     * Create from array.
     */
    public static function fromArray(array $options): self
    {
        $opts = new self();

        if (isset($options['dpi'])) {
            $opts->dpi($options['dpi']);
        }
        if (isset($options['image_format'])) {
            $opts->imageFormat($options['image_format']);
        }
        if (isset($options['jpeg_quality'])) {
            $opts->jpegQuality($options['jpeg_quality']);
        }
        if (isset($options['webp_quality'])) {
            $opts->webpQuality($options['webp_quality']);
        }
        if (isset($options['color_space'])) {
            $opts->colorSpace($options['color_space']);
        }
        if (isset($options['antialias'])) {
            $opts->antialias($options['antialias']);
        }
        if (isset($options['background_color'])) {
            $opts->backgroundColor($options['background_color']);
        }
        if (isset($options['max_width'])) {
            $opts->maxWidth($options['max_width']);
        }
        if (isset($options['max_height'])) {
            $opts->maxHeight($options['max_height']);
        }
        if (isset($options['apply_color_profile'])) {
            $opts->applyColorProfile($options['apply_color_profile']);
        }

        return $opts;
    }
}
