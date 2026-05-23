<?php

declare(strict_types=1);

namespace PdfOxide\Types;

/**
 * Represents an image found in a PDF.
 */
readonly class Image
{
    public function __construct(
        public string $format,
        public int $width,
        public int $height,
        public string $colorspace = '',
        public int $bitsPerComponent = 8
    ) {}

    /**
     * Get the aspect ratio of the image.
     */
    public function getAspectRatio(): float
    {
        return $this->height > 0 ? $this->width / $this->height : 0;
    }

    /**
     * Get total pixels in image.
     */
    public function getPixelCount(): int
    {
        return $this->width * $this->height;
    }

    /**
     * Get file size estimate in bytes (rough calculation).
     */
    public function getEstimatedSize(): int
    {
        return (int)($this->getPixelCount() * $this->bitsPerComponent / 8);
    }

    /**
     * Convert to array format.
     */
    public function toArray(): array
    {
        return [
            'format' => $this->format,
            'width' => $this->width,
            'height' => $this->height,
            'colorspace' => $this->colorspace,
            'bits_per_component' => $this->bitsPerComponent,
            'aspect_ratio' => $this->getAspectRatio(),
            'pixel_count' => $this->getPixelCount(),
            'estimated_size' => $this->getEstimatedSize(),
        ];
    }
}
