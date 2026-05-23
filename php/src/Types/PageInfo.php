<?php

declare(strict_types=1);

namespace PdfOxide\Types;

/**
 * Information about a PDF page.
 */
readonly class PageInfo
{
    public function __construct(
        public int $pageIndex,
        public float $width,
        public float $height,
        public int $rotation = 0,
        public int $fontCount = 0,
        public int $imageCount = 0,
        public int $annotationCount = 0
    ) {}

    /**
     * Get page dimensions as Rect.
     */
    public function getDimensions(): Rect
    {
        return new Rect(0, 0, $this->width, $this->height);
    }

    /**
     * Get page size in millimeters.
     */
    public function getSizeMm(): array
    {
        return [
            'width' => round($this->width * 0.352778, 1),
            'height' => round($this->height * 0.352778, 1),
        ];
    }

    /**
     * Get aspect ratio.
     */
    public function getAspectRatio(): float
    {
        return $this->height > 0 ? $this->width / $this->height : 0;
    }

    /**
     * Get total content count.
     */
    public function getTotalContentCount(): int
    {
        return $this->fontCount + $this->imageCount + $this->annotationCount;
    }

    /**
     * Convert to array format.
     */
    public function toArray(): array
    {
        return [
            'page_index' => $this->pageIndex,
            'width' => $this->width,
            'height' => $this->height,
            'rotation' => $this->rotation,
            'font_count' => $this->fontCount,
            'image_count' => $this->imageCount,
            'annotation_count' => $this->annotationCount,
            'size_mm' => $this->getSizeMm(),
            'aspect_ratio' => $this->getAspectRatio(),
        ];
    }
}
