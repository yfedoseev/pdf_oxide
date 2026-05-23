<?php

declare(strict_types=1);

namespace PdfOxide\Types;

/**
 * Represents a rectangle defined by position and dimensions.
 */
readonly class Rect
{
    public function __construct(
        public float $x,
        public float $y,
        public float $width,
        public float $height
    ) {}

    /**
     * Check if a point is inside this rectangle.
     */
    public function contains(Point $point): bool
    {
        return $point->x >= $this->x
            && $point->x <= $this->x + $this->width
            && $point->y >= $this->y
            && $point->y <= $this->y + $this->height;
    }

    /**
     * Check if this rectangle intersects with another.
     */
    public function intersects(self $other): bool
    {
        return !($this->x + $this->width < $other->x
            || $other->x + $other->width < $this->x
            || $this->y + $this->height < $other->y
            || $other->y + $other->height < $this->y);
    }

    /**
     * Get the right edge coordinate.
     */
    public function getRight(): float
    {
        return $this->x + $this->width;
    }

    /**
     * Get the bottom edge coordinate.
     */
    public function getBottom(): float
    {
        return $this->y + $this->height;
    }

    /**
     * Convert to array format.
     */
    public function toArray(): array
    {
        return [
            'x' => $this->x,
            'y' => $this->y,
            'width' => $this->width,
            'height' => $this->height,
        ];
    }
}
