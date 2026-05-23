<?php

declare(strict_types=1);

namespace PdfOxide\Types;

/**
 * Represents a 2D point with x and y coordinates.
 */
readonly class Point
{
    public function __construct(
        public float $x,
        public float $y
    ) {}

    /**
     * Calculate distance to another point.
     */
    public function distanceTo(self $other): float
    {
        $dx = $other->x - $this->x;
        $dy = $other->y - $this->y;
        return sqrt($dx * $dx + $dy * $dy);
    }

    /**
     * Convert to array format.
     */
    public function toArray(): array
    {
        return [
            'x' => $this->x,
            'y' => $this->y,
        ];
    }
}
