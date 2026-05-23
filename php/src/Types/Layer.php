<?php

declare(strict_types=1);

namespace PdfOxide\Types;

/**
 * Represents a PDF document layer (Optional Content Group).
 */
readonly class Layer
{
    public function __construct(
        public string $name,
        public bool $visible
    ) {}

    /**
     * Convert to array.
     */
    public function toArray(): array
    {
        return [
            'name' => $this->name,
            'visible' => $this->visible,
        ];
    }
}
