<?php

declare(strict_types=1);

namespace PdfOxide\Types;

/**
 * Represents a font found in a PDF.
 */
readonly class Font
{
    public function __construct(
        public string $name,
        public string $type,
        public bool $embedded,
        public string $encoding = '',
        public bool $subset = false,
        public float $size = 0.0
    ) {}

    /**
     * Convert to array format.
     */
    public function toArray(): array
    {
        return [
            'name' => $this->name,
            'type' => $this->type,
            'embedded' => $this->embedded,
            'encoding' => $this->encoding,
            'subset' => $this->subset,
            'size' => $this->size,
        ];
    }
}
