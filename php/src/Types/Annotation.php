<?php

declare(strict_types=1);

namespace PdfOxide\Types;

/**
 * Represents an annotation found in a PDF.
 */
readonly class Annotation
{
    public function __construct(
        public string $type,
        public string $content
    ) {}

    /**
     * Convert to array format.
     */
    public function toArray(): array
    {
        return [
            'type' => $this->type,
            'content' => $this->content,
        ];
    }
}
