<?php

declare(strict_types=1);

namespace PdfOxide\Types;

/**
 * Represents a search result found in a PDF.
 */
readonly class SearchResult
{
    public function __construct(
        public string $text,
        public int $pageIndex,
        public int $position,
        public Rect $boundingBox
    ) {}

    /**
     * Convert to array format.
     */
    public function toArray(): array
    {
        return [
            'text' => $this->text,
            'page_index' => $this->pageIndex,
            'position' => $this->position,
            'bounding_box' => $this->boundingBox->toArray(),
        ];
    }
}
