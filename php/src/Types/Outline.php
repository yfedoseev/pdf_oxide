<?php

declare(strict_types=1);

namespace PdfOxide\Types;

/**
 * Represents a PDF document outline (bookmark).
 */
readonly class Outline
{
    public function __construct(
        public string $title,
        public int $pageIndex,
        public int $level
    ) {}

    /**
     * Convert to array.
     */
    public function toArray(): array
    {
        return [
            'title' => $this->title,
            'page_index' => $this->pageIndex,
            'level' => $this->level,
        ];
    }
}
