<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\Types\Outline;

/**
 * Manages PDF document outlines (bookmarks/table of contents).
 *
 * Provides access to bookmarks and navigation structure.
 */
class OutlineManager
{
    private FunctionBindings $bindings;
    private CData $handle;
    private ?array $cachedOutlines = null;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
    }

    /**
     * Get total number of outlines (bookmarks).
     */
    public function getCount(): int
    {
        return $this->bindings->pdfDocumentGetOutlineCount($this->handle);
    }

    /**
     * Get outline at index.
     */
    public function get(int $index): Outline
    {
        if ($this->cachedOutlines === null) {
            $this->loadAll();
        }

        if (!isset($this->cachedOutlines[$index])) {
            throw new \OutOfRangeException("Outline index $index out of range");
        }

        return $this->cachedOutlines[$index];
    }

    /**
     * Get all outlines.
     */
    public function getAll(): array
    {
        if ($this->cachedOutlines === null) {
            $this->loadAll();
        }
        return $this->cachedOutlines;
    }

    /**
     * Load all outlines into cache.
     */
    private function loadAll(): void
    {
        $this->cachedOutlines = [];
        $count = $this->getCount();

        for ($i = 0; $i < $count; $i++) {
            $outline = new Outline(
                $this->bindings->pdfDocumentGetOutlineTitle($this->handle, $i),
                $this->bindings->pdfDocumentGetOutlinePage($this->handle, $i),
                $this->bindings->pdfDocumentGetOutlineLevel($this->handle, $i)
            );
            $this->cachedOutlines[] = $outline;
        }
    }

    /**
     * Check if document has outlines.
     */
    public function hasOutlines(): bool
    {
        return $this->getCount() > 0;
    }

    /**
     * Get outlines as array.
     */
    public function toArray(): array
    {
        return array_map(fn(Outline $o) => $o->toArray(), $this->getAll());
    }

    /**
     * Clear cached outlines.
     *
     * @internal
     */
    public function clearCache(): void
    {
        $this->cachedOutlines = null;
    }
}
