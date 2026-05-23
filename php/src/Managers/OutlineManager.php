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

    /**
     * v0.3.50 #482 — plan a document split along outline bookmarks.
     *
     * Native side does the planning only (per v0.3.50 design — keep
     * the cdylib lean and let bindings do the file I/O). The return
     * value is a JSON envelope: an array of `{title, page_start,
     * page_end, level}` records the caller can feed to a per-section
     * extractor.
     *
     * @param array{min_level?:int,max_level?:int}|null $options
     *        null / empty → defaults.
     *
     * @return array<int,array<string,mixed>>
     */
    public function planSplit(?array $options = null): array
    {
        // Empty-outline documents are a normal case (most PDFs lack
        // bookmarks); per `feedback_extraction_graceful_fallback`,
        // split-planning is NOT a security op — degrade to [].
        // We deliberately DO NOT pre-check via hasOutlines() because
        // the scaffold's outline-count binding targets a function that
        // doesn't exist in the v0.3.55 C ABI; instead we let the FFI
        // call below raise, and convert to [] in the catch.
        $optionsJson = $options === null ? null : json_encode($options, JSON_THROW_ON_ERROR);
        try {
            $json = $this->bindings->pdfDocumentPlanSplitByBookmarks($this->handle, $optionsJson);
        } catch (\PdfOxide\Exceptions\PdfException) {
            // Most failure paths here are "no usable outline" — return
            // empty rather than raise, matching the Python / Java
            // contract.
            return [];
        }
        if ($json === '') {
            return [];
        }
        try {
            $decoded = json_decode($json, true, 512, JSON_THROW_ON_ERROR);
        } catch (\JsonException) {
            return [];
        }
        // Spec is array-shaped (Python returns list of dicts); be
        // tolerant of `{ "sections": [...] }` envelope too.
        if (is_array($decoded) && isset($decoded['sections']) && is_array($decoded['sections'])) {
            return $decoded['sections'];
        }
        return is_array($decoded) ? $decoded : [];
    }
}
