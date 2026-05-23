<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\Types\Outline;

/**
 * Manages PDF document outlines (bookmarks/table of contents).
 *
 * The v0.3.55 C ABI exposes the outline as ONE JSON-returning entry
 * point — `pdf_document_get_outline()` returns a tree of
 * `{title, dest, children}` records. The per-record accessor
 * functions the pre-v0.3.55 scaffold targeted
 * (`pdf_document_get_outline_count` / `_get_outline_title` etc.) do
 * not exist in the real C ABI; this class loads + flattens the JSON
 * tree once and serves all read paths from the flattened cache.
 */
class OutlineManager
{
    private FunctionBindings $bindings;
    private CData $handle;
    /** @var ?array<int,Outline> */
    private ?array $cachedOutlines = null;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
    }

    /**
     * Get total number of outlines (bookmarks).
     *
     * Counts the flattened depth-first traversal of the outline tree
     * (i.e. nested children count toward the total). Returns 0 for
     * documents without bookmarks.
     */
    public function getCount(): int
    {
        return count($this->getAll());
    }

    /**
     * Get outline at index.
     */
    public function get(int $index): Outline
    {
        $all = $this->getAll();
        if (!isset($all[$index])) {
            throw new \OutOfRangeException("Outline index $index out of range");
        }
        return $all[$index];
    }

    /**
     * Get all outlines.
     *
     * @return array<int,Outline>
     */
    public function getAll(): array
    {
        if ($this->cachedOutlines === null) {
            $this->loadAll();
        }
        return $this->cachedOutlines;
    }

    /**
     * Load all outlines into cache by walking the JSON tree returned
     * from the real C ABI symbol `pdf_document_get_outline()`.
     */
    private function loadAll(): void
    {
        $this->cachedOutlines = [];

        // Fetch raw JSON from the C ABI. Per `outline_to_json` in
        // src/ffi.rs, this is ALWAYS a JSON array (possibly empty);
        // the native side also returns `[]` for error paths so this
        // method never raises in practice.
        try {
            $json = $this->bindings->pdfDocumentGetOutline($this->handle);
        } catch (\PdfOxide\Exceptions\PdfException) {
            // Defensive: even if a future C ABI change starts surfacing
            // errors here, the read-side contract is "empty bookmarks
            // are a normal case" — fall back to [] rather than raise.
            return;
        }

        if ($json === '' || $json === '[]') {
            return;
        }

        try {
            $decoded = json_decode($json, true, 512, JSON_THROW_ON_ERROR);
        } catch (\JsonException) {
            return;
        }
        if (!is_array($decoded)) {
            return;
        }

        $this->flatten($decoded, 0);
    }

    /**
     * Walk the outline tree depth-first, appending {@see Outline}
     * value objects to the cache. Each node's `dest` field is a
     * page index (int) or null for named/unresolved destinations;
     * the binding surfaces it as the page number.
     *
     * @param array<int,array<string,mixed>> $items
     */
    private function flatten(array $items, int $level): void
    {
        foreach ($items as $item) {
            if (!is_array($item)) {
                continue;
            }
            $title = isset($item['title']) ? (string)$item['title'] : '';
            $dest = $item['dest'] ?? null;
            $page = is_int($dest) ? $dest : (is_numeric($dest) ? (int)$dest : 0);

            $this->cachedOutlines[] = new Outline($title, $page, $level);

            if (isset($item['children']) && is_array($item['children']) && !empty($item['children'])) {
                $this->flatten($item['children'], $level + 1);
            }
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
     *
     * @return array<int,array<string,mixed>>
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
