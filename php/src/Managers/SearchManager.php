<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\Types\SearchResult;
use PdfOxide\Builders\SearchOptions;

/**
 * Manages full-text search operations in PDF documents.
 *
 * Provides advanced search capabilities with filtering and result processing.
 */
class SearchManager
{
    private FunctionBindings $bindings;
    private CData $handle;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
    }

    /**
     * Search for text in the entire document.
     *
     * @param string $query The search query
     * @param SearchOptions|null $options Search options
     * @return SearchResult[] Array of search results
     */
    public function search(string $query, ?SearchOptions $options = null): array
    {
        $options ??= new SearchOptions();

        $resultsHandle = $this->bindings->pdfDocumentSearchAll(
            $this->handle,
            $query,
            $options->isCaseSensitive()
        );

        try {
            return $this->parseSearchResults($resultsHandle, $options);
        } finally {
            $this->bindings->oxideSearchResultFree($resultsHandle);
        }
    }

    /**
     * Search for text in a specific page.
     *
     * @param string $query The search query
     * @param int $pageIndex Zero-based page index
     * @param SearchOptions|null $options Search options
     * @return SearchResult[] Array of search results
     */
    public function searchPage(string $query, int $pageIndex, ?SearchOptions $options = null): array
    {
        $options ??= new SearchOptions();

        $resultsHandle = $this->bindings->pdfDocumentSearchPage(
            $this->handle,
            $query,
            $pageIndex,
            $options->isCaseSensitive()
        );

        try {
            return $this->parseSearchResults($resultsHandle, $options);
        } finally {
            $this->bindings->oxideSearchResultFree($resultsHandle);
        }
    }

    /**
     * Count search results without returning them.
     *
     * @param string $query The search query
     * @param SearchOptions|null $options Search options
     * @return int Number of matches
     */
    public function count(string $query, ?SearchOptions $options = null): int
    {
        $results = $this->search($query, $options);
        return count($results);
    }

    /**
     * Check if a word exists in the document.
     *
     * @param string $word The word to search for
     * @param bool $caseSensitive Case sensitivity
     * @return bool True if word is found
     */
    public function contains(string $word, bool $caseSensitive = false): bool
    {
        $options = (new SearchOptions())
            ->wholeWordsOnly(true)
            ->caseSensitive($caseSensitive)
            ->maxResults(1);

        $results = $this->search($word, $options);
        return count($results) > 0;
    }

    /**
     * Get search results grouped by page.
     *
     * @param string $query The search query
     * @param SearchOptions|null $options Search options
     * @return array Results grouped by page index
     */
    public function searchGroupedByPage(string $query, ?SearchOptions $options = null): array
    {
        $results = $this->search($query, $options);
        $grouped = [];

        foreach ($results as $result) {
            if (!isset($grouped[$result->pageIndex])) {
                $grouped[$result->pageIndex] = [];
            }
            $grouped[$result->pageIndex][] = $result;
        }

        return $grouped;
    }

    /**
     * Get unique search terms from a page.
     *
     * @param string $query The search query (can be multiple words)
     * @param int $pageIndex Zero-based page index
     * @return array Unique terms found
     */
    public function uniqueTerms(string $query, int $pageIndex): array
    {
        $words = preg_split('/\s+/', trim($query), -1, PREG_SPLIT_NO_EMPTY);
        $foundTerms = [];

        foreach ($words as $word) {
            if ($this->searchPage($word, $pageIndex)) {
                $foundTerms[] = $word;
            }
        }

        return $foundTerms;
    }

    /**
     * Parse search results with filtering.
     *
     * @internal
     */
    private function parseSearchResults(CData $resultsHandle, SearchOptions $options): array
    {
        $allResults = [];
        $count = $this->bindings->oxideSearchResultCount($resultsHandle);

        for ($i = 0; $i < $count; $i++) {
            $bbox = $this->bindings->oxideSearchResultGetBbox($resultsHandle, $i);
            $allResults[] = new SearchResult(
                $this->bindings->oxideSearchResultGetText($resultsHandle, $i),
                $this->bindings->oxideSearchResultGetPage($resultsHandle, $i),
                $this->bindings->oxideSearchResultGetPosition($resultsHandle, $i),
                new \PdfOxide\Types\Rect(
                    $bbox['x'],
                    $bbox['y'],
                    $bbox['width'],
                    $bbox['height']
                )
            );
        }

        // Apply filtering
        $results = $this->applyFilters($allResults, $options);

        // Apply limits
        $maxResults = $options->getMaxResults();
        if ($maxResults > 0) {
            $results = array_slice($results, 0, $maxResults);
        }

        return $results;
    }

    /**
     * Apply search option filters to results.
     *
     * @internal
     */
    private function applyFilters(array $results, SearchOptions $options): array
    {
        $pageRange = $options->getPageRange();
        if (!empty($pageRange)) {
            $results = array_filter($results, function ($result) use ($pageRange) {
                return $result->pageIndex >= $pageRange['start']
                    && $result->pageIndex <= $pageRange['end'];
            });
        }

        return $results;
    }
}
