<?php

declare(strict_types=1);

namespace PdfOxide\Builders;

/**
 * Options for PDF search operations.
 *
 * Provides fluent interface for configuring search behavior.
 */
class SearchOptions
{
    private bool $caseSensitive = false;
    private bool $wholeWordsOnly = false;
    private bool $useRegex = false;
    private int $maxResults = 0; // 0 = no limit
    private bool $includeAnnotations = false;
    private bool $searchHiddenText = false;
    private array $pageRange = []; // Empty = all pages

    /**
     * Set case sensitivity for search.
     */
    public function caseSensitive(bool $sensitive): self
    {
        $this->caseSensitive = $sensitive;
        return $this;
    }

    /**
     * Set whether to match whole words only.
     */
    public function wholeWordsOnly(bool $whole): self
    {
        $this->wholeWordsOnly = $whole;
        return $this;
    }

    /**
     * Set whether to use regex patterns.
     */
    public function useRegex(bool $regex): self
    {
        $this->useRegex = $regex;
        return $this;
    }

    /**
     * Set maximum number of results.
     */
    public function maxResults(int $max): self
    {
        $this->maxResults = max(0, $max);
        return $this;
    }

    /**
     * Set whether to search annotations.
     */
    public function includeAnnotations(bool $include): self
    {
        $this->includeAnnotations = $include;
        return $this;
    }

    /**
     * Set whether to search hidden text.
     */
    public function searchHiddenText(bool $search): self
    {
        $this->searchHiddenText = $search;
        return $this;
    }

    /**
     * Set page range for search.
     *
     * @param int $startPage Zero-based start page
     * @param int $endPage Zero-based end page (inclusive)
     */
    public function pageRange(int $startPage, int $endPage): self
    {
        $this->pageRange = [
            'start' => max(0, $startPage),
            'end' => max(0, $endPage),
        ];
        return $this;
    }

    // Getters
    public function isCaseSensitive(): bool { return $this->caseSensitive; }
    public function isWholeWordsOnly(): bool { return $this->wholeWordsOnly; }
    public function isUsingRegex(): bool { return $this->useRegex; }
    public function getMaxResults(): int { return $this->maxResults; }
    public function isIncludingAnnotations(): bool { return $this->includeAnnotations; }
    public function isSearchingHiddenText(): bool { return $this->searchHiddenText; }
    public function getPageRange(): array { return $this->pageRange; }

    /**
     * Convert to array for FFI calls.
     */
    public function toArray(): array
    {
        return [
            'case_sensitive' => $this->caseSensitive,
            'whole_words_only' => $this->wholeWordsOnly,
            'use_regex' => $this->useRegex,
            'max_results' => $this->maxResults,
            'include_annotations' => $this->includeAnnotations,
            'search_hidden_text' => $this->searchHiddenText,
            'page_range' => $this->pageRange,
        ];
    }

    /**
     * Create from array.
     */
    public static function fromArray(array $options): self
    {
        $opts = new self();

        if (isset($options['case_sensitive'])) {
            $opts->caseSensitive($options['case_sensitive']);
        }
        if (isset($options['whole_words_only'])) {
            $opts->wholeWordsOnly($options['whole_words_only']);
        }
        if (isset($options['use_regex'])) {
            $opts->useRegex($options['use_regex']);
        }
        if (isset($options['max_results'])) {
            $opts->maxResults($options['max_results']);
        }
        if (isset($options['include_annotations'])) {
            $opts->includeAnnotations($options['include_annotations']);
        }
        if (isset($options['search_hidden_text'])) {
            $opts->searchHiddenText($options['search_hidden_text']);
        }
        if (isset($options['page_range'])) {
            $opts->pageRange(
                $options['page_range']['start'] ?? 0,
                $options['page_range']['end'] ?? 0
            );
        }

        return $opts;
    }
}
