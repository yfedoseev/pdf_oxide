<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;

/**
 * Manages PDF document caching operations.
 *
 * Provides cache control and statistics.
 */
class CacheManager
{
    private FunctionBindings $bindings;
    private CData $handle;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
    }

    /**
     * Clear all cached data for the document.
     */
    public function clear(): void
    {
        $this->bindings->pdfCacheClear($this->handle);
    }

    /**
     * Invalidate cache for a specific page.
     */
    public function invalidatePage(int $pageIndex): void
    {
        $this->bindings->pdfCacheInvalidatePage($this->handle, $pageIndex);
    }

    /**
     * Set the maximum cache size in bytes.
     */
    public function setMaxSize(int $maxSize): void
    {
        $this->bindings->pdfCacheSetMaxSize($this->handle, $maxSize);
    }

    /**
     * Get cache statistics.
     */
    public function getStatistics(): array
    {
        return $this->bindings->pdfCacheGetStatistics($this->handle);
    }

    /**
     * Get cache statistics as JSON string.
     */
    public function getStatisticsJson(): string
    {
        return $this->bindings->pdfCacheGetStatisticsJson($this->handle);
    }

    /**
     * Get cache info summary.
     */
    public function getInfo(): array
    {
        $stats = $this->getStatistics();
        return [
            'size' => $stats['size'] ?? 0,
            'max_size' => $stats['max_size'] ?? 0,
            'items_cached' => $stats['items_cached'] ?? 0,
            'hit_rate' => $stats['hit_rate'] ?? 0.0,
        ];
    }

    /**
     * Check if cache is enabled.
     */
    public function isEnabled(): bool
    {
        $stats = $this->getStatistics();
        return isset($stats['enabled']) ? (bool)$stats['enabled'] : true;
    }

    /**
     * Get current cache size in bytes.
     */
    public function getCurrentSize(): int
    {
        $stats = $this->getStatistics();
        return $stats['size'] ?? 0;
    }

    /**
     * Get maximum cache size in bytes.
     */
    public function getMaxSize(): int
    {
        $stats = $this->getStatistics();
        return $stats['max_size'] ?? 0;
    }

    /**
     * Get cache hit rate (0-1).
     */
    public function getHitRate(): float
    {
        $stats = $this->getStatistics();
        return $stats['hit_rate'] ?? 0.0;
    }

    /**
     * Get number of items in cache.
     */
    public function getItemCount(): int
    {
        $stats = $this->getStatistics();
        return $stats['items_cached'] ?? 0;
    }
}
