<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\Types\PageInfo;
use PdfOxide\Enums\PageSize;

/**
 * Manages page-level operations in PDF documents.
 *
 * Handles page manipulation, sizing, rotation, and information retrieval.
 */
class PageManager
{
    private FunctionBindings $bindings;
    private CData $handle;
    private ?int $cachedPageCount = null;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
    }

    /**
     * Get total page count.
     *
     * @return int Number of pages
     */
    public function count(): int
    {
        if ($this->cachedPageCount !== null) {
            return $this->cachedPageCount;
        }

        $this->cachedPageCount = $this->bindings->pdfDocumentGetPageCount($this->handle);
        return $this->cachedPageCount;
    }

    /**
     * Get information about a page.
     *
     * @param int $pageIndex Zero-based page index
     * @return PageInfo Page information
     */
    public function getInfo(int $pageIndex): PageInfo
    {
        // In a full implementation, this would fetch additional page info
        // For now, return basic info
        return new PageInfo(
            pageIndex: $pageIndex,
            width: 595, // A4 default
            height: 842,
            rotation: 0
        );
    }

    /**
     * Get page dimensions.
     *
     * @param int $pageIndex Zero-based page index
     * @return array Array with 'width' and 'height' in points
     */
    public function getDimensions(int $pageIndex): array
    {
        $info = $this->getInfo($pageIndex);
        return [
            'width' => $info->width,
            'height' => $info->height,
        ];
    }

    /**
     * Get page size in millimeters.
     *
     * @param int $pageIndex Zero-based page index
     * @return array Array with 'width' and 'height' in millimeters
     */
    public function getSizeMm(int $pageIndex): array
    {
        return $this->getInfo($pageIndex)->getSizeMm();
    }

    /**
     * Get page aspect ratio.
     *
     * @param int $pageIndex Zero-based page index
     * @return float Aspect ratio (width / height)
     */
    public function getAspectRatio(int $pageIndex): float
    {
        return $this->getInfo($pageIndex)->getAspectRatio();
    }

    /**
     * Check if page exists.
     *
     * @param int $pageIndex Zero-based page index
     * @return bool True if page exists
     */
    public function exists(int $pageIndex): bool
    {
        return $pageIndex >= 0 && $pageIndex < $this->count();
    }

    /**
     * Validate page index.
     *
     * @param int $pageIndex Zero-based page index
     * @throws \PdfOxide\Exceptions\InvalidStateException if page index invalid
     */
    public function validateIndex(int $pageIndex): void
    {
        if (!$this->exists($pageIndex)) {
            throw new \PdfOxide\Exceptions\InvalidStateException(
                "Invalid page index: {$pageIndex}",
                ['page_index' => $pageIndex, 'total_pages' => $this->count()]
            );
        }
    }

    /**
     * Get all page sizes.
     *
     * @return array Array of page sizes indexed by page number
     */
    public function getAllSizes(): array
    {
        $sizes = [];
        for ($i = 0; $i < $this->count(); $i++) {
            $sizes[$i] = $this->getDimensions($i);
        }
        return $sizes;
    }

    /**
     * Find pages with specific size.
     *
     * @param PageSize $size Page size to match
     * @return array Array of matching page indices
     */
    public function findBySize(PageSize $size): array
    {
        $dims = $size->getDimensions();
        $matches = [];

        for ($i = 0; $i < $this->count(); $i++) {
            $pageDims = $this->getDimensions($i);
            if (abs($pageDims['width'] - $dims['width']) < 1
                && abs($pageDims['height'] - $dims['height']) < 1) {
                $matches[] = $i;
            }
        }

        return $matches;
    }

    /**
     * Check if document is single page.
     *
     * @return bool True if document has only one page
     */
    public function isSinglePage(): bool
    {
        return $this->count() === 1;
    }

    /**
     * Clear the page count cache.
     *
     * @internal
     */
    public function clearCache(): void
    {
        $this->cachedPageCount = null;
    }
}
