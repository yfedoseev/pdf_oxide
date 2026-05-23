<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\Types\RenderedImage;
use PdfOxide\Builders\RenderingOptions;

/**
 * Manages PDF page rendering operations.
 *
 * Converts PDF pages to raster images in various formats (PNG, JPEG, WebP).
 * Supports multiple rendering modes (full page, cropped, zoomed, thumbnail).
 */
class RenderingManager
{
    private FunctionBindings $bindings;
    private CData $handle;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
    }

    /**
     * Render a page to an image.
     *
     * @param int $pageIndex Zero-based page index
     * @param RenderingOptions|null $options Rendering options
     * @return RenderedImage The rendered image
     */
    public function renderPage(int $pageIndex, ?RenderingOptions $options = null): RenderedImage
    {
        $options ??= new RenderingOptions();

        // Call FFI to render page
        $imageHandle = $this->bindings->pdfRenderPage($this->handle, $pageIndex, null);

        return new RenderedImage($imageHandle, $options);
    }

    /**
     * Render a page and save to file.
     *
     * @param int $pageIndex Zero-based page index
     * @param string $filePath Output file path
     * @param RenderingOptions|null $options Rendering options
     * @return void
     */
    public function renderToFile(
        int $pageIndex,
        string $filePath,
        ?RenderingOptions $options = null
    ): void {
        $options ??= new RenderingOptions();

        // Call FFI to render page to file
        $this->bindings->pdfRenderPageToFile($this->handle, $pageIndex, $filePath, null);
    }

    /**
     * Render a page range and save to files.
     *
     * @param int $startPage Start page (zero-based, inclusive)
     * @param int $endPage End page (zero-based, inclusive)
     * @param string $filePrefix File name prefix (will add _1, _2, etc.)
     * @param RenderingOptions|null $options Rendering options
     * @return int Number of pages rendered
     */
    public function renderPageRange(
        int $startPage,
        int $endPage,
        string $filePrefix,
        ?RenderingOptions $options = null
    ): int {
        $options ??= new RenderingOptions();

        // Call FFI to render page range and return count
        return $this->bindings->pdfRenderPageRange($this->handle, $startPage, $endPage, $filePrefix, null);
    }

    /**
     * Render entire document to individual page files.
     *
     * @param string $filePrefix File name prefix
     * @param RenderingOptions|null $options Rendering options
     * @return int Total pages rendered
     */
    public function renderDocument(string $filePrefix, ?RenderingOptions $options = null): int
    {
        $options ??= new RenderingOptions();
        return $this->bindings->pdfRenderDocument($this->handle, $filePrefix, null);
    }

    /**
     * Create a thumbnail of a page.
     *
     * @param int $pageIndex Zero-based page index
     * @param int $maxSize Maximum width/height in pixels
     * @return RenderedImage The thumbnail image
     */
    public function thumbnail(int $pageIndex, int $maxSize = 200): RenderedImage
    {
        $options = new RenderingOptions();
        $imageHandle = $this->bindings->pdfRenderPageThumbnail($this->handle, $pageIndex, $maxSize, null);
        return new RenderedImage($imageHandle, $options);
    }

    /**
     * Render a specific region (crop) of a page.
     *
     * @param int $pageIndex Zero-based page index
     * @param float $x Crop region X
     * @param float $y Crop region Y
     * @param float $width Crop region width
     * @param float $height Crop region height
     * @param RenderingOptions|null $options Rendering options
     * @return RenderedImage The rendered cropped region
     */
    public function renderRegion(
        int $pageIndex,
        float $x,
        float $y,
        float $width,
        float $height,
        ?RenderingOptions $options = null
    ): RenderedImage {
        $options ??= new RenderingOptions();
        $imageHandle = $this->bindings->pdfRenderPageRegion($this->handle, $pageIndex, $x, $y, $width, $height, null);
        return new RenderedImage($imageHandle, $options);
    }

    /**
     * Render a page with zoom level.
     *
     * @param int $pageIndex Zero-based page index
     * @param float $zoomLevel Zoom level (1.0 = 100%, 2.0 = 200%, etc.)
     * @param RenderingOptions|null $options Rendering options
     * @return RenderedImage The rendered image
     */
    public function renderZoom(
        int $pageIndex,
        float $zoomLevel,
        ?RenderingOptions $options = null
    ): RenderedImage {
        $options ??= new RenderingOptions();
        $imageHandle = $this->bindings->pdfRenderPageZoom($this->handle, $pageIndex, $zoomLevel, null);
        return new RenderedImage($imageHandle, $options);
    }

    /**
     * Render a page fitted to specific dimensions.
     *
     * @param int $pageIndex Zero-based page index
     * @param int $fitWidth Target width in pixels
     * @param int $fitHeight Target height in pixels
     * @param RenderingOptions|null $options Rendering options
     * @return RenderedImage The rendered image
     */
    public function renderFit(
        int $pageIndex,
        int $fitWidth,
        int $fitHeight,
        ?RenderingOptions $options = null
    ): RenderedImage {
        $options ??= new RenderingOptions();
        $imageHandle = $this->bindings->pdfRenderPageFit($this->handle, $pageIndex, $fitWidth, $fitHeight, null);
        return new RenderedImage($imageHandle, $options);
    }

    /**
     * Estimate rendering time for a page.
     *
     * @param int $pageIndex Zero-based page index
     * @param RenderingOptions|null $options Rendering options
     * @return int Estimated time in milliseconds
     */
    public function estimateTime(int $pageIndex, ?RenderingOptions $options = null): int
    {
        $options ??= new RenderingOptions();
        return $this->bindings->pdfEstimateRenderTime($this->handle, $pageIndex, null);
    }

    /**
     * Get rendering statistics.
     *
     * @return array Statistics with 'pages_rendered', 'total_time_ms', 'avg_time_ms'
     */
    public function getStatistics(): array
    {
        return $this->bindings->pdfRendererGetStatistics($this->handle);
    }

    /**
     * Reset rendering statistics.
     *
     * @return void
     */
    public function resetStatistics(): void
    {
        $this->bindings->pdfRendererResetStatistics($this->handle);
    }

    /**
     * Get MIME type for image format.
     *
     * @param string $format Image format (png, jpeg, webp)
     * @return string MIME type
     */
    public static function getMimeType(string $format): string
    {
        return match (strtolower($format)) {
            'png' => 'image/png',
            'jpeg', 'jpg' => 'image/jpeg',
            'webp' => 'image/webp',
            default => 'application/octet-stream',
        };
    }

    /**
     * Get file extension for image format.
     *
     * @param string $format Image format
     * @return string File extension (without dot)
     */
    public static function getExtension(string $format): string
    {
        return match (strtolower($format)) {
            'png' => 'png',
            'jpeg', 'jpg' => 'jpg',
            'webp' => 'webp',
            default => 'bin',
        };
    }

    /**
     * Check if rendering is supported.
     *
     * @return bool Always true for now, would check runtime support
     */
    public static function isSupported(): bool
    {
        return true;
    }
}
