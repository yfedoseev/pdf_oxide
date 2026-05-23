<?php

declare(strict_types=1);

namespace PdfOxide\Rendering;

use PdfOxide\Enums\ImageFormat;
use PdfOxide\Types\RenderedImage;
use PdfOxide\PdfPage;
use PdfOxide\Builders\RenderingOptions;

/**
 * Page renderer for converting PDF pages to images.
 *
 * Provides a fluent interface for configuring and rendering PDF pages to various image
 * formats with options for DPI, quality, and rendering modes.
 *
 * Example:
 *     $renderer = new PageRenderer($page);
 *     $image = $renderer
 *         ->setDpi(300)
 *         ->setFormat(ImageFormat::PNG)
 *         ->render();
 *     $image->saveToFile('page.png');
 *
 *     // Render with fit to dimensions
 *     $image = $renderer
 *         ->setDpi(150)
 *         ->renderFit(800, 600);
 *
 *     // Create thumbnail
 *     $thumb = $renderer->thumbnail(100);
 *
 * @since 0.4.0
 */
class PageRenderer
{
    private ImageFormat $format = ImageFormat::PNG;
    private int $dpi = 150;
    private int $quality = 85;
    private bool $antialiasing = true;
    private ?string $backgroundColor = null;

    /**
     * Create a page renderer for the given page.
     *
     * @param PdfPage $page The page to render
     */
    public function __construct(
        private readonly PdfPage $page
    ) {}

    /**
     * Set output image format.
     *
     * @param ImageFormat $format The desired image format
     * @return self Fluent interface
     *
     * Example:
     *     $renderer->setFormat(ImageFormat::JPEG);
     */
    public function setFormat(ImageFormat $format): self
    {
        $this->format = $format;
        return $this;
    }

    /**
     * Set rendering DPI (dots per inch).
     *
     * @param int $dpi DPI value (1-600, clamped to valid range)
     * @return self Fluent interface
     * @throws \InvalidArgumentException If DPI is invalid
     *
     * Higher DPI values produce higher resolution images.
     * Valid range: 1-600 (typical values: 72, 96, 150, 300, 600)
     *
     * Example:
     *     $renderer->setDpi(300);  // High resolution
     */
    public function setDpi(int $dpi): self
    {
        if ($dpi < 1 || $dpi > 600) {
            throw new \InvalidArgumentException(
                "DPI must be between 1 and 600, got {$dpi}"
            );
        }
        $this->dpi = $dpi;
        return $this;
    }

    /**
     * Set quality for lossy formats (JPEG, WebP).
     *
     * @param int $quality Quality value (1-100)
     * @return self Fluent interface
     * @throws \InvalidArgumentException If quality is invalid
     *
     * Only applies to JPEG and WebP formats.
     * PNG quality is not applicable (lossless format).
     * Valid range: 1-100
     *
     * Example:
     *     $renderer->setFormat(ImageFormat::JPEG)->setQuality(90);
     */
    public function setQuality(int $quality): self
    {
        if ($quality < 1 || $quality > 100) {
            throw new \InvalidArgumentException(
                "Quality must be between 1 and 100, got {$quality}"
            );
        }
        $this->quality = $quality;
        return $this;
    }

    /**
     * Set antialiasing enabled/disabled.
     *
     * @param bool $enabled Enable or disable antialiasing
     * @return self Fluent interface
     *
     * Antialiasing produces smoother text and line rendering but may be slower.
     *
     * Example:
     *     $renderer->setAntialiasing(false);  // Faster but less smooth
     */
    public function setAntialiasing(bool $enabled): self
    {
        $this->antialiasing = $enabled;
        return $this;
    }

    /**
     * Set background color.
     *
     * @param ?string $color Hex color code (e.g., '#FFFFFF'), or null for transparent
     * @return self Fluent interface
     * @throws \InvalidArgumentException If color format is invalid
     *
     * Example:
     *     $renderer->setBackgroundColor('#FFFFFF');  // White background
     *     $renderer->setBackgroundColor(null);        // Transparent
     */
    public function setBackgroundColor(?string $color): self
    {
        if ($color !== null) {
            if (!preg_match('/^#[0-9A-Fa-f]{6}$/', $color)) {
                throw new \InvalidArgumentException(
                    "Invalid color format: {$color}. Use hex format like #FFFFFF"
                );
            }
        }
        $this->backgroundColor = $color;
        return $this;
    }

    /**
     * Get the rendering options builder configured with current settings.
     *
     * @return RenderingOptions Configured rendering options
     */
    private function buildRenderingOptions(): RenderingOptions
    {
        $options = new RenderingOptions();

        $options->dpi($this->dpi);
        $options->imageFormat($this->format->value);
        $options->jpegQuality($this->quality);
        $options->webpQuality($this->quality);

        if (!$this->antialiasing) {
            $options->antialiasing(false);
        }

        if ($this->backgroundColor !== null) {
            $options->backgroundColor($this->backgroundColor);
        }

        return $options;
    }

    /**
     * Render the page to an image.
     *
     * @return RenderedImage The rendered image
     * @throws \RuntimeException If rendering fails
     *
     * Example:
     *     $image = $renderer->render();
     *     $image->saveToFile('page.png');
     */
    public function render(): RenderedImage
    {
        try {
            $doc = $this->page->getDocument();
            $options = $this->buildRenderingOptions();

            return $doc->rendering()->renderPage(
                $this->page->getIndex(),
                $options
            );
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "Failed to render page {$this->page->getIndex()}: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Render the page fit to specified dimensions.
     *
     * Scales the page to fit within the specified width and height while preserving
     * aspect ratio. The rendered image dimensions may be smaller than specified.
     *
     * @param int $maxWidth Maximum width in pixels
     * @param int $maxHeight Maximum height in pixels
     * @return RenderedImage The fitted and rendered image
     * @throws \InvalidArgumentException If dimensions are invalid
     * @throws \RuntimeException If rendering fails
     *
     * Example:
     *     $image = $renderer->renderFit(800, 600);  // Fit to 800x600
     */
    public function renderFit(int $maxWidth, int $maxHeight): RenderedImage
    {
        if ($maxWidth <= 0 || $maxHeight <= 0) {
            throw new \InvalidArgumentException(
                "Width and height must be positive, got {$maxWidth}x{$maxHeight}"
            );
        }

        try {
            $doc = $this->page->getDocument();
            $options = $this->buildRenderingOptions();

            return $doc->rendering()->renderPageFit(
                $this->page->getIndex(),
                $maxWidth,
                $maxHeight,
                $options
            );
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "Failed to render page fit: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Create a thumbnail of the page.
     *
     * Renders a thumbnail with maximum dimension of specified size.
     * Aspect ratio is preserved.
     *
     * @param int $maxSize Maximum width or height in pixels
     * @return RenderedImage The thumbnail image
     * @throws \InvalidArgumentException If size is invalid
     * @throws \RuntimeException If rendering fails
     *
     * Example:
     *     $thumb = $renderer->thumbnail(100);  // 100px thumbnail
     */
    public function thumbnail(int $maxSize): RenderedImage
    {
        if ($maxSize <= 0) {
            throw new \InvalidArgumentException(
                "Thumbnail size must be positive, got {$maxSize}"
            );
        }

        try {
            $doc = $this->page->getDocument();
            $options = $this->buildRenderingOptions();

            return $doc->rendering()->renderPageThumbnail(
                $this->page->getIndex(),
                $maxSize,
                $options
            );
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "Failed to create thumbnail: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Render and save page to a file.
     *
     * Renders the page and saves it directly to the specified file path.
     * The file format is determined by the configured image format.
     *
     * @param string $filePath Path where to save the rendered image
     * @return void
     * @throws \InvalidArgumentException If file path is invalid
     * @throws \RuntimeException If rendering or file saving fails
     *
     * Example:
     *     $renderer->setFormat(ImageFormat::PNG)->renderToFile('page.png');
     */
    public function renderToFile(string $filePath): void
    {
        if (empty($filePath)) {
            throw new \InvalidArgumentException('File path cannot be empty');
        }

        $dir = dirname($filePath);
        if (!is_dir($dir)) {
            throw new \InvalidArgumentException(
                "Directory does not exist: {$dir}"
            );
        }

        if (!is_writable($dir)) {
            throw new \InvalidArgumentException(
                "Directory is not writable: {$dir}"
            );
        }

        try {
            $image = $this->render();
            $image->saveToFile($filePath);
        } catch (\Exception $e) {
            throw new \RuntimeException(
                "Failed to save rendered page to {$filePath}: {$e->getMessage()}",
                0,
                $e
            );
        }
    }

    /**
     * Get the currently configured format.
     *
     * @return ImageFormat The output format
     */
    public function getFormat(): ImageFormat
    {
        return $this->format;
    }

    /**
     * Get the currently configured DPI.
     *
     * @return int DPI value
     */
    public function getDpi(): int
    {
        return $this->dpi;
    }

    /**
     * Get the currently configured quality.
     *
     * @return int Quality value (1-100)
     */
    public function getQuality(): int
    {
        return $this->quality;
    }

    /**
     * Check if antialiasing is enabled.
     *
     * @return bool True if antialiasing is enabled
     */
    public function isAntialiasingEnabled(): bool
    {
        return $this->antialiasing;
    }

    /**
     * Get the configured background color.
     *
     * @return ?string Hex color or null
     */
    public function getBackgroundColor(): ?string
    {
        return $this->backgroundColor;
    }

    /**
     * Get the page being rendered.
     *
     * @return PdfPage The page reference
     */
    public function getPage(): PdfPage
    {
        return $this->page;
    }

    /**
     * Get current configuration as array.
     *
     * @return array Configuration array with all current settings
     */
    public function toArray(): array
    {
        return [
            'format' => $this->format->value,
            'dpi' => $this->dpi,
            'quality' => $this->quality,
            'antialiasing' => $this->antialiasing,
            'backgroundColor' => $this->backgroundColor,
            'pageIndex' => $this->page->getIndex(),
        ];
    }
}
